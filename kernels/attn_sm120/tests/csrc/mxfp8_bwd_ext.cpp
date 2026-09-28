// pybind glue for the MXFP8 attention backward (nvcc-built kernels live in
// mxfp8_bwd_kernel.cu). Single op:
//   mxfp8_bwd(Qd,Kd,Vd,Dd,Qt,Kt,Dt, sfQ,sfK,sfV,sfD,sfQt,sfKt,sfDt, lse,delta, sm_scale)
//     -> (dQ, dK, dV) fp32 [H,S,128]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void mxfp8_bwd_launch(
    const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, float*, float*, int, int, float, int, uintptr_t);
extern "C" void mxfp8_dk_ws_launch(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, int, int, float, uintptr_t);
extern uint8_t* g_dvdk_dbg;
extern "C" void quant_nat_launch(const void*, void*, void*, void*, int, int, int, uintptr_t);
extern "C" void quant_trn_launch(const void*, void*, void*, void*, int, int, int, uintptr_t);
extern "C" void mxfp8_dvdk_ws_launch(
    const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, float*, int, int, float, uintptr_t);
extern "C" void mxfp8_dv_ws_launch(
    const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, float*, int, int, float, uintptr_t);

static std::vector<at::Tensor> mxfp8_bwd(
    at::Tensor Qd, at::Tensor Kd, at::Tensor Vd, at::Tensor Dd,
    at::Tensor Qt, at::Tensor Kt, at::Tensor Dt,
    at::Tensor sfQ, at::Tensor sfK, at::Tensor sfV, at::Tensor sfD,
    at::Tensor sfQt, at::Tensor sfKt, at::Tensor sfDt,
    at::Tensor lse, at::Tensor delta, double sm_scale, int64_t use_dk_ws) {
  TORCH_CHECK(Qd.is_cuda() && Qd.dtype() == at::kByte && Qd.is_contiguous());
  const int H = Qd.size(0), S = Qd.size(1), D = Qd.size(2);
  TORCH_CHECK(D == 128 && S % 128 == 0, "mxfp8_bwd: need head_dim=128, S%128==0");
  auto opts = at::TensorOptions().dtype(at::kFloat).device(Qd.device());
  at::Tensor dQ = at::empty({H, S, D}, opts);
  at::Tensor dK = at::empty({H, S, D}, opts);
  at::Tensor dV = at::empty({H, S, D}, opts);
  mxfp8_bwd_launch(Qd.data_ptr(), Kd.data_ptr(), Vd.data_ptr(), Dd.data_ptr(),
                   Qt.data_ptr(), Kt.data_ptr(), Dt.data_ptr(),
                   sfQ.data_ptr(), sfK.data_ptr(), sfV.data_ptr(), sfD.data_ptr(),
                   sfQt.data_ptr(), sfKt.data_ptr(), sfDt.data_ptr(),
                   lse.data_ptr<float>(), delta.data_ptr<float>(),
                   dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
                   S, H, float(sm_scale), int(use_dk_ws),
                   uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dQ, dK, dV};
}

static std::vector<at::Tensor> quant_op(at::Tensor x, bool transposed) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.dim() == 3 && x.size(2) == 128);
  auto xc = x.contiguous();
  const int H = xc.size(0), L = xc.size(1);
  const int S = (L + 127) / 128 * 128;
  auto u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
  at::Tensor q, sf_raw, sf_pack;
  if (!transposed) {
    q = at::empty({H, S, 128}, u8);
    sf_raw = at::empty({H, S, 4}, u8);
    sf_pack = at::empty({H, S / 128, 512}, u8);
    quant_nat_launch(xc.data_ptr(), q.data_ptr(), sf_raw.data_ptr(), sf_pack.data_ptr(),
                     H, S, L, uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  } else {
    q = at::empty({H, 128, S}, u8);
    sf_raw = at::empty({H, 128, S / 32}, u8);
    sf_pack = at::empty({H, S / 128, 512}, u8);
    quant_trn_launch(xc.data_ptr(), q.data_ptr(), sf_raw.data_ptr(), sf_pack.data_ptr(),
                     H, S, L, uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, sf_raw, sf_pack};
}

extern "C" void s3_ragged_mx_launch(
    const void* Qd, const void* Kd, const void* Vt,
    const void* sfQ, const void* sfK, const void* sfV,
    int Sq_pad, int Sk_pad, int Hq, int Hkv, int group,
    float sm_scale, int causal,
    float* out_O, float* out_lse, float* out_l,
    int* work_indptr, int* head_indices, int* qo_tile_indices,
    int* qo_indptr, int* kv_indptr, int* qo_lens, int* kv_lens, int* batch_indices,
    int num_sm, uintptr_t stream_);

// Ragged MXFP8 prefill fwd with a CACHED persistent-scheduler plan (the plan is
// shape-deterministic; the old .so re-planned + did 8 pageable H2D copies per call,
// which dominated CPU time at long L). B=1 only: indptr=[0,S], lens=[L].
namespace {
struct RaggedPlan {
  at::Tensor work_indptr, head_indices, qo_tile_indices, qo_indptr, kv_indptr,
             qo_lens, kv_lens, batch_indices;
};
static std::unordered_map<int64_t, RaggedPlan> g_plan_cache;
static int g_num_sm = 0;
}  // namespace

static std::vector<at::Tensor> fwd_attn(
    at::Tensor QdT, at::Tensor KdT, at::Tensor Vt,   // [S,H,D] [S,H,D] [H,D,S] uint8
    at::Tensor sfQ, at::Tensor sfK, at::Tensor sfV,  // packed ue8m0
    int64_t L_, double sm_scale) {
  const int S = QdT.size(0), Hq = QdT.size(1), D = QdT.size(2);
  const int L = int(L_);
  TORCH_CHECK(D == 128 && S % 128 == 0);
  if (g_num_sm == 0) cudaDeviceGetAttribute(&g_num_sm, cudaDevAttrMultiProcessorCount, QdT.get_device());
  const int64_t key = (int64_t(S) << 40) ^ (int64_t(L) << 16) ^ (int64_t(Hq) << 1);
  auto it = g_plan_cache.find(key);
  if (it == g_plan_cache.end()) {
    constexpr int kBM = 128, kBN = 64;
    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
    const int nqt = cdiv(L, kBM), nkt = cdiv(L, kBN);
    struct W { int qhead, qtile; long cost; };
    std::vector<W> works;
    works.reserve(size_t(nqt) * Hq);
    for (int hq = 0; hq < Hq; ++hq)
      for (int qt = 0; qt < nqt; ++qt) works.push_back({hq, qt, nkt});
    std::stable_sort(works.begin(), works.end(), [](const W& a, const W& b) { return a.cost > b.cost; });
    std::vector<long> load(g_num_sm, 0);
    std::vector<std::vector<W>> cta(g_num_sm);
    for (auto& w : works) {
      int c = int(std::min_element(load.begin(), load.end()) - load.begin());
      cta[c].push_back(w); load[c] += w.cost;
    }
    std::vector<int> work_indptr(g_num_sm + 1, 0), head_i, qtile_i, qo_ip_v, kv_ip_v, qo_l_v, kv_l_v, batch_i;
    for (int c = 0; c < g_num_sm; ++c) work_indptr[c + 1] = work_indptr[c] + int(cta[c].size());
    for (int c = 0; c < g_num_sm; ++c) for (auto& w : cta[c]) {
      head_i.push_back(w.qhead); qtile_i.push_back(w.qtile);
      qo_ip_v.push_back(0); kv_ip_v.push_back(0); qo_l_v.push_back(L); kv_l_v.push_back(L); batch_i.push_back(0);
    }
    auto up = [&](const std::vector<int>& v) {
      auto t = at::empty({int64_t(std::max<size_t>(1, v.size()))},
                         at::TensorOptions().dtype(at::kInt).device(QdT.device()));
      if (!v.empty())
        C10_CUDA_CHECK(cudaMemcpy(t.data_ptr<int>(), v.data(), v.size() * 4, cudaMemcpyHostToDevice));
      return t;
    };
    RaggedPlan pl{up(work_indptr), up(head_i), up(qtile_i), up(qo_ip_v),
                  up(kv_ip_v),      up(qo_l_v),  up(kv_l_v),  up(batch_i)};
    it = g_plan_cache.emplace(key, std::move(pl)).first;
  }
  const RaggedPlan& pl = it->second;
  auto opts_f = at::TensorOptions().dtype(at::kFloat).device(QdT.device());
  // pad q-tiles are skipped by the plan; zero-fill so bwd's delta/lse never see garbage bits
  at::Tensor O = at::zeros({S, Hq, D}, opts_f);
  at::Tensor LSE = at::zeros({Hq, S}, opts_f);
  at::Tensor Lout = at::empty({Hq, S}, opts_f);
  s3_ragged_mx_launch(QdT.data_ptr(), KdT.data_ptr(), Vt.data_ptr(),
                      sfQ.data_ptr(), sfK.data_ptr(), sfV.data_ptr(),
                      S, S, Hq, Hq, 1, float(sm_scale), 0,
                      O.data_ptr<float>(), LSE.data_ptr<float>(), Lout.data_ptr<float>(),
                      pl.work_indptr.data_ptr<int>(), pl.head_indices.data_ptr<int>(),
                      pl.qo_tile_indices.data_ptr<int>(), pl.qo_indptr.data_ptr<int>(),
                      pl.kv_indptr.data_ptr<int>(), pl.qo_lens.data_ptr<int>(),
                      pl.kv_lens.data_ptr<int>(), pl.batch_indices.data_ptr<int>(),
                      g_num_sm, uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {O, LSE};
}

// Fused fwd preparation: quant q/k/v (natural+transposed) and emit the
// [S,H,D] transposed copies the ragged fwd kernel wants. One pybind crossing.
// Returns: qdT, kdT, vt, rq, rk, rvt, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt
static std::vector<at::Tensor> fwd_prep(at::Tensor q, at::Tensor k, at::Tensor v) {
  auto qn = quant_op(q, false);   // qd, rq, sfq
  auto kn = quant_op(k, false);
  auto vn = quant_op(v, false);
  auto qt_ = quant_op(q, true);   // qt, rqt, sfqt
  auto kt_ = quant_op(k, true);
  auto vt_ = quant_op(v, true);
  at::Tensor qdT = qn[0].permute({1, 0, 2}).contiguous();
  at::Tensor kdT = kn[0].permute({1, 0, 2}).contiguous();
  return {qdT, kdT, vt_[0], qn[1], kn[1], vt_[1],
          qn[0], kn[0], vn[0], qt_[0], kt_[0],
          qn[2], kn[2], vn[2], qt_[2], kt_[2], vt_[2]};
}

// Fused bwd: quant dO, delta, mxfp8_bwd, post-scale + slice + bf16 cast.
// o_shd: [S,H,D] fp32 raw fwd output. Returns dq/dk/dv bf16 [H,L,D].
static std::vector<at::Tensor> bwd_full(
    at::Tensor do_, at::Tensor Qd, at::Tensor Kd, at::Tensor Vd,
    at::Tensor Qt, at::Tensor Kt,
    at::Tensor sfQ, at::Tensor sfK, at::Tensor sfV,
    at::Tensor sfQt, at::Tensor sfKt,
    at::Tensor lse, at::Tensor o_shd, double sm_scale, int64_t mode) {
  const int H = do_.size(0), L = do_.size(1);
  const int S = Qd.size(1);
  auto dn = quant_op(do_, false);          // dd, _, sfd
  auto dt_ = quant_op(do_, true);          // dt, _, sfdt
  at::Tensor dop = at::constant_pad_nd(do_, {0, 0, 0, S - L});        // [H,S,D]
  at::Tensor delta = (o_shd * dop.permute({1, 0, 2})).sum(-1).t().contiguous();
  at::Tensor lse_ = lse;
  if (lse_.size(-1) != S)
    lse_ = at::constant_pad_nd(lse_, {0, S - (int)lse_.size(-1)});
  lse_ = lse_.contiguous();
  auto grads = mxfp8_bwd(Qd, Kd, Vd, dn[0], Qt, Kt, dt_[0],
                         sfQ, sfK, sfV, dn[2], sfQt, sfKt, dt_[2],
                         lse_, delta, sm_scale, mode);
  // kernels fold sm_scale into P only; dQ/dK need the exact post-scale
  grads[0].mul_(sm_scale);
  grads[1].mul_(sm_scale);
  return {grads[0].narrow(1, 0, L).to(at::kBFloat16),
          grads[1].narrow(1, 0, L).to(at::kBFloat16),
          grads[2].narrow(1, 0, L).to(at::kBFloat16)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_nat", [](at::Tensor x) { return quant_op(x, false); }, "fused mxfp8 natural quant (pads to S)");
  m.def("fwd_prep", &fwd_prep, "fused fwd quant+layout prep");
  m.def("fwd_attn", &fwd_attn, "ragged mxfp8 fwd, cached plan, packed SF");
  m.def("bwd_full", &bwd_full, "fused bwd: quant dO + delta + kernels + postprocess");
  m.def("quant_trn", [](at::Tensor x) { return quant_op(x, true); }, "fused mxfp8 transposed quant (pads to S)"); m.def("mxfp8_bwd", &mxfp8_bwd, "mxfp8 attention bwd");
  m.def("dvdk_ws", [](at::Tensor Kd, at::Tensor Vd, at::Tensor Qd, at::Tensor Dd,
                      at::Tensor Qt, at::Tensor Dt, at::Tensor sfK, at::Tensor sfV,
                      at::Tensor sfQ, at::Tensor sfD, at::Tensor sfQt, at::Tensor sfDt,
                      at::Tensor lse, at::Tensor delta, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    auto f32 = at::TensorOptions().dtype(at::kFloat).device(Kd.device());
    at::Tensor dV = at::empty({H, S, 128}, f32);
    at::Tensor dK = at::empty({H, S, 128}, f32);
    mxfp8_dvdk_ws_launch(Kd.data_ptr(), Vd.data_ptr(), Qd.data_ptr(), Dd.data_ptr(),
                         Qt.data_ptr(), Dt.data_ptr(), sfK.data_ptr(), sfV.data_ptr(),
                         sfQ.data_ptr(), sfD.data_ptr(), sfQt.data_ptr(), sfDt.data_ptr(),
                         lse.data_ptr<float>(), delta.data_ptr<float>(),
                         dV.data_ptr<float>(), dK.data_ptr<float>(),
                         S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::vector<at::Tensor>{dV, dK};
  }, "fused dv+dk ws kernel (bench)");
  m.def("dvdk_dbg", []() {
    auto t = at::empty({2 * (2 * 64 * 64 + 1024)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    if (g_dvdk_dbg) cudaMemcpy(t.data_ptr(), g_dvdk_dbg, t.numel(), cudaMemcpyDeviceToDevice);
    return t;
  });
  m.def("dv_ws", [](at::Tensor Kd, at::Tensor Qd, at::Tensor Dt,
                    at::Tensor sfK, at::Tensor sfQ, at::Tensor sfDt,
                    at::Tensor lse, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    at::Tensor dV = at::empty({H, S, 128}, at::TensorOptions().dtype(at::kFloat).device(Kd.device()));
    mxfp8_dv_ws_launch(Kd.data_ptr(), Qd.data_ptr(), Dt.data_ptr(),
                       sfK.data_ptr(), sfQ.data_ptr(), sfDt.data_ptr(),
                       lse.data_ptr<float>(), dV.data_ptr<float>(),
                       S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dV;
  }, "dv ws kernel only (bench)");
  m.def("dk_ws", [](at::Tensor Kd, at::Tensor Vd, at::Tensor Qd, at::Tensor Dd, at::Tensor Qt,
                    at::Tensor sfK, at::Tensor sfV, at::Tensor sfQ, at::Tensor sfD, at::Tensor sfQt,
                    at::Tensor lse, at::Tensor delta, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    at::Tensor dK = at::empty({H, S, 128}, at::TensorOptions().dtype(at::kFloat).device(Kd.device()));
    mxfp8_dk_ws_launch(Kd.data_ptr(), Vd.data_ptr(), Qd.data_ptr(), Dd.data_ptr(), Qt.data_ptr(),
                       sfK.data_ptr(), sfV.data_ptr(), sfQ.data_ptr(), sfD.data_ptr(), sfQt.data_ptr(),
                       lse.data_ptr<float>(), delta.data_ptr<float>(), dK.data_ptr<float>(),
                       S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dK;
  }, "dk ws kernel only (bench)"); }
