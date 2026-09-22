// pybind glue for the S9 ragged+GQA+per-tensor-fp8 prefill kernel. Compiled by the HOST
// gcc (torch 2.11 c10 headers need -fpermissive there; nvcc's frontend rejects them, so
// the kernel launcher lives in mxfp8_ragged_kernel.cu with zero torch includes).
//
// Contract (all CUDA tensors, caller pre-pads each request to 128-multiples):
//   Qd  : [Sq_pad, Hq,  D] uint8 e4m3 (token-major, contiguous)
//   Kd  : [Sk_pad, Hkv, D] uint8 e4m3
//   Vt  : [Hkv, D, Sk_pad] uint8 e4m3 (DIM-major per head)
//   qo_indptr / kv_indptr : int32 [B+1] over PADDED token counts (128-multiples)
//   qo_lens / kv_lens     : int32 [B] real lengths
//   sm_scale : full score scale, host-folded: (1/sqrt(D)) * q_scale * k_scale
//   o_scale  : = v_scale
// Returns: O fp32 [Sq_pad, Hq, D], LSE fp32 [Hq, Sq_pad], L fp32 [Hq, Sq_pad].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <queue>
#include <vector>

extern "C" void s3_ragged_fp8_launch(
    const void* Qd, const void* Kd, const void* Vt,
    int Sq_pad, int Sk_pad, int Hq, int Hkv, int group,
    float sm_scale, float o_scale, int causal,
    float* out_O, float* out_lse, float* out_l,
    int* work_indptr, int* head_indices, int* qo_tile_indices,
    int* qo_indptr, int* kv_indptr, int* qo_lens, int* kv_lens, int* batch_indices,
    int num_sm, uintptr_t stream_);
extern "C" int s3_num_sm();

static constexpr int kBlockM_ = 128, kBlockN_ = 64;
static int cdiv_h(int a, int b) { return (a + b - 1) / b; }

std::vector<torch::Tensor> s3_ragged_fp8_attn(
    torch::Tensor Qd, torch::Tensor Kd, torch::Tensor Vt,
    torch::Tensor qo_indptr, torch::Tensor kv_indptr,
    torch::Tensor qo_lens, torch::Tensor kv_lens,
    int64_t num_qo_heads, int64_t num_kv_heads,
    double sm_scale, double o_scale, bool causal, int64_t bench_iters) {
  TORCH_CHECK(Qd.is_cuda() && Kd.is_cuda() && Vt.is_cuda(), "QKV must be CUDA");
  TORCH_CHECK(Qd.dtype() == torch::kUInt8 && Kd.dtype() == torch::kUInt8 && Vt.dtype() == torch::kUInt8,
              "QKV data must be uint8 e4m3 bytes");
  const int Hq = int(num_qo_heads), Hkv = int(num_kv_heads), group = Hq / Hkv;
  const int D = Qd.size(2);
  TORCH_CHECK(Hq % Hkv == 0, "GQA group must divide evenly");

  auto qo_ip_cpu = qo_indptr.to(torch::kCPU, torch::kInt32).contiguous();
  auto kv_ip_cpu = kv_indptr.to(torch::kCPU, torch::kInt32).contiguous();
  auto qo_l_cpu = qo_lens.to(torch::kCPU, torch::kInt32).contiguous();
  auto kv_l_cpu = kv_lens.to(torch::kCPU, torch::kInt32).contiguous();
  const int B = qo_l_cpu.size(0);
  const int* qo_ip = qo_ip_cpu.data_ptr<int>();
  const int* kv_ip = kv_ip_cpu.data_ptr<int>();
  const int* qo_l = qo_l_cpu.data_ptr<int>();
  const int* kv_l = kv_l_cpu.data_ptr<int>();
  const int Sq_pad = qo_ip[B], Sk_pad = kv_ip[B];
  TORCH_CHECK(Qd.size(0) == Sq_pad && Kd.size(0) == Sk_pad, "padded totals mismatch");
  for (int r = 0; r < B; ++r) {
    TORCH_CHECK((qo_ip[r] % kBlockM_) == 0 && (kv_ip[r] % 128) == 0,
                "indptr must be 128-multiples (caller pads)");
    TORCH_CHECK(qo_l[r] <= qo_ip[r + 1] - qo_ip[r] && kv_l[r] <= kv_ip[r + 1] - kv_ip[r],
                "real length exceeds padded extent");
  }

  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(Qd.device());
  auto O = torch::empty({Sq_pad, Hq, D}, opts_f);
  auto LSE = torch::empty({Hq, Sq_pad}, opts_f);
  auto L = torch::empty({Hq, Sq_pad}, opts_f);

  // --- host: LPT work-list over (req, qo_head, q_tile), identical to bench_ragged.cu ---
  const int num_sm = s3_num_sm();
  struct W { int req, qhead, qtile; long cost; };
  std::vector<W> works;
  for (int r = 0; r < B; ++r) {
    int nqt = cdiv_h(qo_l[r], kBlockM_), nkt = cdiv_h(kv_l[r], kBlockN_), offset = kv_l[r] - qo_l[r];
    for (int hq = 0; hq < Hq; ++hq)
      for (int qt = 0; qt < nqt; ++qt) {
        int eff = causal ? std::min(nkt, cdiv_h((qt + 1) * kBlockM_ + offset, kBlockN_)) : nkt;
        works.push_back({r, hq, qt, (long)eff});
      }
  }
  std::stable_sort(works.begin(), works.end(), [](const W& a, const W& b) { return a.cost > b.cost; });
  using PQ = std::pair<long, int>;
  std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> heap;
  for (int c = 0; c < num_sm; ++c) heap.push({0, c});
  std::vector<std::vector<W>> cta(num_sm);
  for (auto& w : works) { auto [load, c] = heap.top(); heap.pop(); cta[c].push_back(w); heap.push({load + w.cost, c}); }
  std::vector<int> work_indptr(num_sm + 1, 0), head_i, qo_tile_i, qo_ip_v, kv_ip_v, qo_l_v, kv_l_v, batch_i;
  for (int c = 0; c < num_sm; ++c) work_indptr[c + 1] = work_indptr[c] + int(cta[c].size());
  for (int c = 0; c < num_sm; ++c) for (auto& w : cta[c]) {
    head_i.push_back(w.qhead); qo_tile_i.push_back(w.qtile);
    qo_ip_v.push_back(qo_ip[w.req]); kv_ip_v.push_back(kv_ip[w.req]);
    qo_l_v.push_back(qo_l[w.req]); kv_l_v.push_back(kv_l[w.req]); batch_i.push_back(w.req);
  }
  auto up = [&](const std::vector<int>& v) {
    auto t = torch::empty({int64_t(std::max<size_t>(1, v.size()))},
                          torch::TensorOptions().dtype(torch::kInt32).device(Qd.device()));
    if (!v.empty()) cudaMemcpy(t.data_ptr<int>(), v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    return t;
  };
  auto d_work_indptr = up(work_indptr), d_head = up(head_i), d_qtile = up(qo_tile_i);
  auto d_qo_ip = up(qo_ip_v), d_kv_ip = up(kv_ip_v), d_qo_l = up(qo_l_v), d_kv_l = up(kv_l_v), d_batch = up(batch_i);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto call = [&]() {
    s3_ragged_fp8_launch(
        Qd.data_ptr(), Kd.data_ptr(), Vt.data_ptr(),
        Sq_pad, Sk_pad, Hq, Hkv, group,
        float(sm_scale), float(o_scale), causal ? 1 : 0,
        O.data_ptr<float>(), LSE.data_ptr<float>(), L.data_ptr<float>(),
        d_work_indptr.data_ptr<int>(), d_head.data_ptr<int>(), d_qtile.data_ptr<int>(),
        d_qo_ip.data_ptr<int>(), d_kv_ip.data_ptr<int>(), d_qo_l.data_ptr<int>(),
        d_kv_l.data_ptr<int>(), d_batch.data_ptr<int>(), num_sm,
        reinterpret_cast<uintptr_t>(stream));
  };

  if (bench_iters <= 0) {
    call();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    for (int i = 0; i < 10; ++i) call();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int i = 0; i < bench_iters; ++i) call();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    printf("s3_ragged_fp8: %.4f ms/iter over %ld iters\n", ms / bench_iters, (long)bench_iters);
  }
  return {O, LSE, L};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("s3_ragged_fp8_attn", &s3_ragged_fp8_attn, "ragged varlen GQA prefill, per-tensor fp8 (kUniformFp8)",
        pybind11::arg("Qd"), pybind11::arg("Kd"), pybind11::arg("Vt"),
        pybind11::arg("qo_indptr"), pybind11::arg("kv_indptr"),
        pybind11::arg("qo_lens"), pybind11::arg("kv_lens"),
        pybind11::arg("num_qo_heads"), pybind11::arg("num_kv_heads"),
        pybind11::arg("sm_scale"), pybind11::arg("o_scale"), pybind11::arg("causal"),
        pybind11::arg("bench_iters") = 0);
}
