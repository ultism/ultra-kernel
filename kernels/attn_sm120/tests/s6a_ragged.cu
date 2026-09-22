// S6a: ragged / variable-length prefill addressing + GQA/MQA + offset_q causal (slices 1-3).
//
// Validates the ragged + multi-head TMA addressing in tests/s3_kernel.cuh: the kernel
// consumes the full BatchPrefillPersistentTileScheduler 8-tuple (qo_indptr/kv_indptr/
// qo_len/kv_len + request-local q_tile + qo_head/kv_head), indexes Q/SFQ by qo_head and
// K/V/SFK/SFV by kv_head (= qo_head/group_size), over per-request 128-padded packed
// tensors with a trailing head mode (Q/K token-major [token,head,head_dim]; V head-major
// transposed [kv_head,head_dim,keys]).
//
// ORACLE = the already-torchao-validated DENSE kernel (SingleTileScheduler, 1 head),
// run per (request, qo_head). For a batch of B varlen requests and a GQA group_size:
//   (1) run dense per (request r, qo_head h): Q-head h vs kv_head (h/group_size)'s K/V -> O_dense[r][h],
//   (2) pack everything into 128-padded multi-head global buffers, build a HOST LPT
//       work-list (one item per (request, qo_head, q-tile); kv_head computed on host),
//       run the ragged kernel (BatchPrefillPersistentTileScheduler) once,
//   (3) assert O_ragged[head h, request r's rows] == O_dense[r][h] (and LSE) bit-for-bit.
// Grid forced small (S3_NUM_SM, default 2) so one CTA serially spans tiles from DIFFERENT
// (request, head) pairs -- the cross-request/cross-head addressing trap.
// ragged==dense (bit-exact) + dense==torchao  =>  ragged==torchao per (request, head).
//
// Also exercises the partial-last-block V kFillZero (S3_V_KFILLZERO): the ragged buffers'
// padding gap [kv_len, kv_pad) is poisoned with NaN bytes (0xFF) for both V data and the
// fully-padded V-SF blocks. Bit-exact still holds ONLY if the kernel sanitizes them, so masked
// P=0 never meets 0*NaN=NaN. Rebuild the kernel with -DS3_V_KFILLZERO=0 to see it FAIL (NaN).
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s6a_ragged.cu -o tests/s6a_ragged

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <queue>
#include <vector>
#include "s3_kernel.cuh"

using namespace flashinfer;

static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }
static int query_num_sm() {
  if (const char* e = std::getenv("S3_NUM_SM")) return std::atoi(e);
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n;
}
static int cdiv(int a, int b) { return (a + b - 1) / b; }

static const std::vector<float> kDV = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};

// One Q head (single-head buffers): Q[qo_pad,HD] e4m3 + qexp[qo_pad,NBLK]. Salted by seed
// so distinct (request,head) pairs differ; padding rows (>=qo_len) stay zero / exp -127.
struct QHead { std::vector<uint8_t> Q; std::vector<int> qexp; };
static QHead gen_q_head(int seed, int qo_len, int qo_pad, int HD) {
  const int NBLK = HD / SFVecSize;
  QHead d; d.Q.assign(qo_pad * HD, 0); d.qexp.assign(qo_pad * NBLK, -127);
  for (int m = 0; m < qo_len; ++m) {
    for (int b = 0; b < NBLK; ++b) d.qexp[m * NBLK + b] = ((m + b + seed) % 4) - 1;
    for (int k = 0; k < HD; ++k) d.Q[m * HD + k] = e4m3_byte(kDV[(m * HD + k + seed * 13) % kDV.size()]);
  }
  return d;
}
// One KV head: K[kv_pad,HD] + kexp; V[HD,kv_pad] (transposed) + vexp[HD, kv_pad/32].
struct KVHead { std::vector<uint8_t> K, V; std::vector<int> kexp, vexp; };
static KVHead gen_kv_head(int seed, int kv_len, int kv_pad, int HD) {
  const int NBLK = HD / SFVecSize, NVK = kv_pad / SFVecSize;
  KVHead d;
  d.K.assign(kv_pad * HD, 0); d.V.assign(HD * kv_pad, 0);
  d.kexp.assign(kv_pad * NBLK, -127); d.vexp.assign(HD * NVK, -127);
  for (int n = 0; n < kv_len; ++n) {
    for (int b = 0; b < NBLK; ++b) d.kexp[n * NBLK + b] = (n + 2 * b + seed) % 3;
    for (int k = 0; k < HD; ++k) d.K[n * HD + k] = e4m3_byte(kDV[(n * 7 + k * 3 + seed * 5) % kDV.size()]);
  }
  for (int h = 0; h < HD; ++h)
    for (int n = 0; n < kv_len; ++n) {
      d.V[h * kv_pad + n] = e4m3_byte(kDV[(h * 5 + n * 2 + seed * 11) % kDV.size()]);
      d.vexp[h * NVK + n / SFVecSize] = ((h + n / SFVecSize + seed) % 3) - 1;
    }
  return d;
}

// Scatter a logical [rows,blocks] exponent table into a cutlass SF tiled layout at a global
// (row offset, head). row_off is a multiple of 128 -> lands on a Blk_MN=128 atom boundary.
template <class Layout>
static void place_sf(std::vector<uint8_t>& dst, Layout layout, const std::vector<int>& exp,
                     int rows, int blocks, int row_off, int head) {
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < blocks; ++b)
      dst[layout(make_coord(row_off + r, b * SFVecSize, head))] = ue8m0_byte_pow2(exp[r * blocks + b]);
}

template <typename Scheduler, bool Causal>
static void launch(Params params, typename Scheduler::Arguments sa, dim3 grid,
                   std::vector<float>& hO, std::vector<float>& hLSE, int O_count, int lse_count) {
  typename Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorage));
  if (cudaFuncSetAttribute(s3_kernel<Scheduler, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    { printf("smem attr fail\n"); std::exit(1); }
  s3_kernel<Scheduler, Causal><<<grid, kNThreads, smem>>>(params, sp);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    { printf("launch fail: %s\n", cudaGetErrorString(cudaGetLastError())); std::exit(1); }
  hO.resize(O_count); hLSE.resize(lse_count);
  cudaMemcpy(hO.data(), params.out_O, hO.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hLSE.data(), params.out_lse, hLSE.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

static Params make_params(Element* dQ, Element* dK, Element* dV, ElementSF* dSFQ, ElementSF* dSFK,
                          ElementSF* dSFV, float* dO, float* dLSE, float* dL, float* dMnb,
                          int Sq_pad, int Sk_pad, int HD, float sm_scale,
                          int num_qo_heads, int num_kv_heads) {
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, num_qo_heads));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, num_kv_heads));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, num_kv_heads));
  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(Sq_pad, HD, num_qo_heads), make_stride(num_qo_heads * HD, _1{}, HD));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Sk_pad, HD, num_kv_heads), make_stride(num_kv_heads * HD, _1{}, HD));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, Sk_pad, num_kv_heads), make_stride(Sk_pad, _1{}, HD * Sk_pad));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSFV), layoutSFV);
  Params p;
  p.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  p.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  p.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  p.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{});
  p.layout_sfq = layoutSFQ; p.layout_sfv = layoutSFV;
  p.seqlen_q = Sq_pad; p.seqlen_k = Sk_pad; p.n_block_total = Sk_pad / kBlockN; p.sm_scale = sm_scale;
  p.num_qo_heads = num_qo_heads; p.num_kv_heads = num_kv_heads;
  p.tile_kv_len = nullptr;
  p.out_O = dO; p.out_lse = dLSE; p.out_l = dL; p.out_Ppre = nullptr; p.out_Mnb = dMnb; p.out_dbg = nullptr;
  return p;
}

static int* upload(const std::vector<int>& v) {
  int* d; cudaMalloc(&d, std::max<size_t>(1, v.size()) * 4);
  if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
  return d;
}

// One config = (num_qo_heads, num_kv_heads, causal). Returns 0 on PASS.
static int run_config(int num_qo_heads, int num_kv_heads, bool causal) {
  const int HD = kHeadDim, NBLK = HD / SFVecSize, group = num_qo_heads / num_kv_heads;
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  // varlen batch. {qo,kv}: 200/360 are non-128-multiples -> partial last-tile key mask.
  // The last three have kv_len > qo_len (slice-3 offset_q = kv_len - qo_len > 0: append /
  // chunked-prefill causal). {128,384} clean offset=256; {64,200} offset=136 w/ partial qo
  // tile AND partial kv; {256,360} offset=104 multi qo-tile w/ partial kv. The dense oracle
  // runs the SAME kernel per request (SingleTileScheduler also carries qo_len,kv_len), so it
  // computes the identical offset_q -> this checks ragged ADDRESSING delivers per-request
  // qo_len/kv_len correctly; the offset MASK MATH is validated by the torchao oracle.
  std::vector<std::pair<int,int>> lens = {{128,128}, {256,256}, {200,200}, {384,384}, {360,360},
                                          {128,384}, {64,200}, {256,360}};
  const int B = lens.size();

  std::vector<int> qo_len(B), kv_len(B), qo_pad(B), kv_pad(B), qo_base(B), kv_base(B);
  int Sq_pad = 0, Sk_pad = 0;
  for (int r = 0; r < B; ++r) {
    qo_len[r] = lens[r].first; kv_len[r] = lens[r].second;
    qo_pad[r] = cdiv(qo_len[r], kBlockM) * kBlockM; kv_pad[r] = cdiv(kv_len[r], kSFBlockN) * kSFBlockN;  // 128-pad: SF atoms align w/ requests
    qo_base[r] = Sq_pad; kv_base[r] = Sk_pad; Sq_pad += qo_pad[r]; Sk_pad += kv_pad[r];
  }

  // per-(request,head) single-head data. Q indexed by qo_head, K/V by kv_head. seed distinct.
  auto qseed = [&](int r, int h){ return r * 100 + h + 1; };
  auto kseed = [&](int r, int h){ return r * 100 + h + 50; };
  std::vector<std::vector<QHead>> QH(B);
  std::vector<std::vector<KVHead>> KV(B);
  for (int r = 0; r < B; ++r) {
    for (int h = 0; h < num_qo_heads; ++h) QH[r].push_back(gen_q_head(qseed(r, h), qo_len[r], qo_pad[r], HD));
    for (int h = 0; h < num_kv_heads; ++h) KV[r].push_back(gen_kv_head(kseed(r, h), kv_len[r], kv_pad[r], HD));
  }

  // ---------------- (1) per-(request,qo_head) DENSE oracle ----------------
  // Odense[r][hq] : [qo_pad, HD],  LSEdense[r][hq] : [qo_pad]
  std::vector<std::vector<std::vector<float>>> Odense(B), LSEdense(B);
  for (int r = 0; r < B; ++r) {
    Odense[r].resize(num_qo_heads); LSEdense[r].resize(num_qo_heads);
    const int NVK = kv_pad[r] / SFVecSize;
    auto lSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(qo_pad[r], int(kBlockN), HD, 1));
    auto lSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(kv_pad[r], int(kBlockN), HD, 1));
    auto lSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, kv_pad[r], 1));
    for (int hq = 0; hq < num_qo_heads; ++hq) {
      int hk = hq / group;
      const QHead& q = QH[r][hq]; const KVHead& v = KV[r][hk];
      std::vector<uint8_t> hSFQ(cosize(lSFQ), 0), hSFK(cosize(lSFK), 0), hSFV(cosize(lSFV), 0);
      place_sf(hSFQ, lSFQ, q.qexp, qo_pad[r], NBLK, 0, 0);
      place_sf(hSFK, lSFK, v.kexp, kv_pad[r], NBLK, 0, 0);
      for (int h = 0; h < HD; ++h) for (int b = 0; b < NVK; ++b)
        hSFV[lSFV(make_coord(h, b * SFVecSize, 0))] = ue8m0_byte_pow2(v.vexp[h * NVK + b]);

      Element *dQ,*dK,*dV; ElementSF *dSFQ,*dSFK,*dSFV; float *dO,*dLSE,*dL,*dMnb;
      cudaMalloc(&dQ, q.Q.size()); cudaMalloc(&dK, v.K.size()); cudaMalloc(&dV, v.V.size());
      cudaMalloc(&dSFQ, hSFQ.size()); cudaMalloc(&dSFK, hSFK.size()); cudaMalloc(&dSFV, hSFV.size());
      cudaMalloc(&dO, qo_pad[r] * HD * sizeof(float)); cudaMalloc(&dLSE, qo_pad[r] * sizeof(float));
      cudaMalloc(&dL, qo_pad[r] * sizeof(float)); cudaMalloc(&dMnb, size_t(qo_pad[r]) * (kv_pad[r] / kBlockN) * sizeof(float));
      cudaMemcpy(dQ, q.Q.data(), q.Q.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dK, v.K.data(), v.K.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dV, v.V.data(), v.V.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice);

      Params p = make_params(dQ, dK, dV, dSFQ, dSFK, dSFV, dO, dLSE, dL, dMnb, qo_pad[r], kv_pad[r], HD, sm_scale, 1, 1);
      using Sched = SingleTileScheduler;
      Sched::Arguments sm{cdiv(qo_len[r], kBlockM), 1, qo_len[r], kv_len[r], cutlass::FastDivmod(1)};
      dim3 grid = Sched::get_grid_dim(sm, query_num_sm());
      std::vector<float> hO, hLSE;
      if (causal) launch<Sched, true >(p, sm, grid, hO, hLSE, qo_pad[r] * HD, qo_pad[r]);
      else        launch<Sched, false>(p, sm, grid, hO, hLSE, qo_pad[r] * HD, qo_pad[r]);
      Odense[r][hq] = hO; LSEdense[r][hq] = hLSE;
      cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV);
      cudaFree(dO); cudaFree(dLSE); cudaFree(dL); cudaFree(dMnb);
    }
  }

  // ---------------- (2) packed multi-head RAGGED run ----------------
  auto gSFQ_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, num_qo_heads));
  auto gSFK_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, num_kv_heads));
  auto gSFV_l = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, num_kv_heads));
  std::vector<uint8_t> gQ(size_t(Sq_pad) * num_qo_heads * HD, 0), gK(size_t(Sk_pad) * num_kv_heads * HD, 0),
                       gV(size_t(num_kv_heads) * HD * Sk_pad, 0);
  std::vector<uint8_t> gSFQ(cosize(gSFQ_l), 0), gSFK(cosize(gSFK_l), 0), gSFV(cosize(gSFV_l), 0);
  for (int r = 0; r < B; ++r) {
    // Q token-major [token, qo_head, HD]: row (qo_base+m), head hq.
    for (int hq = 0; hq < num_qo_heads; ++hq) {
      const QHead& q = QH[r][hq];
      for (int m = 0; m < qo_pad[r]; ++m)
        std::copy(q.Q.begin() + size_t(m) * HD, q.Q.begin() + size_t(m + 1) * HD,
                  gQ.begin() + ((size_t(qo_base[r] + m) * num_qo_heads + hq) * HD));
      place_sf(gSFQ, gSFQ_l, q.qexp, qo_pad[r], NBLK, qo_base[r], hq);
    }
    for (int hk = 0; hk < num_kv_heads; ++hk) {
      const KVHead& v = KV[r][hk]; const int NVK = kv_pad[r] / SFVecSize;
      // K token-major [token, kv_head, HD].
      for (int n = 0; n < kv_pad[r]; ++n)
        std::copy(v.K.begin() + size_t(n) * HD, v.K.begin() + size_t(n + 1) * HD,
                  gK.begin() + ((size_t(kv_base[r] + n) * num_kv_heads + hk) * HD));
      // V head-major-transposed [kv_head, HD, keys]: head hk's [HD, kv_pad] block at column kv_base.
      for (int h = 0; h < HD; ++h)
        std::copy(v.V.begin() + size_t(h) * kv_pad[r], v.V.begin() + size_t(h + 1) * kv_pad[r],
                  gV.begin() + (size_t(hk) * HD + h) * Sk_pad + kv_base[r]);
      place_sf(gSFK, gSFK_l, v.kexp, kv_pad[r], NBLK, kv_base[r], hk);
      for (int h = 0; h < HD; ++h) for (int b = 0; b < NVK; ++b)
        gSFV[gSFV_l(make_coord(h, kv_base[r] + b * SFVecSize, hk))] = ue8m0_byte_pow2(v.vexp[h * NVK + b]);
    }
  }

  // ---- ADVERSARIAL: poison the V padding gap [kv_len, kv_pad) with NaN bytes (0xFF) on the
  // RAGGED buffers only (the dense oracle keeps its benign zero pad). Mirrors a real recycled/
  // uninitialized KV-cache tail. A correct kernel MUST kFillZero these so masked P=0 never
  // meets 0*NaN=NaN (the K/V asymmetry: K is masked to -inf BEFORE softmax, V is read by PV
  // for ALL keys). V DATA: every pad key. V-SF: only FULLY-padded 32-blocks -- the straddling
  // block's SF is shared with its valid keys, so a well-formed cache keeps it finite.
  // To prove the fix is load-bearing: rebuild the kernel with -DS3_V_KFILLZERO=0 -> expect NaN/FAIL.
  for (int r = 0; r < B; ++r)
    for (int hk = 0; hk < num_kv_heads; ++hk) {
      const int NVK = kv_pad[r] / SFVecSize;
      for (int h = 0; h < HD; ++h) {
        for (int n = kv_len[r]; n < kv_pad[r]; ++n)
          gV[(size_t(hk) * HD + h) * Sk_pad + kv_base[r] + n] = 0xFF;
        for (int b = cdiv(kv_len[r], SFVecSize); b < NVK; ++b)
          gSFV[gSFV_l(make_coord(h, kv_base[r] + b * SFVecSize, hk))] = 0xFF;
      }
    }

  Element *dQ,*dK,*dV; ElementSF *dSFQ,*dSFK,*dSFV; float *dO,*dLSE,*dL,*dMnb;
  cudaMalloc(&dQ, gQ.size()); cudaMalloc(&dK, gK.size()); cudaMalloc(&dV, gV.size());
  cudaMalloc(&dSFQ, gSFQ.size()); cudaMalloc(&dSFK, gSFK.size()); cudaMalloc(&dSFV, gSFV.size());
  cudaMalloc(&dO, size_t(Sq_pad) * num_qo_heads * HD * sizeof(float)); cudaMalloc(&dLSE, size_t(num_qo_heads) * Sq_pad * sizeof(float));
  cudaMalloc(&dL, size_t(num_qo_heads) * Sq_pad * sizeof(float)); cudaMalloc(&dMnb, size_t(Sq_pad) * (Sk_pad / kBlockN) * sizeof(float));
  cudaMemset(dO, 0, size_t(Sq_pad) * num_qo_heads * HD * sizeof(float));
  cudaMemcpy(dQ, gQ.data(), gQ.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, gK.data(), gK.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, gV.data(), gV.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFQ, gSFQ.data(), gSFQ.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFK, gSFK.data(), gSFK.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFV, gSFV.data(), gSFV.size(), cudaMemcpyHostToDevice);
  Params p = make_params(dQ, dK, dV, dSFQ, dSFK, dSFV, dO, dLSE, dL, dMnb, Sq_pad, Sk_pad, HD, sm_scale, num_qo_heads, num_kv_heads);

  // ---- HOST LPT work-list: one item per (request, qo_head, q-tile); kv_head = qo_head/group on host ----
  int num_sm = query_num_sm();
  struct W { int req, qhead, qtile; long cost; };
  std::vector<W> works;
  for (int r = 0; r < B; ++r) {
    int nqt = cdiv(qo_len[r], kBlockM), nkt = cdiv(kv_len[r], kBlockN);
    int offset = kv_len[r] - qo_len[r];   // slice-3: match the kernel's offset_q n_block_max bound
    for (int hq = 0; hq < num_qo_heads; ++hq)
      for (int qt = 0; qt < nqt; ++qt) {
        int eff = causal ? std::min(nkt, cdiv((qt + 1) * kBlockM + offset, kBlockN)) : nkt;
        works.push_back({r, hq, qt, (long)eff});
      }
  }
  std::stable_sort(works.begin(), works.end(), [](const W& a, const W& b) { return a.cost > b.cost; });
  using PQ = std::pair<long, int>;
  std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> heap;
  for (int c = 0; c < num_sm; ++c) heap.push({0, c});
  std::vector<std::vector<W>> cta(num_sm);
  for (auto& w : works) { auto [load, c] = heap.top(); heap.pop(); cta[c].push_back(w); heap.push({load + w.cost, c}); }
  std::vector<int> work_indptr(num_sm + 1, 0), head_i, qo_tile_i, qo_ip, kv_ip, qo_l, kv_l, batch_i;
  for (int c = 0; c < num_sm; ++c) work_indptr[c + 1] = work_indptr[c] + cta[c].size();
  for (int c = 0; c < num_sm; ++c) for (auto& w : cta[c]) {
    head_i.push_back(w.qhead); qo_tile_i.push_back(w.qtile);   // group_size_fastdiv divides -> kv_head
    qo_ip.push_back(qo_base[w.req]); kv_ip.push_back(kv_base[w.req]);
    qo_l.push_back(qo_len[w.req]); kv_l.push_back(kv_len[w.req]); batch_i.push_back(w.req);
  }
  int total_works = work_indptr[num_sm];
  using Sched = BatchPrefillPersistentTileScheduler<int>;
  Sched::Arguments sa;
  sa.work_indptr = upload(work_indptr); sa.head_indices = upload(head_i); sa.qo_tile_indices = upload(qo_tile_i);
  sa.qo_indptr = upload(qo_ip); sa.kv_indptr = upload(kv_ip); sa.qo_lens = upload(qo_l); sa.kv_lens = upload(kv_l);
  sa.batch_indices = upload(batch_i); sa.group_size_fastdiv = cutlass::FastDivmod(group); sa.num_qo_heads = num_qo_heads;
  dim3 grid = Sched::get_grid_dim(sa, num_sm);

  std::vector<float> rO, rLSE;
  if (causal) launch<Sched, true >(p, sa, grid, rO, rLSE, size_t(Sq_pad) * num_qo_heads * HD, size_t(num_qo_heads) * Sq_pad);
  else        launch<Sched, false>(p, sa, grid, rO, rLSE, size_t(Sq_pad) * num_qo_heads * HD, size_t(num_qo_heads) * Sq_pad);

  // ---------------- (3) compare ragged[head h, req r rows] vs dense[r][h] ----------------
  double max_abs = 0, max_lse = 0; int bad = 0, fr = -1, fh = -1, fm = -1, fd = -1;
  for (int r = 0; r < B; ++r)
    for (int hq = 0; hq < num_qo_heads; ++hq)
      for (int m = 0; m < qo_len[r]; ++m) {
        int gq = qo_base[r] + m;
        double dl = std::abs((double)rLSE[size_t(hq) * Sq_pad + gq] - (double)LSEdense[r][hq][m]);
        max_lse = std::max(max_lse, dl);
        for (int d = 0; d < HD; ++d) {
          double a = rO[(size_t(gq) * num_qo_heads + hq) * HD + d];   // token-major O
          double b = Odense[r][hq][size_t(m) * HD + d];
          double e = std::abs(a - b);
          // NaN/Inf-aware: a NaN ragged output (0*NaN poison) must FAIL. `NaN > tol` is false,
          // so the naive `e > tol` would silently pass it -- use isfinite + !(e <= tol).
          if (!std::isfinite(a) || !(e <= 1e-5)) { if (!bad) { fr = r; fh = hq; fm = m; fd = d; } ++bad; }
          if (std::isfinite(e) && e > max_abs) max_abs = e;
        }
      }
  printf("  [qo=%d kv=%d g=%d %-10s] grid=%u CTAs %d works | O max|abs|=%.3g LSE max|abs|=%.3g bad=%d\n",
         num_qo_heads, num_kv_heads, group, causal ? "causal" : "non-causal", grid.x, total_works, max_abs, max_lse, bad);
  if (bad) printf("    first bad: req=%d qhead=%d q=%d d=%d  ragged=%.6f dense=%.6f\n", fr, fh, fm, fd,
                  rO[(size_t(qo_base[fr] + fm) * num_qo_heads + fh) * HD + fd], Odense[fr][fh][size_t(fm) * HD + fd]);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV);
  cudaFree(dO); cudaFree(dLSE); cudaFree(dL); cudaFree(dMnb);
  cudaFree(sa.work_indptr); cudaFree(sa.head_indices); cudaFree(sa.qo_tile_indices); cudaFree(sa.qo_indptr);
  cudaFree(sa.kv_indptr); cudaFree(sa.qo_lens); cudaFree(sa.kv_lens); cudaFree(sa.batch_indices);
  return bad ? 1 : 0;
}

int main() {
  printf("S6a: ragged varlen + GQA prefill addressing (head_dim=%d)\n", int(kHeadDim));
  // (1,1) = pure varlen (slice-1 regression); (8,2),(4,4),(4,1=MQA) = GQA addressing.
  std::vector<std::pair<int,int>> cfgs = {{1,1}, {8,2}, {4,4}, {4,1}};
  int rc = 0;
  for (auto [qh, kh] : cfgs) {
    rc |= run_config(qh, kh, false);
    rc |= run_config(qh, kh, true);
  }
  printf(rc == 0 ? "S6a PASS\n" : "S6a FAIL\n");
  return rc;
}
