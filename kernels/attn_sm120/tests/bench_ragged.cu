// S7/S8 bench: varlen (ragged) + GQA + causal prefill timing for OUR mxfp8 kernel.
// Reuses the VALIDATED s6a_ragged section-2 machinery (pack multi-head 128-padded buffers +
// host LPT work-list + BatchPrefillPersistentTileScheduler single persistent launch), but
// strips the dense oracle / NaN-poison / compare and times N launches (CUDA events).
// Prints the per-request seq lengths so the flashinfer fa2 ragged bench runs the SAME shapes.
//   build: nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda -I tmp/cutlass/include -I include \
//     tests/bench_ragged.cu -o tests/bench_ragged
#include <cstdlib>
#include <cmath>
#include <cstdio>
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
template <class Layout>
static void place_sf(std::vector<uint8_t>& dst, Layout layout, const std::vector<int>& exp,
                     int rows, int blocks, int row_off, int head) {
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < blocks; ++b)
      dst[layout(make_coord(row_off + r, b * SFVecSize, head))] = ue8m0_byte_pow2(exp[r * blocks + b]);
}
static Params make_params(Element* dQ, Element* dK, Element* dV, ElementSF* dSFQ, ElementSF* dSFK,
                          ElementSF* dSFV, float* dO, float* dLSE, float* dL, float* dMnb,
                          int Sq_pad, int Sk_pad, int HD, float sm_scale, int num_qo_heads, int num_kv_heads) {
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
  p.num_qo_heads = num_qo_heads; p.num_kv_heads = num_kv_heads; p.tile_kv_len = nullptr;
  p.out_O = dO; p.out_lse = dLSE; p.out_l = dL; p.out_Ppre = nullptr; p.out_Mnb = dMnb; p.out_dbg = nullptr;
  return p;
}
static int* upload(const std::vector<int>& v) {
  int* d; cudaMalloc(&d, std::max<size_t>(1, v.size()) * 4);
  if (!v.empty()) cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
  return d;
}
template <typename Sched, bool Causal>
static float launch_timed(Params params, typename Sched::Arguments sa, dim3 grid, int iters) {
  typename Sched::Params sp = Sched::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorage));
  cudaFuncSetAttribute(s3_kernel<Sched, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  for (int i = 0; i < 10; ++i) s3_kernel<Sched, Causal><<<grid, kNThreads, smem>>>(params, sp);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch fail: %s\n", cudaGetErrorString(cudaGetLastError())); std::exit(1); }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) s3_kernel<Sched, Causal><<<grid, kNThreads, smem>>>(params, sp);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

// time OUR ragged kernel on a varlen batch (qo_len=kv_len=lens[r], prefill). Returns ms/iter.
static float bench_ours(const std::vector<int>& lens, int num_qo_heads, int num_kv_heads, bool causal, int iters) {
  const int HD = kHeadDim, NBLK = HD / SFVecSize, group = num_qo_heads / num_kv_heads, B = lens.size();
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  std::vector<int> qo_len(B), kv_len(B), qo_pad(B), kv_pad(B), qo_base(B), kv_base(B);
  int Sq_pad = 0, Sk_pad = 0;
  for (int r = 0; r < B; ++r) {
    qo_len[r] = lens[r]; kv_len[r] = lens[r];
    qo_pad[r] = cdiv(qo_len[r], kBlockM) * kBlockM; kv_pad[r] = cdiv(kv_len[r], kSFBlockN) * kSFBlockN;  // 128-pad for SF atom alignment
    qo_base[r] = Sq_pad; kv_base[r] = Sk_pad; Sq_pad += qo_pad[r]; Sk_pad += kv_pad[r];
  }
  auto qseed = [&](int r, int h){ return r * 100 + h + 1; };
  auto kseed = [&](int r, int h){ return r * 100 + h + 50; };
  std::vector<std::vector<QHead>> QH(B); std::vector<std::vector<KVHead>> KV(B);
  for (int r = 0; r < B; ++r) {
    for (int h = 0; h < num_qo_heads; ++h) QH[r].push_back(gen_q_head(qseed(r, h), qo_len[r], qo_pad[r], HD));
    for (int h = 0; h < num_kv_heads; ++h) KV[r].push_back(gen_kv_head(kseed(r, h), kv_len[r], kv_pad[r], HD));
  }
  auto gSFQ_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, num_qo_heads));
  auto gSFK_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, num_kv_heads));
  auto gSFV_l = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, num_kv_heads));
  std::vector<uint8_t> gQ(size_t(Sq_pad) * num_qo_heads * HD, 0), gK(size_t(Sk_pad) * num_kv_heads * HD, 0),
                       gV(size_t(num_kv_heads) * HD * Sk_pad, 0);
  std::vector<uint8_t> gSFQ(cosize(gSFQ_l), 0), gSFK(cosize(gSFK_l), 0), gSFV(cosize(gSFV_l), 0);
  for (int r = 0; r < B; ++r) {
    for (int hq = 0; hq < num_qo_heads; ++hq) {
      const QHead& q = QH[r][hq];
      for (int m = 0; m < qo_pad[r]; ++m)
        std::copy(q.Q.begin() + size_t(m) * HD, q.Q.begin() + size_t(m + 1) * HD,
                  gQ.begin() + ((size_t(qo_base[r] + m) * num_qo_heads + hq) * HD));
      place_sf(gSFQ, gSFQ_l, q.qexp, qo_pad[r], NBLK, qo_base[r], hq);
    }
    for (int hk = 0; hk < num_kv_heads; ++hk) {
      const KVHead& v = KV[r][hk]; const int NVK = kv_pad[r] / SFVecSize;
      for (int n = 0; n < kv_pad[r]; ++n)
        std::copy(v.K.begin() + size_t(n) * HD, v.K.begin() + size_t(n + 1) * HD,
                  gK.begin() + ((size_t(kv_base[r] + n) * num_kv_heads + hk) * HD));
      for (int h = 0; h < HD; ++h)
        std::copy(v.V.begin() + size_t(h) * kv_pad[r], v.V.begin() + size_t(h + 1) * kv_pad[r],
                  gV.begin() + (size_t(hk) * HD + h) * Sk_pad + kv_base[r]);
      place_sf(gSFK, gSFK_l, v.kexp, kv_pad[r], NBLK, kv_base[r], hk);
      for (int h = 0; h < HD; ++h) for (int b = 0; b < NVK; ++b)
        gSFV[gSFV_l(make_coord(h, kv_base[r] + b * SFVecSize, hk))] = ue8m0_byte_pow2(v.vexp[h * NVK + b]);
    }
  }
  Element *dQ,*dK,*dV; ElementSF *dSFQ,*dSFK,*dSFV; float *dO,*dLSE,*dL,*dMnb;
  cudaMalloc(&dQ, gQ.size()); cudaMalloc(&dK, gK.size()); cudaMalloc(&dV, gV.size());
  cudaMalloc(&dSFQ, gSFQ.size()); cudaMalloc(&dSFK, gSFK.size()); cudaMalloc(&dSFV, gSFV.size());
  cudaMalloc(&dO, size_t(Sq_pad) * num_qo_heads * HD * sizeof(float)); cudaMalloc(&dLSE, size_t(num_qo_heads) * Sq_pad * sizeof(float));
  cudaMalloc(&dL, size_t(num_qo_heads) * Sq_pad * sizeof(float)); cudaMalloc(&dMnb, size_t(Sq_pad) * (Sk_pad / kBlockN) * sizeof(float));
  cudaMemcpy(dQ, gQ.data(), gQ.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, gK.data(), gK.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, gV.data(), gV.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFQ, gSFQ.data(), gSFQ.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFK, gSFK.data(), gSFK.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFV, gSFV.data(), gSFV.size(), cudaMemcpyHostToDevice);
  Params p = make_params(dQ, dK, dV, dSFQ, dSFK, dSFV, dO, dLSE, dL, dMnb, Sq_pad, Sk_pad, HD, sm_scale, num_qo_heads, num_kv_heads);
  int num_sm = query_num_sm();
  struct W { int req, qhead, qtile; long cost; };
  std::vector<W> works;
  for (int r = 0; r < B; ++r) {
    int nqt = cdiv(qo_len[r], kBlockM), nkt = cdiv(kv_len[r], kBlockN), offset = kv_len[r] - qo_len[r];
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
    head_i.push_back(w.qhead); qo_tile_i.push_back(w.qtile);
    qo_ip.push_back(qo_base[w.req]); kv_ip.push_back(kv_base[w.req]);
    qo_l.push_back(qo_len[w.req]); kv_l.push_back(kv_len[w.req]); batch_i.push_back(w.req);
  }
  using Sched = BatchPrefillPersistentTileScheduler<int>;
  Sched::Arguments sa;
  sa.work_indptr = upload(work_indptr); sa.head_indices = upload(head_i); sa.qo_tile_indices = upload(qo_tile_i);
  sa.qo_indptr = upload(qo_ip); sa.kv_indptr = upload(kv_ip); sa.qo_lens = upload(qo_l); sa.kv_lens = upload(kv_l);
  sa.batch_indices = upload(batch_i); sa.group_size_fastdiv = cutlass::FastDivmod(group); sa.num_qo_heads = num_qo_heads;
  dim3 grid = Sched::get_grid_dim(sa, num_sm);
  float ms = causal ? launch_timed<Sched, true>(p, sa, grid, iters) : launch_timed<Sched, false>(p, sa, grid, iters);
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV);
  cudaFree(dO); cudaFree(dLSE); cudaFree(dL); cudaFree(dMnb);
  cudaFree(sa.work_indptr); cudaFree(sa.head_indices); cudaFree(sa.qo_tile_indices); cudaFree(sa.qo_indptr);
  cudaFree(sa.kv_indptr); cudaFree(sa.qo_lens); cudaFree(sa.kv_lens); cudaFree(sa.batch_indices);
  return ms;
}

int main() {
  // production-like varlen prefill batch (qo_len == kv_len), mixed lengths incl non-128-multiples.
  // env overrides (5090 evidence runs): S3_LENS="512,1024,..." S3_QH=8 S3_KH=2 (single cfg)
  std::vector<int> lens = {512,1024,768,1536,2048,640,1280,896,384,1792,512,2560,1024,768,2048,1100};
  if (const char* e = std::getenv("S3_LENS")) {
    lens.clear();
    std::string s(e); size_t p = 0;
    while (p <= s.size()) { size_t c = s.find(',', p); lens.push_back(std::stoi(s.substr(p, c - p))); if (c == std::string::npos) break; p = c + 1; }
  }
  int total = 0, mn = 1<<30, mx = 0; for (int L : lens) { total += L; mn = std::min(mn,L); mx = std::max(mx,L); }
  printf("LENS"); for (int L : lens) printf(" %d", L); printf("\n");
  printf("# batch B=%d  total_tokens=%d  min=%d max=%d  head_dim=%d  SMs=%d\n",
         (int)lens.size(), total, mn, mx, int(kHeadDim), query_num_sm());
  std::vector<std::pair<int,int>> cfgs = {{1,1},{8,2},{32,8}};
  if (const char* e = std::getenv("S3_QH")) {
    int kh = std::getenv("S3_KH") ? std::atoi(std::getenv("S3_KH")) : 1;
    cfgs = {{std::atoi(e), kh}};
  }
  printf("\n%-12s %-8s | %12s\n", "cfg(qh,kh)", "causal", "ours ms");
  printf("--------------------------------------------\n");
  for (auto [qh,kh] : cfgs)
    for (bool causal : {true})
      printf("%2d,%-9d %-8s | %12.4f\n", qh, kh, causal?"causal":"dense", bench_ours(lens, qh, kh, causal, 100));
  printf("\nOURS_DONE\n");
  return 0;
}
