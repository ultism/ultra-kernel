// S4b-4: does the LPT-scheduler win (10-20% from the synthetic bench/sched_varlen.cu)
// TRANSFER to the REAL S3 mxfp8 attention kernel under variable-length load?
//
// Same kernel/data; per-q-tile key count drawn from a varlen/tail distribution
// (params.tile_kv_len, multiples of kBlockN). Three schedulers:
//   single : SingleTileScheduler        grid = n_tiles, HW block scheduler
//   static : StaticPersistentScheduler   grid = num_sm, grid-stride (unsorted)
//   lpt    : LPTPersistentScheduler      grid = num_sm, grid-stride over a HOST-sorted
//            (heaviest-first) tile order -- the only thing HW can't do.
// Non-causal (each q-tile attends the first kv_len[m] keys of the shared K/V) so the
// per-tile cost is exactly kv_len[m]/kBlockN -- the variance the scheduler balances.
//
// Build: nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//        --expt-relaxed-constexpr --expt-extended-lambda -I tmp/cutlass/include -I include \
//        bench/bench_varlen_sched.cu -o bench/bench_varlen_sched
// Run:   ./bench/bench_varlen_sched [dist=varlen|tail|uniform] [waves] [seed]
#include <algorithm>
#include <numeric>
#include <random>
#include "../tests/s3_kernel.cuh"

static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static int query_num_sm() { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n; }

template <typename Scheduler>
static float run(Params params, typename Scheduler::Arguments sa, int iters, float& mean, float& sd) {
  typename Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  dim3 grid = Scheduler::get_grid_dim(sa, query_num_sm());
  int smem = int(sizeof(SharedStorage));
  auto kern = s3_kernel<Scheduler, false>;   // non-causal
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  auto launch = [&] { kern<<<grid, kNThreads, smem>>>(params, sp); };
  for (int i = 0; i < 10; ++i) launch();
  cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; ++i) { cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms[i], a, b); }
  std::sort(ms.begin(), ms.end());
  double s = 0; for (float v : ms) s += v; mean = s / iters;
  double v2 = 0; for (float v : ms) v2 += (v - mean) * (v - mean); sd = std::sqrt(v2 / iters);
  return ms[0];
}

int main(int argc, char** argv) {
  const char* dist = argc > 1 ? argv[1] : "varlen";
  const float waves = argc > 2 ? atof(argv[2]) : 4.0f;
  unsigned seed = argc > 3 ? atoi(argv[3]) : 1234;
  const int HD = kHeadDim, iters = 200, num_sm = query_num_sm();
  const int n_tiles = (int)(waves * num_sm);
  const int SQ = n_tiles * kBlockM;
  const int max_blocks = 32;                  // kv_len up to 32*128 = 4096
  const int SK = max_blocks * kBlockN;
  const int n_block_total = SK / kBlockN, NVK = SK / SFVecSize;
  const float sm_scale = 1.0f / std::sqrt((float)HD);

  // per-tile kv_len in key-blocks, drawn from the distribution, then *kBlockN.
  std::mt19937 rng(seed);
  std::vector<int> kvb(n_tiles);
  auto draw = [&]() -> int {
    if (!strcmp(dist, "uniform")) return max_blocks / 2;
    if (!strcmp(dist, "tail")) { std::uniform_int_distribution<int> c(0, 99); return c(rng) < 88 ? 2 : max_blocks; }
    std::vector<int> menu = {1, 2, 4, 8, 16, 32}, wt = {28, 26, 20, 13, 8, 5};   // varlen serving-like
    std::discrete_distribution<int> pick(wt.begin(), wt.end()); return menu[pick(rng)];
  };
  long long tot = 0;
  for (int t = 0; t < n_tiles; ++t) { kvb[t] = std::max(1, std::min(max_blocks, draw())); tot += kvb[t]; }
  std::vector<int> kv_len(n_tiles), order(n_tiles);
  for (int t = 0; t < n_tiles; ++t) kv_len[t] = kvb[t] * kBlockN;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return kvb[a] > kvb[b]; });  // heaviest-first

  // device buffers (constant e4m3 data + flat SF=127; timing is data-independent)
  std::vector<uint8_t> hQ(SQ * HD, e4m3_byte(1.f)), hK(SK * HD, e4m3_byte(1.f)), hV(HD * SK, e4m3_byte(1.f));
  auto layoutSF  = BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim)));
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, int(kBlockN), int(kHeadDim), 1));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(SK, int(kBlockN), int(kHeadDim), 1));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), SK, 1));
  std::vector<uint8_t> hSFQ(cosize(layoutSFQ), 127), hSFK(cosize(layoutSFK), 127), hSFV(cosize(layoutSFV), 127);

  Element *dQ, *dK, *dV; ElementSF *dSFQ, *dSFK, *dSFV; float *dO, *dLSE, *dL; int* dKV;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size())); CK(cudaMalloc(&dV, hV.size()));
  CK(cudaMalloc(&dSFQ, hSFQ.size())); CK(cudaMalloc(&dSFK, hSFK.size())); CK(cudaMalloc(&dSFV, hSFV.size()));
  CK(cudaMalloc(&dO, (size_t)SQ * HD * sizeof(float))); CK(cudaMalloc(&dLSE, SQ * sizeof(float)));
  CK(cudaMalloc(&dL, SQ * sizeof(float))); CK(cudaMalloc(&dKV, n_tiles * sizeof(int)));
  int* dOrder; CK(cudaMalloc(&dOrder, n_tiles * sizeof(int)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, hV.data(), hV.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dKV, kv_len.data(), n_tiles * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dOrder, order.data(), n_tiles * sizeof(int), cudaMemcpyHostToDevice));

  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(SQ, HD, 1), make_stride(HD, _1{}, SQ * HD));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(SK, HD, 1), make_stride(HD, _1{}, SK * HD));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, SK, 1), make_stride(SK, _1{}, HD * SK));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSFV), layoutSFV);
  Params params;
  params.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  params.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  params.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  params.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  params.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kSFPadHD>{}), _1{});
  params.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kBlockN>{}), _1{});
  params.layout_sfq = layoutSFQ; params.layout_sfv = layoutSFV;
  params.seqlen_q = SQ; params.seqlen_k = SK; params.n_block_total = n_block_total; params.sm_scale = sm_scale;
  params.num_qo_heads = 1; params.num_kv_heads = 1;
  params.out_O = dO; params.out_lse = dLSE; params.out_l = dL;
  params.out_Ppre = nullptr; params.out_Mnb = nullptr; params.out_dbg = nullptr;
  params.tile_kv_len = dKV;

  double avg_blk = (double)tot / n_tiles, max_blk = *std::max_element(kvb.begin(), kvb.end());
  printf("dist=%-7s tiles=%d (%.1f waves, %d SMs) kv-blocks avg=%.1f max=%.0f | HD=%d SK=%d\n",
         dist, n_tiles, waves, num_sm, avg_blk, max_blk, HD, SK);

  float mn, mean, sd;
  using flashinfer::SingleTileScheduler; using flashinfer::StaticPersistentScheduler; using flashinfer::LPTPersistentScheduler;
  float tS = run<SingleTileScheduler>(params, {n_tiles, 1, SQ, SK, cutlass::FastDivmod(1)}, iters, mean, sd);
  printf("  single (HW)  min %.4f ms  (mean %.4f sd %.4f)\n", tS, mean, sd);
  float tB = run<StaticPersistentScheduler>(params, {n_tiles, 1, SQ, SK, cutlass::FastDivmod(1)}, iters, mean, sd);
  printf("  static       min %.4f ms  (mean %.4f sd %.4f)  %.2fx vs HW\n", tB, mean, sd, tB / tS);
  float tL = run<LPTPersistentScheduler>(params, {n_tiles, 1, SQ, SK, cutlass::FastDivmod(1), dOrder}, iters, mean, sd);
  printf("  lpt (sorted) min %.4f ms  (mean %.4f sd %.4f)  %.2fx vs HW\n", tL, mean, sd, tL / tS);
  printf("  => LPT vs HW: %.1f%% %s\n", 100.0 * (tS - tL) / tS, tL < tS ? "FASTER" : "slower");
  return 0;
}
