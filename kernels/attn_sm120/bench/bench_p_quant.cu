// Timing A/B for the S3 P-quant simplification (fixed scalar scale=256 vs dynamic
// per-32-block MX). Same kernel, same data, ONLY the P-quant path differs -- build
// twice and diff the kernel time:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include bench/bench_p_quant.cu -o bench/bench_p_fixed
//   (same line + -DS3_P_DYNAMIC_SCALE=1 ... -o bench/bench_p_dyn)
// Run: ./bench/bench_p_fixed [SQ] [SK] [causal 0/1] [iters]
// All debug dumps are off (out_Ppre/out_Mnb/out_dbg = nullptr) so the full-P gmem
// write does not pollute the measurement.
#include <algorithm>
#include "../tests/s3_kernel.cuh"

#ifndef S3_SCHEDULER
#define S3_SCHEDULER SingleTileScheduler   // override: -DS3_SCHEDULER=StaticPersistentScheduler
#endif
#define STR2(x) #x
#define STR(x) STR2(x)
using Scheduler = flashinfer::S3_SCHEDULER;

static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }
static int query_num_sm() { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n; }

template <bool Causal>
static void time_kernel(Params params, Scheduler::Arguments sa, int iters,
                        float& mn, float& mean, float& sd, int& regs, int& occ) {
  typename Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  dim3 grid = Scheduler::get_grid_dim(sa, query_num_sm());
  int smem_bytes = int(sizeof(SharedStorage));
  auto kern = s3_kernel<Scheduler, Causal>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kern); regs = fa.numRegs;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, kNThreads, smem_bytes);
  auto launch = [&] { kern<<<grid, kNThreads, smem_bytes>>>(params, sp); };
  for (int i = 0; i < 10; ++i) launch();          // warmup
  cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms[i], a, b);
  }
  std::sort(ms.begin(), ms.end());
  mn = ms[0];
  double s = 0; for (float v : ms) s += v; mean = s / iters;
  double v2 = 0; for (float v : ms) v2 += (v - mean) * (v - mean); sd = std::sqrt(v2 / iters);
}

int main(int argc, char** argv) {
  const int SQ = argc > 1 ? atoi(argv[1]) : 4096;
  const int SK = argc > 2 ? atoi(argv[2]) : 4096;
  const bool causal = argc > 3 ? atoi(argv[3]) : 0;
  const int iters = argc > 4 ? atoi(argv[4]) : 100;
  const int HD = kHeadDim;
  const int m_block_max = SQ / kBlockM, n_block_total = SK / kBlockN;
  const int NVK = SK / SFVecSize;
  const float sm_scale = 1.0f / std::sqrt((float)HD);

  std::vector<float> dataVals = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  std::vector<uint8_t> hQ(SQ * HD), hK(SK * HD), hV(HD * SK);
  for (size_t i = 0; i < hQ.size(); ++i) hQ[i] = e4m3_byte(dataVals[i % dataVals.size()]);
  for (size_t i = 0; i < hK.size(); ++i) hK[i] = e4m3_byte(dataVals[(i * 7) % dataVals.size()]);
  for (size_t i = 0; i < hV.size(); ++i) hV[i] = e4m3_byte(dataVals[(i * 5) % dataVals.size()]);

  auto layoutSF  = BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim)));
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, int(kBlockN), int(kHeadDim), 1));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(SK, int(kBlockN), int(kHeadDim), 1));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), SK, 1));
  std::vector<uint8_t> hSFQ(cosize(layoutSFQ), 127), hSFK(cosize(layoutSFK), 127), hSFV(cosize(layoutSFV), 127);

  Element *dQ, *dK, *dV; ElementSF *dSFQ, *dSFK, *dSFV; float *dO, *dLSE, *dL;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size())); CK(cudaMalloc(&dV, hV.size()));
  CK(cudaMalloc(&dSFQ, hSFQ.size())); CK(cudaMalloc(&dSFK, hSFK.size())); CK(cudaMalloc(&dSFV, hSFV.size()));
  CK(cudaMalloc(&dO, (size_t)SQ * HD * sizeof(float))); CK(cudaMalloc(&dLSE, SQ * sizeof(float)));
  CK(cudaMalloc(&dL, SQ * sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, hV.data(), hV.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice));

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
  params.out_Ppre = nullptr; params.out_Mnb = nullptr; params.out_dbg = nullptr;   // timing: no dumps
  params.tile_kv_len = nullptr;

  Scheduler::Arguments sa{m_block_max, 1, SQ, SK, cutlass::FastDivmod(1)};
  float mn, mean, sd; int regs, occ;
  if (causal) time_kernel<true >(params, sa, iters, mn, mean, sd, regs, occ);
  else        time_kernel<false>(params, sa, iters, mn, mean, sd, regs, occ);

  // attention FLOPs: QK (2*SQ*SK*HD) + PV (2*SQ*SK*HD); causal ~ half the n-blocks.
  double flop = 4.0 * (double)SQ * SK * HD * (causal ? 0.5 : 1.0);
  const char* mode = kPDynamicScale ? "DYN-amax" : (kPConstSF ? "FIXED+constSF" : "FIXED+smemSF");
  int grid_x = Scheduler::get_grid_dim(sa, query_num_sm()).x;
  printf("sched=%-22s P=%-13s tiles=%d causal=%d | grid=%d CTAs %dreg occ=%d/SM | "
         "min %.4f mean %.4f sd %.4f ms  %.1f TFLOP/s\n",
         STR(S3_SCHEDULER), mode, m_block_max, (int)causal, grid_x, regs, occ,
         mn, mean, sd, flop / (mn * 1e-3) / 1e12);
  return 0;
}
