// S3: end-to-end mxfp8 prefill inner loop = S1 (multi-n_block online softmax) +
// S2 (P-requant + PV) glued, with the two pieces neither earlier milestone
// exercised:
//   (1) ONLINE accO rescale  -- before each block's PV accumulate, accO *= scores_scale
//       (per row), so the running-max renormalization telescopes onto the O
//       accumulator exactly as flash-attention. S1 had no O; S2 had no rescale.
//   (2) LSE output           -- LSE = M_last*sm_scale + ln(row_sum)  (natural log,
//       == logsumexp(S*sm_scale); matches flash-attn / torchao convention).
// K AND V now ride a kStages TMA ring (S2's V was single-buffered). sP/sSFP stay a
// per-nb scratch transpose buffer; a NamedBarrier *before* the P-write protects the
// previous nb's PV readers from the next nb overwriting it.
//
// head_dim == kBlockN here, so V's block-scale SF layout coincides with K's (S2
// trick). head_dim=64 (where it doesn't) is the next step.
//
// PRIMARY validation (bit-exact, no torch): device dumps per-block pre-quant fp32 P,
// the running-max sequence M_nb, and row_sum. The host REPLAYS the online algo from
// those dumps with the identical e4m3/ue8m0 rule:
//   O[m,h] = (1/l[m]) * sum_nb exp2((M_nb - M_last)*scale_log2)
//                        * sum_{n in nb} dequant(to_mx(Ppre[m, 32-blk]))[n] * Vdq[h,n]
//   l[m]   = sum_nb exp2((M_nb - M_last)*scale_log2) * sum_{n in nb} Ppre[m,n]
//   lse[m] = M_last*sm_scale + ln(l[m])
// This isolates the online-accumulation + LSE arithmetic from fp32-vs-fp64 quant-scale
// noise (host re-quantizes the device's OWN P floats). Causal + non-causal, multiple
// n_blocks so the rescale path is genuinely exercised.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s3_e2e.cu -o tests/s3_e2e


#include <cstdlib>
#include "s3_kernel.cuh"


// ---- host ----
#ifndef S3_SCHEDULER
#define S3_SCHEDULER SingleTileScheduler   // override: -DS3_SCHEDULER=StaticPersistentScheduler
#endif
static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }
static int query_num_sm() {
  if (const char* e = std::getenv("S3_NUM_SM")) return std::atoi(e);   // force a small grid to test multi-tile looping
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n;
}

template <typename Scheduler, bool Causal>
static int run_case(const char* tag, Params params, typename Scheduler::Arguments sched_args,
                    int SQ, int SK, int HD, int n_block_total, float sm_scale,
                    const std::vector<uint8_t>& hV, const std::vector<int>& vexp,
                    const std::vector<int>& tile_kv_len = {}) {
  typename Scheduler::Params sp = Scheduler::to_underlying_arguments(sched_args);
  dim3 grid = Scheduler::get_grid_dim(sched_args, query_num_sm());
  int smem_bytes = int(sizeof(SharedStorage));
  CK(cudaFuncSetAttribute(s3_kernel<Scheduler, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  s3_kernel<Scheduler, Causal><<<grid, kNThreads, smem_bytes>>>(params, sp);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<float> hO(SQ * HD), hLSE(SQ), hL(SQ), hPpre(SQ * SK), hMnb(SQ * n_block_total);
  CK(cudaMemcpy(hO.data(), params.out_O, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hLSE.data(), params.out_lse, hLSE.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hL.data(), params.out_l, hL.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hPpre.data(), params.out_Ppre, hPpre.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hMnb.data(), params.out_Mnb, hMnb.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float sm_scale_log2 = sm_scale * kLog2e;
  const int NVK = SK / SFVecSize;   // V SF blocks along the full keys axis (matches host vexp stride)
  auto deqV = [&](int h, int n) { return double(float(reinterpret_cast<const cutlass::float_e4m3_t&>(hV[h * SK + n]))) * std::ldexp(1.0, vexp[h * NVK + n / SFVecSize]); };
  auto host_se = [&](float amax) -> int { if (!(amax > 0.f)) return -127; uint32_t b; std::memcpy(&b, &amax, 4); return mx_scale_exp_bits(b); };

  // Replay the online accumulation from the device's own dumped P / M_nb / row_sum.
  std::vector<double> Oref(SQ * HD, 0.0), Lref(SQ, 0.0), LSEref(SQ, 0.0);
  for (int m = 0; m < SQ; ++m) {
    int m_block = m / kBlockM;
    int nb_tile = tile_kv_len.empty() ? n_block_total : (tile_kv_len[m_block] + kBlockN - 1) / kBlockN;
    int n_block_max = Causal ? std::min(nb_tile, ((m_block + 1) * kBlockM + kBlockN - 1) / kBlockN) : nb_tile;
    double Mlast = hMnb[m * n_block_total + (n_block_max - 1)];
    double l = 0.0;
    std::vector<double> O(HD, 0.0);
    for (int nb = 0; nb < n_block_max; ++nb) {
      double rescale = std::exp2((double)sm_scale_log2 * (hMnb[m * n_block_total + nb] - Mlast));
      // per-32-key-block re-quant of the device's pre-quant P (bit-identical rule).
      double Pq[kBlockN];
      double Praw[kBlockN];
      for (int kb = 0; kb < NKB; ++kb) {
        int se;
        if (kPDynamicScale) {
          float amax = 0.f;
          for (int j = 0; j < SFVecSize; ++j) amax = std::max(amax, std::fabs(hPpre[m * SK + nb * kBlockN + kb * SFVecSize + j]));
          se = host_se(amax);
        } else {
          se = kPScaleExp;   // fixed scale 256.0, matching the kernel
        }
        double scale = std::ldexp(1.0, se), inv = std::ldexp(1.0, -se);
        for (int j = 0; j < SFVecSize; ++j) {
          int n = kb * SFVecSize + j;
          float praw = hPpre[m * SK + nb * kBlockN + n];
          Praw[n] = praw;
          Pq[n] = double(float(cutlass::float_e4m3_t(praw * (float)inv))) * scale;
        }
      }
      for (int n = 0; n < kBlockN; ++n) l += rescale * Praw[n];
      for (int h = 0; h < HD; ++h) {
        double acc = 0;
        for (int n = 0; n < kBlockN; ++n) acc += Pq[n] * deqV(h, nb * kBlockN + n);
        O[h] += rescale * acc;
      }
    }
    Lref[m] = l;
    LSEref[m] = Mlast * sm_scale + std::log(l);
    for (int h = 0; h < HD; ++h) Oref[m * HD + h] = O[h] / l;
  }

  double mr_o = 0, ma_o = 0, mr_l = 0, ma_lse = 0; int bad = 0, fm = -1, fh = -1;
  for (int m = 0; m < SQ; ++m) {
    double rl = std::abs(hL[m] - Lref[m]) / std::max(1e-4, std::abs(Lref[m]));
    mr_l = std::max(mr_l, rl);
    ma_lse = std::max(ma_lse, std::abs((double)hLSE[m] - LSEref[m]));
    for (int h = 0; h < HD; ++h) {
      double r = Oref[m * HD + h], g = hO[m * HD + h];
      double rel = std::abs(g - r) / std::max(1e-4, std::abs(r));
      ma_o = std::max(ma_o, std::abs(g - r));
      if (rel > mr_o) mr_o = rel;
      if (rel > 5e-3 && std::abs(g - r) > 1e-4) { if (!bad) { fm = m; fh = h; } ++bad; }
    }
  }
  printf("  [%s] O max|rel|=%.3g max|abs|=%.3g | l max|rel|=%.3g | lse max|abs|=%.3g | bad(O>5e-3)=%d/%d\n",
         tag, mr_o, ma_o, mr_l, ma_lse, bad, SQ * HD);
  if (bad) printf("    first bad O[%d,%d]: got=%.5f exp=%.5f\n", fm, fh, hO[fm * HD + fh], Oref[fm * HD + fh]);
  return bad == 0 ? 0 : 1;
}

int main() {
  constexpr int m_block_max = 3, n_block_total = 4;
  const int SQ = m_block_max * kBlockM, SK = n_block_total * kBlockN, HD = kHeadDim;
  const int NVK = SK / SFVecSize;   // V SF blocks along the FULL keys axis (= 16), not per-128-block
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  printf("S3: end-to-end mxfp8 prefill (online softmax + accO rescale + P-requant + PV + O/LSE)\n");
  printf("  seqlen_q=%d (%d m_blocks), seqlen_k=%d (%d n_blocks), head_dim=%d, kStages=%d, sm_scale=%.4f\n",
         SQ, m_block_max, SK, n_block_total, HD, kStages, sm_scale);

  std::vector<float> dataVals = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  std::vector<uint8_t> hQ(SQ * HD), hK(SK * HD), hV(HD * SK);   // hV = V^T [head_dim, keys]
  std::vector<int> qexp(SQ * NBLK), kexp(SK * NBLK), vexp(HD * NVK);
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) qexp[m * NBLK + b] = ((m + b) % 4) - 1;
  // FLAT scales: every n_block's max is comparable so ALL blocks contribute materially to O.
  // (The earlier "+ n/kBlockN" growing-max data underflowed the rescale and collapsed O onto
  //  block 0 -- the only block whose V-SF happened to be correct -- masking the multi-block bug.)
  for (int n = 0; n < SK; ++n) for (int b = 0; b < NBLK; ++b) kexp[n * NBLK + b] = (n + 2 * b) % 3;
  for (int h = 0; h < HD; ++h) for (int b = 0; b < NVK; ++b) vexp[h * NVK + b] = ((h + b) % 3) - 1;
  for (int m = 0; m < SQ; ++m) for (int k = 0; k < HD; ++k) hQ[m * HD + k] = e4m3_byte(dataVals[(m * HD + k) % dataVals.size()]);
  for (int n = 0; n < SK; ++n) for (int k = 0; k < HD; ++k) hK[n * HD + k] = e4m3_byte(dataVals[(n * 7 + k * 3) % dataVals.size()]);
  for (int h = 0; h < HD; ++h) for (int n = 0; n < SK; ++n) hV[h * SK + n] = e4m3_byte(dataVals[(h * 5 + n * 2) % dataVals.size()]);

  // SF layouts carry a trailing head (L) mode; this single-head test uses L=1.
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, int(kBlockN), int(kHeadDim), 1));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(SK, int(kBlockN), int(kHeadDim), 1));
  // V's OWN SFB layout: (head_dim, keys) over the full seqlen_k, scale along keys.
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), SK, 1));
  std::vector<uint8_t> hSFQ(cosize(layoutSFQ), 0), hSFK(cosize(layoutSFK), 0), hSFV(cosize(layoutSFV), 0);
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) hSFQ[layoutSFQ(make_coord(m, b * SFVecSize, 0))] = ue8m0_byte_pow2(qexp[m * NBLK + b]);
  for (int n = 0; n < SK; ++n) for (int b = 0; b < NBLK; ++b) hSFK[layoutSFK(make_coord(n, b * SFVecSize, 0))] = ue8m0_byte_pow2(kexp[n * NBLK + b]);
  // V SF: index (head_dim_row, key_col), scale block along keys -> spans all NVK=16 key blocks.
  for (int h = 0; h < HD; ++h) for (int b = 0; b < NVK; ++b) hSFV[layoutSFV(make_coord(h, b * SFVecSize, 0))] = ue8m0_byte_pow2(vexp[h * NVK + b]);

  Element *dQ, *dK, *dV; ElementSF *dSFQ, *dSFK, *dSFV;
  float *dO, *dLSE, *dL, *dPpre, *dMnb;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size())); CK(cudaMalloc(&dV, hV.size()));
  CK(cudaMalloc(&dSFQ, hSFQ.size())); CK(cudaMalloc(&dSFK, hSFK.size())); CK(cudaMalloc(&dSFV, hSFV.size()));
  CK(cudaMalloc(&dO, SQ * HD * sizeof(float))); CK(cudaMalloc(&dLSE, SQ * sizeof(float)));
  CK(cudaMalloc(&dL, SQ * sizeof(float))); CK(cudaMalloc(&dPpre, SQ * SK * sizeof(float)));
  CK(cudaMalloc(&dMnb, SQ * n_block_total * sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, hV.data(), hV.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice));

  // 3D data tensors (..., head); single-head test uses extent 1.
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
  params.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{});
  params.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{});
  params.layout_sfq = layoutSFQ; params.layout_sfv = layoutSFV;
  params.seqlen_q = SQ; params.seqlen_k = SK; params.n_block_total = n_block_total; params.sm_scale = sm_scale;
  params.num_qo_heads = 1; params.num_kv_heads = 1;
  params.out_O = dO; params.out_lse = dLSE; params.out_l = dL; params.out_Ppre = dPpre; params.out_Mnb = dMnb; params.out_dbg = nullptr;
  params.tile_kv_len = nullptr;

  using Scheduler = flashinfer::S3_SCHEDULER;
  Scheduler::Arguments sa{m_block_max, 1, SQ, SK, cutlass::FastDivmod(1)};
  dim3 grid = Scheduler::get_grid_dim(sa, query_num_sm());
  printf("  grid=(%u,%u,%u) smem=%d threads=%d\n", grid.x, grid.y, grid.z, int(sizeof(SharedStorage)), kNThreads);

  // Optional variable per-tile kv_len (varlen-cost proxy for the scheduler, S4b-1).
  // S3_VARKV=1 -> each q-tile attends a different #key-blocks (multiples of kBlockN);
  // exercises the per-tile n_block_max path the LPT scheduler will balance.
  std::vector<int> tile_kv;
  int* d_tile_kv = nullptr;
  if (std::getenv("S3_VARKV")) {
    tile_kv.resize(m_block_max);
    for (int t = 0; t < m_block_max; ++t) tile_kv[t] = std::min(SK, ((t % n_block_total) + 1) * kBlockN);
    CK(cudaMalloc(&d_tile_kv, sizeof(int) * m_block_max));
    CK(cudaMemcpy(d_tile_kv, tile_kv.data(), sizeof(int) * m_block_max, cudaMemcpyHostToDevice));
    params.tile_kv_len = d_tile_kv;
    printf("  VARKV per-tile n_blocks:");
    for (int t = 0; t < m_block_max; ++t) printf(" %d", tile_kv[t] / kBlockN);
    printf("\n");
  }

  int rc = 0;
  rc |= run_case<Scheduler, false>("non-causal", params, sa, SQ, SK, HD, n_block_total, sm_scale, hV, vexp, tile_kv);
  rc |= run_case<Scheduler, true >("causal    ", params, sa, SQ, SK, HD, n_block_total, sm_scale, hV, vexp, tile_kv);
  printf(rc == 0 ? "S3 PASS\n" : "S3 FAIL\n");
  return rc;
}

