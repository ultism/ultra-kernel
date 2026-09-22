// Torchao independent oracle for the S3 WS+TMA MXFP8 prefill kernel.
//
// Exposes the EXACT kernel that tests/s3_e2e.cu validates by self-replay
// (tests/s3_kernel.cuh) to Python, driven by torchao to_mx-quantized Q/K/V.
// The Python side (tests/test_mxfp8_prefill.py::test_attn_matches_torchao)
// dequantizes with torchao and replays the online attention as an external
// reference -- so this validates the kernel's quant rule + full QK->softmax->
// P-requant->PV->O/LSE pipeline against the authoritative OCP MX definition,
// independent of the C++ self-replay.
//
// e4m3 data is passed as raw uint8 (cutlass float_e4m3_t storage == torchao
// float8_e4m3fn byte); ue8m0 scales as raw uint8 (biased exponent, identical
// in cutlass float_ue8m0_t and torchao float8_e8m0). The host scatters the
// row-major torchao scale tensors into cutlass tile_atom_to_shape_SF* layouts
// (exactly as tests/s3_e2e.cu's host does), then builds the same TMA params.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#include "../s3_kernel.cuh"

using Scheduler = flashinfer::SingleTileScheduler;

#define ACHECK(call)                                                          \
  TORCH_CHECK((call) == cudaSuccess, "CUDA error: ",                          \
              cudaGetErrorString(cudaGetLastError()), " @ ", __LINE__)

// Scatter a row-major [rows, blocks] torchao ue8m0 scale tensor (CPU uint8)
// into a cutlass SF tiled layout, indexed by logical (row, block*SFVecSize).
template <class Layout>
static std::vector<uint8_t> place_sf(const torch::Tensor& s_cpu, Layout layout,
                                     int rows, int blocks) {
  std::vector<uint8_t> h(cosize(layout), 0);
  auto a = s_cpu.accessor<uint8_t, 2>();
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < blocks; ++b)
      h[layout(make_coord(r, b * SFVecSize, 0))] = a[r][b];   // single-head: L=0
  return h;
}

std::vector<torch::Tensor> mxfp8_attn(
    torch::Tensor Qd, torch::Tensor Kd, torch::Tensor Vd,    // e4m3 uint8: [Sq,d] [Sk,d] [d,Sk]
    torch::Tensor Qs, torch::Tensor Ks, torch::Tensor Vs,    // ue8m0 uint8: [Sq,d/32] [Sk,d/32] [d,Sk/32]
    double sm_scale, bool causal, int64_t kv_len = -1,
    // S6b: per-tensor fp8 cache (kUniformFp8). uniform_sf=True -> the kernel SKIPS the SF TMA loads
    // and synthesizes a uniform byte-127 (2^0) scale, so the Qs/Ks/Vs passed here are IGNORED (the
    // tests pass 0xFF poison to PROVE it). The caller folds q_scale*k_scale into sm_scale and passes
    // v_scale as o_scale (O *= o_scale). Defaults reproduce the block-scaled mxfp8 path exactly.
    bool uniform_sf = false, double o_scale = 1.0,
    // S7 timing: bench_iters>0 -> skip the Ppre/dbg gmem dump (oracle-only) and instead run
    // warmup + bench_iters timed kernel launches (CUDA events), returning a 5th tensor = ms/iter.
    int64_t bench_iters = 0) {
  const bool bench = bench_iters > 0;
  TORCH_CHECK(Qd.is_cuda() && Kd.is_cuda() && Vd.is_cuda(), "data must be CUDA");
  TORCH_CHECK(Qd.dtype() == torch::kUInt8 && Qs.dtype() == torch::kUInt8, "pass raw e4m3/ue8m0 bytes");
  Qd = Qd.contiguous(); Kd = Kd.contiguous(); Vd = Vd.contiguous();
  const int SQ = Qd.size(0), HD = Qd.size(1), SK = Kd.size(0);
  // SK is the PADDED (128-tiled) extent for addressing; KVLEN is the real key count the
  // mask honors. KVLEN<SK exercises the partial-last-block mask + V kFillZero with the
  // padded tail [KVLEN, SK) free to carry garbage (the test injects 0xFF there).
  const int KVLEN = (kv_len < 0) ? SK : int(kv_len);
  TORCH_CHECK(HD == kHeadDim, "head_dim must be ", int(kHeadDim), ", got ", HD);
  TORCH_CHECK(SQ % kBlockM == 0 && SK % kBlockN == 0, "padded seqlens must tile 128");
  TORCH_CHECK(KVLEN > 0 && KVLEN <= SK, "kv_len must be in (0, SK]");
  TORCH_CHECK(Vd.size(0) == HD && Vd.size(1) == SK, "V must be [head_dim, seqlen_k]");
  const int n_block_total = SK / kBlockN, m_block_max = SQ / kBlockM;
  const int NBLK = HD / SFVecSize, NVK = SK / SFVecSize;

  // ---- SF gmem: scatter torchao scales into cutlass tiled layouts ----
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(int(SQ), int(kBlockN), int(kHeadDim), 1));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(int(SK), int(kBlockN), int(kHeadDim), 1));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), int(SK), 1));
  auto hSFQ = place_sf(Qs.to(torch::kCPU).contiguous(), layoutSFQ, SQ, NBLK);
  auto hSFK = place_sf(Ks.to(torch::kCPU).contiguous(), layoutSFK, SK, NBLK);
  auto hSFV = place_sf(Vs.to(torch::kCPU).contiguous(), layoutSFV, HD, NVK);

  ElementSF *dSFQ, *dSFK, *dSFV;
  ACHECK(cudaMalloc(&dSFQ, hSFQ.size())); ACHECK(cudaMalloc(&dSFK, hSFK.size())); ACHECK(cudaMalloc(&dSFV, hSFV.size()));
  ACHECK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  ACHECK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));
  ACHECK(cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice));

  // ---- outputs (Ppre/Mnb are written unconditionally by the kernel; unused here) ----
  auto fopt = torch::dtype(torch::kFloat32).device(Qd.device());
  auto O   = torch::empty({SQ, HD}, fopt);
  auto LSE = torch::empty({SQ}, fopt);
  auto Pdbg = torch::zeros({SQ, SK}, fopt);   // kernel's dequantized requant-P (debug oracle)
  auto Ppre = torch::zeros({SQ, SK}, fopt);   // kernel's pre-quant fp32 P
  float *dL, *dMnb;
  ACHECK(cudaMalloc(&dL, SQ * sizeof(float)));
  ACHECK(cudaMalloc(&dMnb, size_t(SQ) * n_block_total * sizeof(float)));

  auto* dQ = reinterpret_cast<Element*>(Qd.data_ptr<uint8_t>());
  auto* dK = reinterpret_cast<Element*>(Kd.data_ptr<uint8_t>());
  auto* dV = reinterpret_cast<Element*>(Vd.data_ptr<uint8_t>());

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
  params.seqlen_q = SQ; params.seqlen_k = SK; params.n_block_total = n_block_total; params.sm_scale = float(sm_scale);
  params.o_scale = float(o_scale);     // S6b kUniformFp8: per-tensor v_scale (1.0 for the mxfp8 path)
  params.num_qo_heads = 1; params.num_kv_heads = 1;
  params.out_O = O.data_ptr<float>(); params.out_lse = LSE.data_ptr<float>();
  // bench: nullptr Ppre/dbg -> kernel skips the full-P [SQ,SK] gmem dump that otherwise dominates time.
  params.out_l = dL; params.out_Mnb = dMnb;
  params.out_Ppre = bench ? nullptr : Ppre.data_ptr<float>();
  params.out_dbg  = bench ? nullptr : Pdbg.data_ptr<float>();
  params.tile_kv_len = nullptr;

  Scheduler::Arguments sa{m_block_max, 1, SQ, KVLEN, cutlass::FastDivmod(1)};
  Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  dim3 grid = Scheduler::get_grid_dim(sa, 132);
  int smem = int(sizeof(SharedStorage));

  auto launch = [&](auto causal_c, auto uniform_c) {
    constexpr bool C = decltype(causal_c)::value;
    constexpr SFSource S = decltype(uniform_c)::value ? SFSource::kUniformFp8 : SFSource::kMxFp8;
    ACHECK(cudaFuncSetAttribute(s3_kernel<Scheduler, C, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    s3_kernel<Scheduler, C, S><<<grid, kNThreads, smem>>>(params, sp);
  };
  auto dispatch_sf = [&](auto causal_c) {
    if (uniform_sf) launch(causal_c, std::true_type{}); else launch(causal_c, std::false_type{});
  };
  auto run_once = [&]{ if (causal) dispatch_sf(std::true_type{}); else dispatch_sf(std::false_type{}); };
  float ms_per_iter = 0.f;
  if (bench) {
    for (int i = 0; i < 10; ++i) run_once();            // warmup
    ACHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    ACHECK(cudaEventRecord(e0));
    for (int i = 0; i < bench_iters; ++i) run_once();
    ACHECK(cudaEventRecord(e1));
    ACHECK(cudaEventSynchronize(e1));
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    ms_per_iter = ms / float(bench_iters);
  } else {
    run_once();
  }
  ACHECK(cudaGetLastError());
  ACHECK(cudaDeviceSynchronize());

  cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV); cudaFree(dL); cudaFree(dMnb);
  if (bench) return {O, LSE, Pdbg, Ppre, torch::full({1}, double(ms_per_iter), torch::kFloat32)};
  return {O, LSE, Pdbg, Ppre};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp8_attn", &mxfp8_attn,
        "S3 WS+TMA MXFP8 prefill attention (torchao oracle)",
        pybind11::arg("Qd"), pybind11::arg("Kd"), pybind11::arg("Vd"),
        pybind11::arg("Qs"), pybind11::arg("Ks"), pybind11::arg("Vs"),
        pybind11::arg("sm_scale"), pybind11::arg("causal"), pybind11::arg("kv_len") = -1,
        pybind11::arg("uniform_sf") = false, pybind11::arg("o_scale") = 1.0,
        pybind11::arg("bench_iters") = 0);
}
