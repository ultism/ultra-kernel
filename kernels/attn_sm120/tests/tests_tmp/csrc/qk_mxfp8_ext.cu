// Torch extension wrapping the SM120a MXFP8 block-scaled QK kernel, for
// numerical validation against a torchao MX-quant baseline.
//
// Unlike the C++ smoke tests (which fill Q=K=1.0 and so cannot catch A/B data
// layout bugs), this loads REAL e4m3 data from gmem into the atom's A/B
// fragments via thr_mma.partition_A/B + copy -- the data-load path the smoke
// tests bypassed. Single warp, M=16, N=8, head_dim in {32,64,128}.

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
    cutlass::float_ue8m0_t, /*SFVecSize=*/32>;

// A: [16, HD] e4m3 bytes (row-major = TN A, K contiguous per row)
// B: [8,  HD] e4m3 bytes (row-major = TN B)
// SFA: [16, HD/32] ue8m0 bytes (one scale per (row, K-block))
// SFB: [8,  HD/32] ue8m0 bytes
template <int HEAD_DIM>
__global__ void qk_mxfp8_kernel(const uint8_t* A, const uint8_t* B,
                                const uint8_t* SFA, const uint8_t* SFB,
                                float* out_S) {
  using ue8m0 = cutlass::float_ue8m0_t;
  constexpr int NBLK = HEAD_DIM / 32;
  int tid = threadIdx.x;

  TiledMMA tiled_mma = make_tiled_mma(AtomMXF8{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Real data, loaded through the atom's A/B partitioning.
  Tensor gA = make_tensor(make_gmem_ptr(A),
                          make_layout(make_shape(_16{}, Int<HEAD_DIM>{}), LayoutRight{}));
  Tensor gB = make_tensor(make_gmem_ptr(B),
                          make_layout(make_shape( _8{}, Int<HEAD_DIM>{}), LayoutRight{}));

  Tensor tCrA = thr_mma.partition_fragment_A(gA);
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  copy(thr_mma.partition_A(gA), tCrA);
  copy(thr_mma.partition_B(gB), tCrB);

  // SF source over gmem: (MN,(32,NBLK)):(NBLK,(0,1)) -> SF[row*NBLK + blk],
  // broadcast across the 32 lanes of each K block.
  auto sfa_layout = make_layout(make_shape(_16{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  auto sfb_layout = make_layout(make_shape(_8{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  Tensor gSFA = make_tensor(make_gmem_ptr(reinterpret_cast<const ue8m0*>(SFA)), sfa_layout);
  Tensor gSFB = make_tensor(make_gmem_ptr(reinterpret_cast<const ue8m0*>(SFB)), sfb_layout);

  Tensor tCrSFA = mxfp8::partition_fragment_SFA(gSFA, thr_mma);
  Tensor tCrSFB = mxfp8::partition_fragment_SFB(gSFB, thr_mma);
  copy(mxfp8::partition_SFA(gSFA, thr_mma), tCrSFA);
  copy(mxfp8::partition_SFB(gSFB, thr_mma), tCrSFB);

  Tensor gS    = make_tensor(make_gmem_ptr(out_S),
                             make_layout(Shape<_16, _8>{}, Stride<_8, _1>{}));
  Tensor tCgS  = thr_mma.partition_C(gS);
  Tensor accum = thr_mma.partition_fragment_C(gS);
  clear(accum);

  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < K_BLOCK_MAX; ++k) {
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_, _, k), tCrSFA(_, _, k)),
               make_zip_tensor(tCrB(_, _, k), tCrSFB(_, _, k)),
               accum);
  }
  copy(accum, tCgS);
}

torch::Tensor qk_mxfp8(torch::Tensor A, torch::Tensor B,
                       torch::Tensor SFA, torch::Tensor SFB) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kUInt8 && SFA.dtype() == torch::kUInt8,
              "pass raw e4m3 / ue8m0 bytes as uint8");
  A = A.contiguous(); B = B.contiguous();
  SFA = SFA.contiguous(); SFB = SFB.contiguous();
  int M = A.size(0), HD = A.size(1), N = B.size(0);
  TORCH_CHECK(M == 16 && N == 8, "single-atom MN (M=16,N=8) for now");

  auto S = torch::empty({16, 8}, torch::dtype(torch::kFloat32).device(A.device()));
  const uint8_t* a = A.data_ptr<uint8_t>();
  const uint8_t* b = B.data_ptr<uint8_t>();
  const uint8_t* sfa = SFA.data_ptr<uint8_t>();
  const uint8_t* sfb = SFB.data_ptr<uint8_t>();
  float* s = S.data_ptr<float>();

  switch (HD) {
    case 32:  qk_mxfp8_kernel<32> <<<1, 32>>>(a, b, sfa, sfb, s); break;
    case 64:  qk_mxfp8_kernel<64> <<<1, 32>>>(a, b, sfa, sfb, s); break;
    case 128: qk_mxfp8_kernel<128><<<1, 32>>>(a, b, sfa, sfb, s); break;
    default: TORCH_CHECK(false, "head_dim must be 32/64/128, got ", HD);
  }
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "kernel failed: ", cudaGetErrorString(cudaGetLastError()));
  return S;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_mxfp8", &qk_mxfp8, "MXFP8 block-scaled QK (single-atom MN)");
}
