// QK single-tile smoke test for the SM120a MXFP8 block-scaled path.
//
// Goal: prove the *CuTe abstraction* (TiledMMA + block-scaled traits + the SF
// partition helpers + cute::gemm with zipped operands) composes and is
// numerically correct at an attention-QK tile shape -- not just the raw PTX
// atom (already verified separately in bench/mxf8_verify.cu).
//
// QK^T maps onto the m16n8k32 TN atom as: A = Q (M=seq_q rows, K=head_dim
// contiguous), B = K (N=seq_k rows, K=head_dim contiguous), C = S = Q.K^T.
// One atom covers head_dim = 32 (= one ue8m0 scale block).
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/qk_smoke.cu -o tests/qk_smoke

#include <cstdio>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t e_ = (call);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,    \
             __LINE__);                                                       \
      return 1;                                                               \
    }                                                                         \
  } while (0)

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
    cutlass::float_ue8m0_t, /*SFVecSize=*/32>;

// One warp computes a single m16n8k32 block. Q/K data are all 1.0 (e4m3); the
// per-block scales are uniform (scale_a, scale_b). Expected accumulator value:
//   S = sum_{k<32} (1 * scale_a) * (1 * scale_b) = 32 * scale_a * scale_b.
// Every thread writes its 4 accumulator regs; with uniform inputs all 128
// logical outputs are identical, so we can check them without inverting the
// C thread-value layout.
__global__ void qk_smoke_kernel(float scale_a, float scale_b, float* out) {
  TiledMMA tiled_mma = make_tiled_mma(AtomMXF8{});
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  // Shape-only reference tensors (data pointers unused: partition_fragment_*
  // returns fresh register fragments sized from these layouts).
  auto refA = make_identity_tensor(Shape<_16, _32>{});  // Q: (M, head_dim)
  auto refB = make_identity_tensor(Shape< _8, _32>{});  // K: (N, head_dim)
  auto refC = make_identity_tensor(Shape<_16,  _8>{});  // S: (M, N)

  Tensor tCrA  = thr_mma.partition_fragment_A(refA);     // (MMA, MMA_M, MMA_K)
  Tensor tCrB  = thr_mma.partition_fragment_B(refB);     // (MMA, MMA_N, MMA_K)
  Tensor accum = thr_mma.partition_fragment_C(refC);     // (MMA, MMA_M, MMA_N)

  // Block-scale fragments. For uniform scales we build the broadcast fragment
  // directly: logical K-extent 32, physical cosize 1 (one ue8m0 byte per K=32
  // block), which is what the block-scaled atom consumes (RegNumSFA == 1).
  // The smem-tiled partition_fragment_SFA/SFB path (for per-row/col scales) is
  // validated later, when the mainloop wires real SF shared memory.
  Tensor tCrSFA = make_tensor<cutlass::float_ue8m0_t>(
      Layout<Shape<_32, _1, _1>, Stride<_0, _0, _0>>{});
  Tensor tCrSFB = make_tensor<cutlass::float_ue8m0_t>(
      Layout<Shape<_32, _1, _1>, Stride<_0, _0, _0>>{});

  // e4m3 1.0 == 0x38; fill the 8-bit data fragments directly.
  fill(tCrA, uint8_t(0x38));
  fill(tCrB, uint8_t(0x38));
  fill(tCrSFA, cutlass::float_ue8m0_t(scale_a));
  fill(tCrSFB, cutlass::float_ue8m0_t(scale_b));
  clear(accum);

  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < K_BLOCK_MAX; ++k) {
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_, _, k), tCrSFA(_, _, k)),
               make_zip_tensor(tCrB(_, _, k), tCrSFB(_, _, k)),
               accum);
  }

  // Each thread stores its 4 accumulator registers contiguously.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    out[threadIdx.x * size(accum) + i] = accum(i);
  }
}

static int run_case(float scale_a, float scale_b, float expected) {
  float* d_out;
  const int N = 32 * 4;  // 32 threads * 4 accumulator regs
  CK(cudaMalloc(&d_out, N * sizeof(float)));
  qk_smoke_kernel<<<1, 32>>>(scale_a, scale_b, d_out);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  float h_out[N];
  CK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(d_out));

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != expected) ++bad;
  }
  printf("  scale_a=%.1f scale_b=%.1f  expected=%-7.1f got[0]=%-7.1f  %s (%d/%d ok)\n",
         scale_a, scale_b, expected, h_out[0],
         bad == 0 ? "PASS" : "FAIL", N - bad, N);
  return bad == 0 ? 0 : 1;
}

int main() {
  printf("QK single-tile MXFP8 block-scaled smoke test (m16n8k32, head_dim=32)\n");
  int fails = 0;
  fails += run_case(1.0f, 1.0f, 32.0f);    // all ones -> K
  fails += run_case(2.0f, 1.0f, 64.0f);    // SFA = 2^1
  fails += run_case(2.0f, 4.0f, 256.0f);   // SFA = 2^1, SFB = 2^2
  fails += run_case(0.5f, 0.5f, 8.0f);     // SFA = 2^-1, SFB = 2^-1
  printf(fails == 0 ? "ALL PASS\n" : "FAILURES: %d\n", fails);
  return fails;
}
