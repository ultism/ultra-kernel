// SF-smem validation for the SM120a MXFP8 block-scaled path.
//
// Extends tests/qk_smoke.cu (uniform scales, hand-built broadcast fragment) to
// the real path: per-row SFA / per-col SFB *non-uniform* ue8m0 scales staged in
// shared memory, partitioned into register fragments by the lifted
// partition_fragment_SFA/SFB helpers, copied smem->reg, and fed to cute::gemm.
//
// This exercises (a) the SF source layout (M,K)=(16/8, 32):(1,0) -- one ue8m0
// per row/col, broadcast across the 32-element K block -- and (b) that each
// thread's scale lands on the correct accumulator row/col. With Q=K=1.0 and
// per-row/col scales, S[m,n] = sum_{k<32} (1*sa[m])*(1*sb[n]) = 32*sa[m]*sb[n].
// We read the result back through thr_mma.partition_C, so the C thread-value
// inversion is validated too.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/qk_sf_smem.cu -o tests/qk_sf_smem

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

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

// One warp, one m16n8k32 atom. Q/K data are all 1.0 (e4m3). SFA/SFB are
// per-row / per-col ue8m0 scales delivered via gmem (sa[16], sb[8]).
__global__ void qk_sf_smem_kernel(const float* sa, const float* sb, float* out_S) {
  using ue8m0 = cutlass::float_ue8m0_t;
  __shared__ ue8m0 smemSFA[16];
  __shared__ ue8m0 smemSFB[8];

  int tid = threadIdx.x;
  if (tid < 16) smemSFA[tid] = ue8m0(sa[tid]);
  if (tid < 8)  smemSFB[tid] = ue8m0(sb[tid]);
  __syncthreads();

  TiledMMA tiled_mma = make_tiled_mma(AtomMXF8{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // SF source: logical (M/N, K) with K broadcast (stride 0) -> one scale per
  // row/col shared across the K=32 block. cosize = 16 / 8 physical bytes.
  Tensor sSFA = make_tensor(make_smem_ptr(smemSFA),
                            make_layout(Shape<_16, _32>{}, Stride<_1, _0>{}));
  Tensor sSFB = make_tensor(make_smem_ptr(smemSFB),
                            make_layout(Shape< _8, _32>{}, Stride<_1, _0>{}));

  auto refA = make_identity_tensor(Shape<_16, _32>{});
  auto refB = make_identity_tensor(Shape< _8, _32>{});

  Tensor tCrA   = thr_mma.partition_fragment_A(refA);
  Tensor tCrB   = thr_mma.partition_fragment_B(refB);
  Tensor tCrSFA = mxfp8::partition_fragment_SFA(sSFA, thr_mma);
  Tensor tCrSFB = mxfp8::partition_fragment_SFB(sSFB, thr_mma);

  fill(tCrA, uint8_t(0x38));  // e4m3 1.0
  fill(tCrB, uint8_t(0x38));
  copy(mxfp8::partition_SFA(sSFA, thr_mma), tCrSFA);  // smem -> reg
  copy(mxfp8::partition_SFB(sSFB, thr_mma), tCrSFB);

  Tensor gS    = make_tensor(make_gmem_ptr(out_S),
                             make_layout(Shape<_16, _8>{}, Stride<_8, _1>{}));  // row-major
  Tensor tCgS  = thr_mma.partition_C(gS);   // (MMA, MMA_M, MMA_N) gmem view
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

int main() {
  printf("QK SF-smem validation (m16n8k32, per-row SFA / per-col SFB)\n");

  float h_sa[16], h_sb[8];
  // Distinct exact powers of two so the ue8m0 (pure-exponent) encoding is exact.
  for (int m = 0; m < 16; ++m) h_sa[m] = std::ldexp(1.0f, (m % 4) - 1);  // {.5,1,2,4}
  for (int n = 0; n < 8; ++n)  h_sb[n] = std::ldexp(1.0f, (n % 3));      // {1,2,4}

  float *d_sa, *d_sb, *d_S;
  CK(cudaMalloc(&d_sa, 16 * sizeof(float)));
  CK(cudaMalloc(&d_sb, 8 * sizeof(float)));
  CK(cudaMalloc(&d_S, 16 * 8 * sizeof(float)));
  CK(cudaMemcpy(d_sa, h_sa, 16 * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sb, h_sb, 8 * sizeof(float), cudaMemcpyHostToDevice));

  qk_sf_smem_kernel<<<1, 32>>>(d_sa, d_sb, d_S);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  float h_S[16 * 8];
  CK(cudaMemcpy(h_S, d_S, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

  int bad = 0, first_m = -1, first_n = -1;
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 8; ++n) {
      float expected = 32.0f * h_sa[m] * h_sb[n];
      float got = h_S[m * 8 + n];
      if (got != expected) {
        if (bad == 0) { first_m = m; first_n = n; }
        ++bad;
      }
    }
  }

  if (bad == 0) {
    printf("  PASS (128/128 ok) -- per-row/col block scales map correctly\n");
    printf("  sample: S[0,0]=%.1f (exp %.1f)  S[3,2]=%.1f (exp %.1f)  S[15,7]=%.1f (exp %.1f)\n",
           h_S[0], 32.0f * h_sa[0] * h_sb[0],
           h_S[3 * 8 + 2], 32.0f * h_sa[3] * h_sb[2],
           h_S[15 * 8 + 7], 32.0f * h_sa[15] * h_sb[7]);
  } else {
    printf("  FAIL (%d/128 wrong); first bad at [m=%d,n=%d]: got=%.3f exp=%.3f\n",
           bad, first_m, first_n, h_S[first_m * 8 + first_n],
           32.0f * h_sa[first_m] * h_sb[first_n]);
  }

  CK(cudaFree(d_sa));
  CK(cudaFree(d_sb));
  CK(cudaFree(d_S));
  return bad == 0 ? 0 : 1;
}
