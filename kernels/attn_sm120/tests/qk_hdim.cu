// head_dim K-loop validation for the SM120a MXFP8 block-scaled path.
//
// Extends tests/qk_sf_smem.cu (single atom, head_dim=32) to head_dim 64 / 128,
// i.e. 2 / 4 atom-K accumulation steps. Each 32-element K block carries its own
// ue8m0 scale, so SFA is per-(row, k_block) and SFB per-(col, k_block). With
// Q=K=1.0 the reference is
//   S[m,n] = sum_{blk} sum_{k in blk} (1*sa[m,blk])*(1*sb[n,blk])
//          = sum_{blk} 32 * sa[m,blk] * sb[n,blk].
// This validates the K-loop accumulation and that the SF fragment advances to
// the right block each atom-K step. Spatial M/N multi-warp tiling is deferred
// to the mainloop step.
//
// The SF smem layout is hand-rolled (we own quantization, so we are not bound
// to cutlass's TMA-oriented Sm1xxBlockScaledConfig format): logical (MN, K) as
// (MN, (32, nblk)) with stride (nblk, (0, 1)) -- one ue8m0 per (row/col, block),
// broadcast across the 32 lanes of a block.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/qk_hdim.cu -o tests/qk_hdim

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

// One warp, M=16, N=8, head_dim = HEAD_DIM (= NBLK * 32 atom-K steps).
template <int HEAD_DIM>
__global__ void qk_hdim_kernel(const float* sa, const float* sb, float* out_S) {
  using ue8m0 = cutlass::float_ue8m0_t;
  constexpr int NBLK = HEAD_DIM / 32;
  __shared__ ue8m0 smemSFA[16 * NBLK];
  __shared__ ue8m0 smemSFB[8 * NBLK];

  int tid = threadIdx.x;
  for (int i = tid; i < 16 * NBLK; i += blockDim.x) smemSFA[i] = ue8m0(sa[i]);
  for (int i = tid; i < 8 * NBLK; i += blockDim.x)  smemSFB[i] = ue8m0(sb[i]);
  __syncthreads();

  TiledMMA tiled_mma = make_tiled_mma(AtomMXF8{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // SF source: (MN, (32, NBLK)) : (NBLK, (0, 1)) -> smemSF[row*NBLK + blk].
  auto sfa_layout = make_layout(make_shape(_16{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  auto sfb_layout = make_layout(make_shape(_8{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  Tensor sSFA = make_tensor(make_smem_ptr(smemSFA), sfa_layout);
  Tensor sSFB = make_tensor(make_smem_ptr(smemSFB), sfb_layout);

  // Shape-only refs with integer strides (null ptr: never dereferenced;
  // partition_fragment_A/B builds fresh register fragments). make_identity_tensor
  // here would give ScaledBasis strides that break make_fragment_like for
  // multi-atom-K layouts.
  auto refA = make_tensor(make_gmem_ptr((uint8_t const*)nullptr),
                          make_layout(make_shape(_16{}, Int<HEAD_DIM>{})));
  auto refB = make_tensor(make_gmem_ptr((uint8_t const*)nullptr),
                          make_layout(make_shape( _8{}, Int<HEAD_DIM>{})));

  Tensor tCrA   = thr_mma.partition_fragment_A(refA);   // (MMA, MMA_M, MMA_K=NBLK)
  Tensor tCrB   = thr_mma.partition_fragment_B(refB);   // (MMA, MMA_N, MMA_K=NBLK)
  Tensor tCrSFA = mxfp8::partition_fragment_SFA(sSFA, thr_mma);
  Tensor tCrSFB = mxfp8::partition_fragment_SFB(sSFB, thr_mma);

  fill(tCrA, uint8_t(0x38));  // e4m3 1.0
  fill(tCrB, uint8_t(0x38));
  copy(mxfp8::partition_SFA(sSFA, thr_mma), tCrSFA);
  copy(mxfp8::partition_SFB(sSFB, thr_mma), tCrSFB);

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

template <int HEAD_DIM>
static int run_case() {
  constexpr int NBLK = HEAD_DIM / 32;
  float h_sa[16 * NBLK], h_sb[8 * NBLK];
  // Distinct exact powers of two per (row/col, block).
  for (int m = 0; m < 16; ++m)
    for (int b = 0; b < NBLK; ++b)
      h_sa[m * NBLK + b] = std::ldexp(1.0f, ((m + b) % 4) - 1);  // {.5,1,2,4}
  for (int n = 0; n < 8; ++n)
    for (int b = 0; b < NBLK; ++b)
      h_sb[n * NBLK + b] = std::ldexp(1.0f, (n + 2 * b) % 3);    // {1,2,4}

  float *d_sa, *d_sb, *d_S;
  CK(cudaMalloc(&d_sa, sizeof(h_sa)));
  CK(cudaMalloc(&d_sb, sizeof(h_sb)));
  CK(cudaMalloc(&d_S, 16 * 8 * sizeof(float)));
  CK(cudaMemcpy(d_sa, h_sa, sizeof(h_sa), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sb, h_sb, sizeof(h_sb), cudaMemcpyHostToDevice));

  qk_hdim_kernel<HEAD_DIM><<<1, 32>>>(d_sa, d_sb, d_S);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  float h_S[16 * 8];
  CK(cudaMemcpy(h_S, d_S, sizeof(h_S), cudaMemcpyDeviceToHost));

  int bad = 0, fm = -1, fn = -1;
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 8; ++n) {
      float expected = 0.0f;
      for (int b = 0; b < NBLK; ++b)
        expected += 32.0f * h_sa[m * NBLK + b] * h_sb[n * NBLK + b];
      if (h_S[m * 8 + n] != expected) {
        if (bad == 0) { fm = m; fn = n; }
        ++bad;
      }
    }
  }

  if (bad == 0) {
    float e00 = 0, e_last = 0;
    for (int b = 0; b < NBLK; ++b) {
      e00 += 32.0f * h_sa[0 * NBLK + b] * h_sb[0 * NBLK + b];
      e_last += 32.0f * h_sa[15 * NBLK + b] * h_sb[7 * NBLK + b];
    }
    printf("  head_dim=%-3d (%d K-blocks): PASS (128/128)  S[0,0]=%.1f(exp %.1f)  S[15,7]=%.1f(exp %.1f)\n",
           HEAD_DIM, NBLK, h_S[0], e00, h_S[15 * 8 + 7], e_last);
  } else {
    float expected = 0.0f;
    for (int b = 0; b < NBLK; ++b)
      expected += 32.0f * h_sa[fm * NBLK + b] * h_sb[fn * NBLK + b];
    printf("  head_dim=%-3d: FAIL (%d/128 wrong); first [m=%d,n=%d] got=%.3f exp=%.3f\n",
           HEAD_DIM, bad, fm, fn, h_S[fm * 8 + fn], expected);
  }

  CK(cudaFree(d_sa));
  CK(cudaFree(d_sb));
  CK(cudaFree(d_S));
  return bad;
}

int main() {
  printf("QK head_dim K-loop validation (m16n8, per-(row/col, block) scales)\n");
  int fails = 0;
  fails += run_case<32>();
  fails += run_case<64>();
  fails += run_case<128>();
  printf(fails == 0 ? "ALL PASS\n" : "FAILURES\n");
  return fails == 0 ? 0 : 1;
}
