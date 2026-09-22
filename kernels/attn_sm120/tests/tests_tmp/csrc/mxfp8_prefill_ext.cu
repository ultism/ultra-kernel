// SM120a MXFP8 block-scaled prefill attention -- staged build (design step 1).
//
// Grown sub-milestone by sub-milestone, each validated against a torchao oracle
// (see tests/test_mxfp8_prefill.py). The kernel here is the *non-WS* numerical
// core: one CTA owns one (m_block, head, batch) query tile, streams kBlockN-wide
// key blocks. Layouts (TiledMMA, SF, accumulators) are A-form so the later
// WS+TMA wrap adds code rather than rewriting.
//
// Stage 1a (this file): full-tile QK only. S = Q @ K^T through the block-scaled
// atom at kBlockM=128 x kBlockN=128, head_dim 64/128, 8-warp TiledMMA. Validates
// that MN tiling + tile-scale SF stay bit-exact at 256 threads. Later stages add
// the n-loop + online softmax (1b), P requant (1c), PV (1d), fuse + LSE (1e).

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

// 8 warps stacked along M: each warp owns 16 contiguous query rows and ALL key
// columns, so a row's softmax reduction is warp-local (design §3).
using TiledMMA_QK = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}));

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;

#define MX_LOG2E 1.4426950408889634f

// ---- softmax reduction helpers (m16n8 acc fragment; FA2 idiom) ----
// Convert an MMA accumulator (MMA=4, MMA_M, MMA_N) to a logical (row, col) view
// ((2,MMA_M), (2,MMA_N)): get<0>=rows owned by this thread, get<1>=its columns.
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2,2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
}

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Op>
  static __device__ __forceinline__ T run(T x, Op& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};
template <>
struct Allreduce<2> {
  template <typename T, typename Op>
  static __device__ __forceinline__ T run(T x, Op& op) {
    return op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
  }
};

struct MaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct SumOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };

// Reduce a per-thread row stat across the 4 lanes sharing each row (quad).
template <typename Tensor, typename Op>
__forceinline__ __device__ void quad_allreduce(Tensor& t, Op& op) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(t); ++i) t(i) = Allreduce<4>::run(t(i), op);
}

// -------- Stage 1a: full-tile QK --------
// A (Q): [kBlockM, HD] e4m3 bytes, row-major (TN, K=head_dim contiguous).
// B (K): [kBlockN, HD] e4m3 bytes, row-major.
// SFA:   [kBlockM, HD/32] ue8m0 bytes (one scale per (row, head_dim-block)).
// SFB:   [kBlockN, HD/32] ue8m0 bytes.
// out_S: [kBlockM, kBlockN] f32.
template <int HEAD_DIM>
__global__ void qk_tile_kernel(const uint8_t* A, const uint8_t* B,
                               const uint8_t* SFA, const uint8_t* SFB,
                               float* out_S) {
  using ue8m0 = cutlass::float_ue8m0_t;
  constexpr int NBLK = HEAD_DIM / 32;
  int tid = threadIdx.x;

  TiledMMA_QK tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);

  Tensor gA = make_tensor(make_gmem_ptr(A),
                          make_layout(make_shape(Int<kBlockM>{}, Int<HEAD_DIM>{}), LayoutRight{}));
  Tensor gB = make_tensor(make_gmem_ptr(B),
                          make_layout(make_shape(Int<kBlockN>{}, Int<HEAD_DIM>{}), LayoutRight{}));

  Tensor tCrA = thr_mma.partition_fragment_A(gA);   // (MMA, MMA_M=1, MMA_K=NBLK)
  Tensor tCrB = thr_mma.partition_fragment_B(gB);   // (MMA, MMA_N=16, MMA_K=NBLK)
  copy(thr_mma.partition_A(gA), tCrA);
  copy(thr_mma.partition_B(gB), tCrB);

  // SF source: (MN,(32,NBLK)):(NBLK,(0,1)) -> SF[row*NBLK + blk], broadcast
  // across the 32 lanes of each head_dim block (validated layout).
  auto sfa_layout = make_layout(make_shape(Int<kBlockM>{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  auto sfb_layout = make_layout(make_shape(Int<kBlockN>{}, make_shape(_32{}, Int<NBLK>{})),
                                make_stride(Int<NBLK>{}, make_stride(_0{}, _1{})));
  Tensor gSFA = make_tensor(make_gmem_ptr(reinterpret_cast<const ue8m0*>(SFA)), sfa_layout);
  Tensor gSFB = make_tensor(make_gmem_ptr(reinterpret_cast<const ue8m0*>(SFB)), sfb_layout);

  Tensor tCrSFA = mxfp8::partition_fragment_SFA(gSFA, thr_mma);
  Tensor tCrSFB = mxfp8::partition_fragment_SFB(gSFB, thr_mma);
  copy(mxfp8::partition_SFA(gSFA, thr_mma), tCrSFA);
  copy(mxfp8::partition_SFB(gSFB, thr_mma), tCrSFB);

  Tensor gS    = make_tensor(make_gmem_ptr(out_S),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), LayoutRight{}));
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

torch::Tensor qk_tile(torch::Tensor A, torch::Tensor B,
                      torch::Tensor SFA, torch::Tensor SFB) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kUInt8 && SFA.dtype() == torch::kUInt8,
              "pass raw e4m3 / ue8m0 bytes as uint8");
  A = A.contiguous(); B = B.contiguous();
  SFA = SFA.contiguous(); SFB = SFB.contiguous();
  int M = A.size(0), HD = A.size(1), N = B.size(0);
  TORCH_CHECK(M == kBlockM && N == kBlockN, "stage 1a is a single tile (M=N=128)");

  auto S = torch::empty({kBlockM, kBlockN}, torch::dtype(torch::kFloat32).device(A.device()));
  const uint8_t* a = A.data_ptr<uint8_t>();
  const uint8_t* b = B.data_ptr<uint8_t>();
  const uint8_t* sfa = SFA.data_ptr<uint8_t>();
  const uint8_t* sfb = SFB.data_ptr<uint8_t>();
  float* s = S.data_ptr<float>();

  switch (HD) {
    case 64:  qk_tile_kernel<64> <<<1, 256>>>(a, b, sfa, sfb, s); break;
    case 128: qk_tile_kernel<128><<<1, 256>>>(a, b, sfa, sfb, s); break;
    default: TORCH_CHECK(false, "head_dim must be 64/128, got ", HD);
  }
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "kernel failed: ", cudaGetErrorString(cudaGetLastError()));
  return S;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_tile", &qk_tile, "MXFP8 block-scaled QK, full 128x128 tile (stage 1a)");
}
