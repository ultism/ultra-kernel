// S0a: single-CTA TMA-fed block-scaled QK. The first real piece of the WS+TMA
// skeleton. Proves, beyond 1a (which loaded gmem->reg directly), that:
//   - TMA moves Q/K data tiles into swizzled smem,
//   - SF in the cutlass-canonical gmem layout TMA-loads into the canonical smem
//     SF atom, and our partition_SFA/SFB + get_layoutSFA_TV read it correctly,
//   - the warp atom then produces QK == fp64 dequant reference.
// No warp specialization / pipeline / scheduler yet (those are S0b/S0c). One CTA,
// 256 threads (= the 8-warp QK TiledMMA), thread 0 issues the TMA, all wait on a
// bare transaction mbarrier, then all consume.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s0_qk_tma.cu -o tests/s0_qk_tma

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cutlass/numeric_types.h>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"  // sm120_rr_smem_selector

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 128, SFVecSize = 32;
constexpr int NBLK = kHeadDim / SFVecSize;  // scales per row

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
// 8-warp QK TiledMMA with SageAttention's permutation tile (PermN=32 so the x4
// LDSM smem->reg copy on the B operand has enough vals; A-form for PV later).
using TiledMmaQK = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kHeadDim>>{}));

// --- data smem (swizzled, TMA + LDSM compatible) ---
namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<1, 2>(TileShape_MNK{})));

// --- SF smem atom (cutlass-canonical, replicated from sm120 blockscaled builder) ---
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVecSize;  // 1
using Blk_MN = typename BlkSF::Blk_MN;
using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape  = Shape<_32, _4>;
using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape   = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
using kBasicBlockStride  = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK   = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                                 Int<kHeadDim>{} / Int<SFVecSize>{} / Blk_SF{}),
                                      kBasicBlockShape{}));
using sSFA_shapeM  = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}),
                                      kBasicBlockStride{}));
using SmemLayoutAtomSFQ = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}),
                                               make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
using sSFB_shapeN  = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}),
                                      kBasicBlockStride{}));
using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}),
                                               make_stride(sSF_strideMN{}, sSFB_strideK{})));

using SmemCopyAtomQ  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

// canonical gmem SF layout type (filled on host, TMA source)
using LayoutSF = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim))));

using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                make_shape(int(kBlockM), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));
using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                make_shape(int(kBlockN), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutK{}, select<1, 2>(TileShape_MNK{}), _1{}));
using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutAtomSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
using TMA_SFK = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutAtomSFK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{}));

struct Params {
  TMA_Q tma_q; TMA_K tma_k; TMA_SFQ tma_sfq; TMA_SFK tma_sfk;
  float* out_S;
};

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element,   cute::cosize_v<SmemLayoutQ>> sQ;
  alignas(1024) cute::ArrayEngine<Element,   cute::cosize_v<SmemLayoutK>> sK;
  alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutAtomSFQ>> sSFQ;
  alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutAtomSFK>> sSFK;
  alignas(8) uint64_t mbar;
};

__global__ void __launch_bounds__(256, 1) s0_qk_kernel(CUTE_GRID_CONSTANT Params const params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  int tid = threadIdx.x;

  Tensor sQ   = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sK   = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutK{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutAtomSFQ{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutAtomSFK{});

  // ---- TMA load (thread 0), bare transaction mbarrier ----
  constexpr uint32_t tma_bytes =
      cute::cosize_v<SmemLayoutQ> * sizeof(Element) +
      cute::cosize_v<SmemLayoutK> * sizeof(Element) +
      cute::cosize_v<SmemLayoutAtomSFQ> * sizeof(ElementSF) +
      cute::cosize_v<SmemLayoutAtomSFK> * sizeof(ElementSF);

  if (tid == 0) {
    cute::initialize_barrier(ss.mbar, /*arrive_count=*/1);
  }
  __syncthreads();

  if (tid == 0) {
    Tensor mQ = params.tma_q.get_tma_tensor(make_shape(int(kBlockM), int(kHeadDim)));
    Tensor mK = params.tma_k.get_tma_tensor(make_shape(int(kBlockN), int(kHeadDim)));
    Tensor mSFQ = params.tma_sfq.get_tma_tensor(shape(LayoutSF{}));
    Tensor mSFK = params.tma_sfk.get_tma_tensor(shape(LayoutSF{}));
    Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
    Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
    Tensor gSFQ = local_tile(mSFQ, select<0, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
    Tensor gSFK = local_tile(mSFK, select<1, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
    auto bq  = params.tma_q.get_slice(_0{});
    auto bk  = params.tma_k.get_slice(_0{});
    auto bsq = params.tma_sfq.get_slice(_0{});
    auto bsk = params.tma_sfk.get_slice(_0{});
    cute::set_barrier_transaction_bytes(ss.mbar, tma_bytes);
    copy(params.tma_q.with(ss.mbar),   bq.partition_S(gQ),   bq.partition_D(sQ));
    copy(params.tma_k.with(ss.mbar),   bk.partition_S(gK),   bk.partition_D(sK));
    copy(params.tma_sfq.with(ss.mbar), bsq.partition_S(gSFQ), bsq.partition_D(sSFQ));
    copy(params.tma_sfk.with(ss.mbar), bsk.partition_S(gSFK), bsk.partition_D(sSFK));
  }
  cute::wait_barrier(ss.mbar, /*phase=*/0);

  // ---- consumer: smem -> reg, QK ----
  TiledMmaQK tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_mma);
  Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK, thr_mma);

  auto sc_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma);
  auto thr_sc_Q = sc_Q.get_thread_slice(tid);
  copy(sc_Q, thr_sc_Q.partition_S(as_position_independent_swizzle_tensor(sQ)),
       thr_sc_Q.retile_D(tSrQ));
  auto sc_K = make_tiled_copy_B(SmemCopyAtomQ{}, tiled_mma);
  auto thr_sc_K = sc_K.get_thread_slice(tid);
  copy(sc_K, thr_sc_K.partition_S(as_position_independent_swizzle_tensor(sK)),
       thr_sc_K.retile_D(tSrK));

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto sc_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(tiled_mma),
                                     make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_sc_SFQ = sc_SFQ.get_thread_slice(tid);
  copy(sc_SFQ, thr_sc_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ)),
       thr_sc_SFQ.retile_D(tSrSFQ));
  auto sc_SFK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(tiled_mma),
                                     make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_sc_SFK = sc_SFK.get_thread_slice(tid);
  copy(sc_SFK, thr_sc_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK)),
       thr_sc_SFK.retile_D(tSrSFK));

  Tensor gS = make_tensor(make_gmem_ptr(params.out_S),
                          make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}), LayoutRight{}));
  Tensor tSgS = thr_mma.partition_C(gS);
  Tensor accum = thr_mma.partition_fragment_C(gS);
  clear(accum);

  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(tSrQ); ++k) {
    cute::gemm(tiled_mma,
               make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
               make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accum);
  }
  copy(accum, tSgS);
}

// ---------- host: exact-representable data, canonical SF fill, fp64 reference ----------
static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int exp2) { return uint8_t(exp2 + 127); }  // 2^exp2

int main() {
  printf("S0a: single-CTA TMA-fed block-scaled QK (kBlockM=kBlockN=128, hd=128)\n");
  const int M = kBlockM, N = kBlockN, HD = kHeadDim;

  // Non-uniform but exactly-representable: data in {-2,-1,-0.5,0.5,1,2} (exact
  // e4m3), per-(row/col,block) scales distinct powers of two (exact ue8m0).
  std::vector<float> dataVals = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  std::vector<uint8_t> hQ(M * HD), hK(N * HD);
  std::vector<int> qexp(M * NBLK), kexp(N * NBLK);
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < NBLK; ++b) qexp[m * NBLK + b] = ((m + b) % 4) - 1;  // {.5,1,2,4}
  for (int n = 0; n < N; ++n)
    for (int b = 0; b < NBLK; ++b) kexp[n * NBLK + b] = (n + 2 * b) % 3;     // {1,2,4}
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < HD; ++k) hQ[m * HD + k] = e4m3_byte(dataVals[(m * HD + k) % dataVals.size()]);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < HD; ++k) hK[n * HD + k] = e4m3_byte(dataVals[(n * 7 + k * 3) % dataVals.size()]);

  // canonical SF buffers
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(M, N, HD));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(N, N, HD));
  size_t sfq_cosize = cosize(layoutSFQ), sfk_cosize = cosize(layoutSFK);
  std::vector<uint8_t> hSFQ(sfq_cosize, 0), hSFK(sfk_cosize, 0);
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < NBLK; ++b)
      hSFQ[layoutSFQ(make_coord(m, b * SFVecSize))] = ue8m0_byte_pow2(qexp[m * NBLK + b]);
  for (int n = 0; n < N; ++n)
    for (int b = 0; b < NBLK; ++b)
      hSFK[layoutSFK(make_coord(n, b * SFVecSize))] = ue8m0_byte_pow2(kexp[n * NBLK + b]);

  // reference S = sum_k dequant(Q)*dequant(K), fp64
  std::vector<double> ref(M * N, 0.0);
  auto deq = [&](uint8_t db, int e) { return double(float(reinterpret_cast<cutlass::float_e4m3_t&>(db))) * std::ldexp(1.0, e); };
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int k = 0; k < HD; ++k)
        s += deq(hQ[m * HD + k], qexp[m * NBLK + k / SFVecSize]) *
             deq(hK[n * HD + k], kexp[n * NBLK + k / SFVecSize]);
      ref[m * N + n] = s;
    }

  // device
  Element *dQ, *dK; ElementSF *dSFQ, *dSFK; float* dS;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size()));
  CK(cudaMalloc(&dSFQ, hSFQ.size())); CK(cudaMalloc(&dSFK, hSFK.size()));
  CK(cudaMalloc(&dS, M * N * sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));

  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(M, HD), make_stride(HD, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(N, HD), make_stride(HD, _1{}));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Params params;
  params.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  params.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}, select<1, 2>(TileShape_MNK{}), _1{});
  params.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutAtomSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{});
  params.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutAtomSFK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});
  params.out_S = dS;

  int smem_bytes = int(sizeof(SharedStorage));
  CK(cudaFuncSetAttribute(s0_qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  printf("smem = %d bytes\n", smem_bytes);
  s0_qk_kernel<<<1, 256, smem_bytes>>>(params);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float> hS(M * N);
  CK(cudaMemcpy(hS.data(), dS, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  int bad = 0; double maxrel = 0; int fm = -1, fn = -1;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double r = ref[m * N + n], g = hS[m * N + n];
      double rel = std::abs(g - r) / std::max(1e-6, std::abs(r));
      if (rel > maxrel) { maxrel = rel; }
      if (rel > 1e-3) { if (bad == 0) { fm = m; fn = n; } ++bad; }
    }
  printf("max|rel| = %.3g, bad(>1e-3) = %d / %d\n", maxrel, bad, M * N);
  if (bad) printf("  first bad [%d,%d]: got=%.4f exp=%.4f\n", fm, fn, hS[fm * N + fn], ref[fm * N + fn]);
  printf(bad == 0 ? "S0a PASS\n" : "S0a FAIL\n");
  return bad == 0 ? 0 : 1;
}
