#pragma once
// S3B: MXFP8 attention BACKWARD (dkdv + dq kernels) for sm_120a.
// Algorithm validated in the CuTeDSL prototype (/tmp/opencode/mxattn/mxfp8_bwd_dsl.py):
//   dkdv kernel: block = (n_tile, h). accV/accK in registers, loop over m tiles:
//     S = QK^T (mxfp8), dP = dO V^T (mxfp8), P = exp2((S*sm - lse)*log2e),
//     dS = P*(dP - delta[row]); P,dS quantized per-32-along-q (contraction) with
//     cross-warp-pair amax via smem, written TRANSPOSED to smem;
//     dV += P^T dO_t, dK += dS^T Q_t (B operands = host-transposed dO_t/Q_t [d, S]).
//   dq kernel: block = (m_tile, h). accQ in registers, loop over n tiles:
//     S, dP recomputed; dS quantized per-32-along-kv (per-thread + quad_reduce,
//     fwd S5 pattern); dQ += dS K_t (B = host-transposed K_t [d, S]).
// v1 simplifications (correctness first): no TMA/pipeline/warp-specialization/
// persistence; plain cooperative gmem->smem copies + __syncthreads phases.
// SF gmem layout (host contract, matches DSL prototype _pack_sf):
//   flat 512B tiles, tile-linear index = (h*TM + mt)*TK + kt, byte offset within
//   tile = 16*(mn%32) + 4*(mn/32) + kb  (mn: row-in-tile, kb: 32-block-in-tile).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

namespace s3b {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlk = 128, SFVecSize = 32;
constexpr int kNThreads = 256;                    // 8 warps, single role
constexpr float kLog2e = 1.4426950408889634f;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TiledMmaB = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _128>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomT = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlk>>());
using SmemLayoutT = decltype(tile_to_shape(SmemLayoutAtomT{}, Shape<Int<kBlk>, Int<kBlk>>{}));

// Canonical block-scaled SF smem atom for a 128x128 operand tile (512B).
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaB::AtomShape_MNK{}) / SFVecSize;  // =1
using Blk_MN = typename BlkSF::Blk_MN; using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>; using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>; using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                               Int<kBlk>{} / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(Int<kBlk>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlk>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFT = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
static_assert(cosize_v<SmemLayoutSFT> == 512, "SF tile must be 512B");

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

// gmem -> smem data tile copy: 128 rows x 128B (8 x 16B vectors), 256 threads.
using GmemCopyTiled = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
    Layout<Shape<_32, _8>, Stride<_8, _1>>{},   // 32 rows x 8 vec-cols per step
    Layout<Shape<_1, _16>>{}));                  // 16 e4m3 = 16B per thread

struct ParamsBwd {
  // quantized operands (e4m3 bytes): Q,K,V,dO natural [H,S,128]; Qt,Kt,dOt [H,128,S]
  const uint8_t *Q, *K, *V, *D, *Qt, *Kt, *Dt;
  // SF flat-512B-tile blobs: natural [H*(S/128)*1, 512]; transposed [H*1*(S/128), 512]
  const uint8_t *sfQ, *sfK, *sfV, *sfD, *sfQt, *sfKt, *sfDt;
  const float* lse;    // [H, S] natural-log logsumexp
  const float* delta;  // [H, S] rowsum(dO*O)
  float *dQ, *dK, *dV; // [H, S, 128] fp32
  int S, H;
  float sm_scale;
  float* dbg = nullptr;
};

struct SharedStorageBwd {
  // 4 generic 128x128 e4m3 tile slots, aliased across phases:
  //   dkdv: {sA=Q, sB=K, sC=V, sD=dO} -> {sA=Dt, sB=Pt, sC=DSt, sD=Qt}
  //   dq:   {sA=Q, sD=dO resident} + {sB=K->DSt? see kernel} {sC=V->Kt}
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutT>> sA;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutT>> sB;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutT>> sC;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutT>> sD;
  alignas(128) uint8_t sSF[8][512];
  alignas(128) float sAmP[8][kBlk];   // per-warp col amax partials (dkdv, P)
  alignas(128) float sAmS[8][kBlk];   // (dkdv, dS)
};

__device__ __forceinline__ Element quant_e4m3(float v, int se) {
  return Element(v * exp2f(float(-se)));
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  // ceil(log2(amax)) - 8, clamped to ue8m0 [1, 254]; amax==0 -> byte 1 (2^-126 scale,
  // all-zero block: data bytes are 0 anyway).
  if (amax <= 0.f) return -126;
  int e = (int)ceilf(log2f(amax)) - 8;
  return max(-126, min(127, e));
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(uint32_t(-1), v, 1));
  v = fmaxf(v, __shfl_xor_sync(uint32_t(-1), v, 2));
  return v;
}

// cooperative 16KB gmem->smem tile copy. src: row-major [128, ld>=128] e4m3.
template <class GmemTensor>
__device__ __forceinline__ void load_tile(GmemTensor const& gTile,
                                          Element* smem_slot, int tid) {
  Tensor sT = make_tensor(make_smem_ptr(smem_slot), SmemLayoutT{});
  GmemCopyTiled gc; auto tgc = gc.get_thread_slice(tid);
  copy(gc, tgc.partition_S(gTile), tgc.partition_D(sT));
}

// cooperative 512B SF tile copy (flat).
__device__ __forceinline__ void load_sf(const uint8_t* src, uint8_t* dst, int tid) {
  if (tid < 128) reinterpret_cast<uint32_t*>(dst)[tid] = reinterpret_cast<const uint32_t*>(src)[tid];
}

// SF gmem tile base: tile (mt, kt) of head h. TM row-tiles, TK k-tiles per head.
__device__ __forceinline__ const uint8_t* sf_tile(const uint8_t* base, int h, int mt, int kt,
                                                  int TM, int TK) {
  return base + ((size_t(h) * TM + mt) * TK + kt) * 512;
}

// ===========================================================================
// dkdv kernel: grid = (S/128 n_tiles, H). accV/accK resident; loop m.
// ===========================================================================
__global__ void __launch_bounds__(kNThreads, 1)
dkdv_kernel(CUTE_GRID_CONSTANT ParamsBwd const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageBwd*>(smem_raw);
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int const n = blockIdx.x, h = blockIdx.y;
  int const MT = p.S / kBlk;

  TiledMmaB mma; auto thr_mma = mma.get_thread_slice(tid);
  auto ts = tile_shape(mma);

  Tensor sQA = make_tensor(make_smem_ptr(ss.sA.begin()), SmemLayoutT{});  // Q -> Dt
  Tensor sKB = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutT{});  // K -> Pt
  Tensor sVC = make_tensor(make_smem_ptr(ss.sC.begin()), SmemLayoutT{});  // V -> DSt
  Tensor sDD = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutT{});  // dO -> Qt

  // gmem views for this head
  auto gQ = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.Q) + size_t(h) * p.S * kHeadDim),
                        make_layout(make_shape(kBlk, kBlk), make_stride(kHeadDim, _1{})));
  auto gD = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.D) + size_t(h) * p.S * kHeadDim),
                        make_layout(make_shape(kBlk, kBlk), make_stride(kHeadDim, _1{})));
  auto gKt = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.Kt) + size_t(h) * kHeadDim * p.S),
                         make_layout(make_shape(kBlk, kBlk), make_stride(p.S, _1{})));   // [d, S] tile (0, kt)
  // K/V loaded per-m below (bisect: v1 slot arrangement)
  load_sf(sf_tile(p.sfK, h, n, 0, MT, 1), ss.sSF[2], tid);
  load_sf(sf_tile(p.sfV, h, n, 0, MT, 1), ss.sSF[3], tid);

  // operand fragments
  Tensor tSrA = thr_mma.partition_fragment_A(sQA);
  Tensor tSrB = thr_mma.partition_fragment_B(sKB);
  Tensor tSrSFA = mxfp8::partition_fragment_SFA(
      make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[0])), SmemLayoutSFT{}), thr_mma);
  Tensor tSrSFB = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[0])), SmemLayoutSFT{}), thr_mma);
  auto scA = make_tiled_copy_A(SmemCopyAtomData{}, mma); auto tscA = scA.get_thread_slice(tid);
  auto scB = make_tiled_copy_B(SmemCopyAtomData{}, mma); auto tscB = scB.get_thread_slice(tid);
  auto scSFA = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma), make_shape(size<0>(ts), size<2>(ts)));
  auto scSFB = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma), make_shape(size<1>(ts), size<2>(ts)));
  auto tscSFA = scSFA.get_thread_slice(tid); auto tscSFB = scSFB.get_thread_slice(tid);

  Tensor accV = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  Tensor accK = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  clear(accV); clear(accK);
  Tensor accS  = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  Tensor accDP = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  // (row, col) reduction views: ((_,b),_,n2) -> rc(mi=b, ni=a+2*n2)
  auto rc_view = [](auto& f) {
    return make_tensor(f.data(), make_layout(
        make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
        make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
  };
  Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
  Tensor accV_rc = rc_view(accV); Tensor accK_rc = rc_view(accK);
  constexpr int kNRow = 2, kNCol = kBlk / 4;   // 32 cols per thread
  int const row0 = warp * 16 + lane / 4;       // thread rows: row0, row0+8
  int const col0 = (lane % 4) * 2;             // col(ni) = (ni/2)*8 + col0 + (ni%2)
  int const grp = warp / 2;                    // 32-row scale group of this warp's rows

  auto sf_smem = [&](int i) {
    return make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[i])), SmemLayoutSFT{});
  };

  for (int m = 0; m < MT; ++m) {
    // ---- load Q[m], K[n], V[n], dO[m] + SF (natural) ----
    {
      Tensor gQm = make_tensor(gQ.data() + size_t(m) * kBlk * kHeadDim, gQ.layout());
      Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + size_t(h) * p.S * kHeadDim + size_t(n) * kBlk * kHeadDim), gQ.layout());
      Tensor gVn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + size_t(h) * p.S * kHeadDim + size_t(n) * kBlk * kHeadDim), gQ.layout());
      Tensor gDm = make_tensor(gD.data() + size_t(m) * kBlk * kHeadDim, gD.layout());
      load_tile(gQm, ss.sA.begin(), tid);
      load_tile(gKn, ss.sB.begin(), tid);
      load_tile(gVn, ss.sC.begin(), tid);
      load_tile(gDm, ss.sD.begin(), tid);
      load_sf(sf_tile(p.sfQ, h, m, 0, MT, 1), ss.sSF[0], tid);
      load_sf(sf_tile(p.sfD, h, m, 0, MT, 1), ss.sSF[1], tid);
    }
    __syncthreads();
    // ---- S = Q K^T ----
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sQA)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(0))), tscSFA.retile_D(tSrSFA));
    copy(scB, tscB.partition_S(as_position_independent_swizzle_tensor(sKB)), tscB.retile_D(tSrB));
    copy(scSFB, tscSFB.partition_S(as_position_independent_swizzle_tensor(sf_smem(2))), tscSFB.retile_D(tSrSFB));
    clear(accS);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accS);
    // ---- dP = dO V^T ----
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sDD)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(1))), tscSFA.retile_D(tSrSFA));
    copy(scB, tscB.partition_S(as_position_independent_swizzle_tensor(sVC)), tscB.retile_D(tSrB));
    copy(scSFB, tscSFB.partition_S(as_position_independent_swizzle_tensor(sf_smem(3))), tscSFB.retile_D(tSrSFB));
    clear(accDP);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accDP);
    __syncthreads();   // sK/sV dead after this point (fragments in regs)

    // ---- P, dS, per-col along-q amax ----
    float lse_r[kNRow], dlt_r[kNRow];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kNRow; ++mi) {
      int q = m * kBlk + row0 + mi * 8;
      lse_r[mi] = p.lse[size_t(h) * p.S + q];
      dlt_r[mi] = p.delta[size_t(h) * p.S + q];
    }
    float amaxP[kNCol], amaxS[kNCol];
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < kNCol; ++ni) {
      float ap = 0.f, as = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_r[mi]) * kLog2e);
        float dsv = pv * (accDP_rc(mi, ni) - dlt_r[mi]);
        accS_rc(mi, ni) = pv; accDP_rc(mi, ni) = dsv;
        ap = fmaxf(ap, fabsf(pv)); as = fmaxf(as, fabsf(dsv));
      }
      // reduce over lanes with same col (vary lane/4): masks 4,8,16
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 4));
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 8));
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 16));
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 4));
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 8));
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 16));
      amaxP[ni] = ap; amaxS[ni] = as;
      int c = (ni / 2) * 8 + col0 + (ni % 2);
      ss.sAmP[warp][c] = ap; ss.sAmS[warp][c] = as;
    }
    __syncthreads();
    // ---- quantize P^T / dS^T into sKB / sVC slots + SF bytes ----
    {
      Tensor sPt  = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutT{});  // [kv, q]
      Tensor sDSt = make_tensor(make_smem_ptr(ss.sC.begin()), SmemLayoutT{});  // [kv, q]
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < kNCol; ++ni) {
        int c = (ni / 2) * 8 + col0 + (ni % 2);
        float ap = fmaxf(ss.sAmP[2 * grp][c], ss.sAmP[2 * grp + 1][c]);
        float as = fmaxf(ss.sAmS[2 * grp][c], ss.sAmS[2 * grp + 1][c]);
        int sep = mx_scale_exp(ap), ses = mx_scale_exp(as);
        if (lane / 4 == 0) {   // single writer per (col, grp): warps 2g,2g+1 lanes/4==0 share grp
          int off = 16 * (c % 32) + 4 * (c / 32) + grp;
          ss.sSF[4][off] = uint8_t(sep + 127);
          ss.sSF[5][off] = uint8_t(ses + 127);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          int q = row0 + mi * 8;
          sPt(c, q)  = quant_e4m3(accS_rc(mi, ni), sep);
          sDSt(c, q) = quant_e4m3(accDP_rc(mi, ni), ses);
        }
      }
    }
    __syncthreads();   // Pt/DSt + SF visible
    // ---- dV += P^T dO ; dK += dS^T Q : A from smem (Pt=sA slot / DSt=sD slot),
    //      B gathered scalar from gmem natural dO/Q (transposed view, L2-hot) ----
    Tensor cB = thr_mma.partition_B(make_identity_tensor(make_shape(Int<kBlk>{}, Int<kBlk>{})));
    Tensor cSFB = mxfp8::partition_SFB(make_identity_tensor(make_shape(Int<kBlk>{}, Int<kBlk>{})), thr_mma);
    auto gather_B = [&](auto& frB, auto& frSFB, const uint8_t* gNat, const uint8_t* sfT) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(frB); ++i) {
        auto c = cB(i);
        frB(i) = gNat[(size_t(h) * kHeadDim + int(get<0>(c))) * p.S + size_t(m) * kBlk + int(get<1>(c))];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(frSFB); ++i) {
        auto c = cSFB(i);
        int d = int(get<0>(c)), q = int(get<1>(c));
        frSFB(i) = ElementSF::bitcast(sfT[(size_t(h) * MT + m) * 512 + 16 * (d % 32) + 4 * (d / 32) + (q % 128) / 32]);
      }
    };
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sKB)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(4))), tscSFA.retile_D(tSrSFA));
    gather_B(tSrB, tSrSFB, p.Dt, p.sfDt);
    if (p.dbg && m == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      Tensor sPt = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutT{});
      Tensor sDSt = make_tensor(make_smem_ptr(ss.sC.begin()), SmemLayoutT{});
      for (int i = tid; i < 128 * 128; i += kNThreads) {
        int kv = i / 128, q = i % 128;
        int sfP = ss.sSF[4][16 * (kv % 32) + 4 * (kv / 32) + q / 32];
        int sfS = ss.sSF[5][16 * (kv % 32) + 4 * (kv / 32) + q / 32];
        p.dbg[i]         = float(Element::bitcast(sPt(kv, q).storage)) * exp2f(float(sfP - 127));
        p.dbg[16384 + i] = float(Element::bitcast(sDSt(kv, q).storage)) * exp2f(float(sfS - 127));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accV);
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sVC)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(5))), tscSFA.retile_D(tSrSFA));
    gather_B(tSrB, tSrSFB, p.Qt, p.sfQt);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accK);
    __syncthreads();   // slots free for next m
  }

  // ---- epilogue: dV[h, n-tile], dK[h, n-tile] ----
  auto write_out = [&](auto const& acc, float* base) {
    Tensor gO = make_tensor(make_gmem_ptr(base + size_t(h) * p.S * kHeadDim + size_t(n) * kBlk * kHeadDim),
                            make_layout(make_shape(kBlk, kBlk), make_stride(kHeadDim, _1{})));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, acc, thr_mma.partition_C(gO));
  };
  write_out(accV, p.dV);
  write_out(accK, p.dK);
}

// ===========================================================================
// dq kernel: grid = (S/128 m_tiles, H). accQ resident; loop n.
//   slots: sA=Q (resident), sD=dO (resident), sB=K per-n -> DSt, sC=V per-n -> Kt
// ===========================================================================
__global__ void __launch_bounds__(kNThreads, 1)
dq_kernel(CUTE_GRID_CONSTANT ParamsBwd const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageBwd*>(smem_raw);
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int const m = blockIdx.x, h = blockIdx.y;
  int const MT = p.S / kBlk;

  TiledMmaB mma; auto thr_mma = mma.get_thread_slice(tid);
  auto ts = tile_shape(mma);

  Tensor sQA = make_tensor(make_smem_ptr(ss.sA.begin()), SmemLayoutT{});
  Tensor sKB = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutT{});
  Tensor sVC = make_tensor(make_smem_ptr(ss.sC.begin()), SmemLayoutT{});
  Tensor sDD = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutT{});

  auto nat = make_layout(make_shape(kBlk, kBlk), make_stride(kHeadDim, _1{}));
  auto trn = make_layout(make_shape(kBlk, kBlk), make_stride(p.S, _1{}));
  load_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.Q) + (size_t(h) * p.S + size_t(m) * kBlk) * kHeadDim), nat),
            ss.sA.begin(), tid);
  load_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.D) + (size_t(h) * p.S + size_t(m) * kBlk) * kHeadDim), nat),
            ss.sD.begin(), tid);
  load_sf(sf_tile(p.sfQ, h, m, 0, MT, 1), ss.sSF[0], tid);
  load_sf(sf_tile(p.sfD, h, m, 0, MT, 1), ss.sSF[1], tid);

  Tensor tSrA = thr_mma.partition_fragment_A(sQA);
  Tensor tSrB = thr_mma.partition_fragment_B(sKB);
  Tensor tSrSFA = mxfp8::partition_fragment_SFA(
      make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[0])), SmemLayoutSFT{}), thr_mma);
  Tensor tSrSFB = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[0])), SmemLayoutSFT{}), thr_mma);
  auto scA = make_tiled_copy_A(SmemCopyAtomData{}, mma); auto tscA = scA.get_thread_slice(tid);
  auto scB = make_tiled_copy_B(SmemCopyAtomData{}, mma); auto tscB = scB.get_thread_slice(tid);
  auto scSFA = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma), make_shape(size<0>(ts), size<2>(ts)));
  auto scSFB = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma), make_shape(size<1>(ts), size<2>(ts)));
  auto tscSFA = scSFA.get_thread_slice(tid); auto tscSFB = scSFB.get_thread_slice(tid);

  Tensor accQ  = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  clear(accQ);
  Tensor accS  = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  Tensor accDP = partition_fragment_C(mma, Shape<Int<kBlk>, Int<kBlk>>{});
  auto rc_view = [](auto& f) {
    return make_tensor(f.data(), make_layout(
        make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
        make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
  };
  Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
  constexpr int kNRow = 2, kNCol = kBlk / 4;
  int const row0 = warp * 16 + lane / 4;
  int const col0 = (lane % 4) * 2;

  float lse_r[kNRow], dlt_r[kNRow];
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < kNRow; ++mi) {
    int q = m * kBlk + row0 + mi * 8;
    lse_r[mi] = p.lse[size_t(h) * p.S + q];
    dlt_r[mi] = p.delta[size_t(h) * p.S + q];
  }

  auto sf_smem = [&](int i) {
    return make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSF[i])), SmemLayoutSFT{});
  };

  for (int n = 0; n < MT; ++n) {
    load_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlk) * kHeadDim), nat),
              ss.sB.begin(), tid);
    load_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + (size_t(h) * p.S + size_t(n) * kBlk) * kHeadDim), nat),
              ss.sC.begin(), tid);
    load_sf(sf_tile(p.sfK, h, n, 0, MT, 1), ss.sSF[2], tid);
    load_sf(sf_tile(p.sfV, h, n, 0, MT, 1), ss.sSF[3], tid);
    __syncthreads();
    // ---- S = Q K^T (Q resident in sQA) ----
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sQA)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(0))), tscSFA.retile_D(tSrSFA));
    copy(scB, tscB.partition_S(as_position_independent_swizzle_tensor(sKB)), tscB.retile_D(tSrB));
    copy(scSFB, tscSFB.partition_S(as_position_independent_swizzle_tensor(sf_smem(2))), tscSFB.retile_D(tSrSFB));
    clear(accS);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accS);
    // ---- dP = dO V^T (dO resident in sDD) ----
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sDD)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(1))), tscSFA.retile_D(tSrSFA));
    copy(scB, tscB.partition_S(as_position_independent_swizzle_tensor(sVC)), tscB.retile_D(tSrB));
    copy(scSFB, tscSFB.partition_S(as_position_independent_swizzle_tensor(sf_smem(3))), tscSFB.retile_D(tSrSFB));
    clear(accDP);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accDP);
    __syncthreads();   // sK/sV dead

    // ---- dS, quantize per-32-along-kv (per-thread + quad_reduce), write natural [q, kv] ----
    {
      Tensor sDS = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutT{});   // [q, kv]
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int q = row0 + mi * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < kBlk / SFVecSize; ++kb) {
          float as = 0.f;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_r[mi]) * kLog2e);
            float dsv = pv * (accDP_rc(mi, ni) - dlt_r[mi]);
            accDP_rc(mi, ni) = dsv;          // keep for byte write below
            as = fmaxf(as, fabsf(dsv));
          }
          as = quad_max(as);                  // over lanes l%4 (same rows, complementary cols)
          int ses = mx_scale_exp(as);
          if ((lane % 4) == 0)
            ss.sSF[4][16 * (q % 32) + 4 * (q / 32) + kb] = uint8_t(ses + 127);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            int c = (ni / 2) * 8 + col0 + (ni % 2);
            sDS(q, c) = quant_e4m3(accDP_rc(mi, ni), ses);
          }
        }
      }
    }
    // ---- load Kt[n] tile [d, kv] into sVC slot ----
    load_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.Kt) + size_t(h) * kHeadDim * p.S + size_t(n) * kBlk), trn),
              ss.sC.begin(), tid);
    load_sf(sf_tile(p.sfKt, h, 0, n, 1, MT), ss.sSF[5], tid);
    __syncthreads();
    // ---- dQ += dS Kt ----
    copy(scA, tscA.partition_S(as_position_independent_swizzle_tensor(sKB)), tscA.retile_D(tSrA));
    copy(scSFA, tscSFA.partition_S(as_position_independent_swizzle_tensor(sf_smem(4))), tscSFA.retile_D(tSrSFA));
    copy(scB, tscB.partition_S(as_position_independent_swizzle_tensor(sVC)), tscB.retile_D(tSrB));
    copy(scSFB, tscSFB.partition_S(as_position_independent_swizzle_tensor(sf_smem(5))), tscSFB.retile_D(tSrSFB));
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrA); ++k)
      cute::gemm(mma, make_zip_tensor(tSrA(_, _, k), tSrSFA(_, _, k)),
                 make_zip_tensor(tSrB(_, _, k), tSrSFB(_, _, k)), accQ);
    __syncthreads();
  }

  Tensor gO = make_tensor(make_gmem_ptr(p.dQ + (size_t(h) * p.S + size_t(m) * kBlk) * kHeadDim),
                          make_layout(make_shape(kBlk, kBlk), make_stride(kHeadDim, _1{})));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accQ, thr_mma.partition_C(gO));
}

}  // namespace s3b
