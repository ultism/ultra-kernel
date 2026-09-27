#pragma once
// S3B-64: MXFP8 attention backward dV/dK kernels, small-tile occupancy variant.
// Split of dkdv into two kernels so smem stays <= ~50KB (2-3 blocks/SM hide
// latency without TMA pipelines). Tiles: kv=64 (block = (n64, head)), q=64
// stream. 128 threads, TiledMMA atom layout (4,1,1), PermM=64.
//   dv_kernel: resident K; per-m: Q, Dt(host-transposed dO); S=QK^T, P quant
//     along-q, dV += P^T Dt.
//   dk_kernel: resident K,V; per-m: Q, dO, Qt; S, dP=dO V^T, dS=P*(dP-delta)
//     quant along-q, dK += dS^T Qt.
// SF gmem: same flat-512B-tile contract as s3b (natural: tile (mt,kt)=
//   (h*TM+mt)*TK+kt; transposed likewise). 64-row tiles live in the LOWER/UPPER
//   half of the 128-row SF atom: byte half-offset +8*(m&1); 64-k side: +2*(n&1).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

namespace s3b64 {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBQ = 64, kBK = 64, SFVecSize = 32;
constexpr int kNThreads = 128;                    // 4 warps
constexpr float kLog2e = 1.4426950408889634f;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
// S/dP: (M=q 64, N=kv 64, K=d 128); dV/dK: (M=kv 64, N=d 128, K=q 64)
using TiledMmaK128 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _32, _128>{}));
using TiledMmaK64 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _32, _64>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtom128 = decltype(ccd::sm120_rr_smem_selector<Element, Int<128>>());
using SmemLayoutAtom64  = decltype(ccd::sm120_rr_smem_selector<Element, Int<64>>());
// Q/dO: [q, d] 64x128 ; K/V: [kv, d] 64x128 ; Qt/Dt: [d, q] 128x64 ; Pt/DSt: [kv, q] 64x64
using SmemLayoutQD = decltype(tile_to_shape(SmemLayoutAtom128{}, Shape<Int<kBQ>, Int<128>>{}));
using SmemLayoutKV = SmemLayoutQD;
using SmemLayoutDt = decltype(tile_to_shape(SmemLayoutAtom64{}, Shape<Int<128>, Int<kBQ>>{}));
using SmemLayoutPt = decltype(tile_to_shape(SmemLayoutAtom64{}, Shape<Int<kBK>, Int<kBQ>>{}));

// SF smem: (64 mn, 128 k) half-atom view (byte off 16*(mn%32)+4*(mn/32)+kb)
using SmemLayoutSF_M64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _2{}), make_shape(_32{}, _4{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));
// (128 mn, 64 k) view
using SmemLayoutSF_K64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _4{}), make_shape(_32{}, _2{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));
// (64 mn, 64 k) view (produced Pt/DSt SF)
using SmemLayoutSF_64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _2{}), make_shape(_32{}, _2{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

struct ParamsBwd64 {
  const uint8_t *Q, *K, *V, *D, *Qt, *Kt, *Dt;
  const uint8_t *sfQ, *sfK, *sfV, *sfD, *sfQt, *sfKt, *sfDt;
  const float* lse;
  const float* delta;
  float *dQ, *dK, *dV;
  int S, H;
  float sm_scale;
  float* dbg = nullptr;
};

struct SharedStorageDV {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;   // resident
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQD>> sQ;   // per-m
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutDt>> sDt;  // per-m
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutPt>> sPt;  // produced
  alignas(128) uint8_t sSFK[512], sSFQ[512], sSFDt[512], sSFPt[512];
  alignas(128) float sAm[4][kBK];
};
struct SharedStorageDK {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;   // resident
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sV;   // resident
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQD>> sQ;   // per-m
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQD>> sD;   // per-m
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutDt>> sQt;  // per-m
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutPt>> sDSt; // produced
  alignas(128) uint8_t sSFK[512], sSFV[512], sSFQ[512], sSFD[512], sSFQt[512], sSFDSt[512];
  alignas(128) float sAm[4][kBK];
};

__device__ __forceinline__ Element quant_e4m3(float v, int se) {
  return Element(v * exp2f(float(-se)));
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (amax <= 0.f) return -126;
  int e = (int)ceilf(log2f(amax)) - 8;
  return max(-126, min(127, e));
}

// cooperative gmem->smem tile copy, 128 thr, 16B vectors; gmem row-major [R, C].
template <class SmemT>
__device__ __forceinline__ void load_tile64(const uint8_t* src, int ldbytes, int R, int C,
                                            Element* smem_slot, int tid) {
  Tensor s = make_tensor(make_smem_ptr(smem_slot), SmemT{});
  constexpr int vec = 16;
  for (int i = tid; i < R * C / vec; i += kNThreads) {
    int r = (i * vec) / C, c = (i * vec) % C;
    *reinterpret_cast<cute::uint128_t*>(&s(r, c)) =
        *reinterpret_cast<const cute::uint128_t*>(src + (size_t)r * ldbytes + c);
  }
}
__device__ __forceinline__ void load_sf64(const uint8_t* src, uint8_t* dst, int tid) {
  if (tid < 128) reinterpret_cast<uint32_t*>(dst)[tid] = reinterpret_cast<const uint32_t*>(src)[tid];
}

// ---------------------------------------------------------------------------
// dV kernel: grid (S/64, H). resident K; stream Q, Dt.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kNThreads, 1)
dv64_kernel(CUTE_GRID_CONSTANT ParamsBwd64 const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDV*>(smem_raw);
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int const n = blockIdx.x, h = blockIdx.y;
  int const MT128 = p.S / 128;                 // SF atom rows are 128
  int const MT = p.S / kBQ;                    // q tiles (64)

  TiledMmaK128 mmaS; TiledMmaK64 mmaV;
  auto thrS = mmaS.get_thread_slice(tid);
  auto thrV = mmaV.get_thread_slice(tid);
  auto tsS = tile_shape(mmaS); auto tsV = tile_shape(mmaV);

  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQD{});
  Tensor sDt = make_tensor(make_smem_ptr(ss.sDt.begin()), SmemLayoutDt{});
  Tensor sPt = make_tensor(make_smem_ptr(ss.sPt.begin()), SmemLayoutPt{});

  // resident K (kv 64 rows of the n-block) + its SF (full 128-row atom, h64 = n&1)
  {
    int const kv0 = n * kBK;
    load_tile64<SmemLayoutKV>(p.K + (size_t(h) * p.S + kv0) * kHeadDim, kHeadDim, kBK, kHeadDim, ss.sK.begin(), tid);
    load_sf64(p.sfK + (size_t(h) * MT128 + n / 2) * 512, ss.sSFK, tid);
  }
  __syncthreads();

  // operand fragments
  Tensor tSrQ  = thrS.partition_fragment_A(sQ);
  Tensor tSrK  = thrS.partition_fragment_B(sK);
  Tensor tSrDt = thrV.partition_fragment_B(sDt);
  Tensor tSrPt = thrV.partition_fragment_A(sPt);
  Tensor sSFQ_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQ)), SmemLayoutSF_M64{});
  Tensor sSFK_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFK)), SmemLayoutSF_M64{});
  Tensor sSFDt_v = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFDt)), SmemLayoutSF_K64{});
  Tensor sSFPt_v = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFPt)), SmemLayoutSF_64{});
  Tensor tSrSFQ  = mxfp8::partition_fragment_SFA(sSFQ_v, thrS);
  Tensor tSrSFK  = mxfp8::partition_fragment_SFB(sSFK_v, thrS);
  Tensor tSrSFDt = mxfp8::partition_fragment_SFB(sSFDt_v, thrV);
  Tensor tSrSFPt = mxfp8::partition_fragment_SFA(sSFPt_v, thrV);
  auto scAS = make_tiled_copy_A(SmemCopyAtomData{}, mmaS); auto tscAS = scAS.get_thread_slice(tid);
  auto scBS = make_tiled_copy_B(SmemCopyAtomData{}, mmaS); auto tscBS = scBS.get_thread_slice(tid);
  auto scAV = make_tiled_copy_A(SmemCopyAtomData{}, mmaV); auto tscAV = scAV.get_thread_slice(tid);
  auto scBV = make_tiled_copy_B(SmemCopyAtomData{}, mmaV); auto tscBV = scBV.get_thread_slice(tid);
  auto scSFAS = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mmaS), make_shape(size<0>(tsS), size<2>(tsS)));
  auto scSFBS = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mmaS), make_shape(size<1>(tsS), size<2>(tsS)));
  auto scSFAV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mmaV), make_shape(size<0>(tsV), size<2>(tsV)));
  auto scSFBV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mmaV), make_shape(size<1>(tsV), size<2>(tsV)));
  auto tscSFAS = scSFAS.get_thread_slice(tid); auto tscSFBS = scSFBS.get_thread_slice(tid);
  auto tscSFAV = scSFAV.get_thread_slice(tid); auto tscSFBV = scSFBV.get_thread_slice(tid);

  // resident K fragments (once)
  copy(scBS, tscBS.partition_S(as_position_independent_swizzle_tensor(sK)), tscBS.retile_D(tSrK));

  Tensor accV = partition_fragment_C(mmaV, Shape<Int<kBK>, Int<kHeadDim>>{});
  clear(accV);
  Tensor accS = partition_fragment_C(mmaS, Shape<Int<kBQ>, Int<kBK>>{});
  auto rc_view = [](auto& f) {
    return make_tensor(f.data(), make_layout(
        make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
        make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
  };
  Tensor accS_rc = rc_view(accS);
  constexpr int kNRow = 2, kNCol = kBK / 4;      // 16
  int const row0 = warp * 16 + lane / 4;         // q rows in tile (warp < 4)
  int const col0 = (lane % 4) * 2;
  int const grp = warp / 2;                      // 32-row group (2 groups per 64-tile)

  for (int m = 0; m < MT; ++m) {
    int const mh = m & 1;                        // q-tile half inside 128-row SF atom
    load_tile64<SmemLayoutQD>(p.Q + (size_t(h) * p.S + m * kBQ) * kHeadDim, kHeadDim, kBQ, kHeadDim, ss.sQ.begin(), tid);
    load_tile64<SmemLayoutDt>(p.Dt + size_t(h) * kHeadDim * p.S + size_t(m) * kBQ, p.S, kHeadDim, kBQ, ss.sDt.begin(), tid);
    load_sf64(p.sfQ + (size_t(h) * MT128 + m / 2) * 512, ss.sSFQ, tid);
    load_sf64(p.sfDt + (size_t(h) * MT128 + m / 2) * 512, ss.sSFDt, tid);
    __syncthreads();
    // SF views at half-atom offsets (M-side +8*mh for Q; K-side +2*mh for Dt)
    Tensor sSFQ_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQ) + 8 * mh), SmemLayoutSF_M64{});
    Tensor sSFDt_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFDt) + 2 * mh), SmemLayoutSF_K64{});
    copy(scAS, tscAS.partition_S(as_position_independent_swizzle_tensor(sQ)), tscAS.retile_D(tSrQ));
    copy(scSFAS, tscSFAS.partition_S(as_position_independent_swizzle_tensor(sSFQ_h)), tscSFAS.retile_D(tSrSFQ));
    {
      Tensor sSFK_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFK) + 8 * (n & 1)), SmemLayoutSF_M64{});
      copy(scSFBS, tscSFBS.partition_S(as_position_independent_swizzle_tensor(sSFK_h)), tscSFBS.retile_D(tSrSFK));
    }
    clear(accS);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrQ); ++k)
      cute::gemm(mmaS, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                 make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accS);
    __syncthreads();   // sQ dead; Pt writes go nowhere near it but keep phases clean

    // P = exp2((S*sm - lse)*log2e); amax per (kv col, q32 group) across warp pair
    float lse_r[kNRow];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kNRow; ++mi) lse_r[mi] = p.lse[size_t(h) * p.S + m * kBQ + row0 + mi * 8];
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < kNCol; ++ni) {
      float ap = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_r[mi]) * kLog2e);
        accS_rc(mi, ni) = pv;
        ap = fmaxf(ap, fabsf(pv));
      }
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 4));
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 8));
      ap = fmaxf(ap, __shfl_xor_sync(uint32_t(-1), ap, 16));
      int c = (ni / 2) * 8 + col0 + (ni % 2);
      ss.sAm[warp][c] = ap;
    }
    __syncthreads();
    // quantize P^T into sPt [kv, q] + SF
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < kNCol; ++ni) {
      int c = (ni / 2) * 8 + col0 + (ni % 2);
      float ap = fmaxf(ss.sAm[2 * grp][c], ss.sAm[2 * grp + 1][c]);
      int sep = mx_scale_exp(ap);
      if (lane / 4 == 0)
        ss.sSFPt[16 * (c % 32) + 4 * (c / 32) + grp] = uint8_t(sep + 127);
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi)
        sPt(c, row0 + mi * 8) = quant_e4m3(accS_rc(mi, ni), sep);
    }
    __syncthreads();
    // dV += P^T Dt : A = sPt (M=kv, K=q), B = sDt (N=d, K=q)
    copy(scAV, tscAV.partition_S(as_position_independent_swizzle_tensor(sPt)), tscAV.retile_D(tSrPt));
    copy(scSFAV, tscSFAV.partition_S(as_position_independent_swizzle_tensor(sSFPt_v)), tscSFAV.retile_D(tSrSFPt));
    copy(scBV, tscBV.partition_S(as_position_independent_swizzle_tensor(sDt)), tscBV.retile_D(tSrDt));
    copy(scSFBV, tscSFBV.partition_S(as_position_independent_swizzle_tensor(sSFDt_h)), tscSFBV.retile_D(tSrSFDt));
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrPt); ++k)
      cute::gemm(mmaV, make_zip_tensor(tSrPt(_, _, k), tSrSFPt(_, _, k)),
                 make_zip_tensor(tSrDt(_, _, k), tSrSFDt(_, _, k)), accV);
    __syncthreads();
  }

  Tensor gO = make_tensor(make_gmem_ptr(p.dV + (size_t(h) * p.S + size_t(n) * kBK) * kHeadDim),
                          make_layout(make_shape(kBK, kHeadDim), make_stride(kHeadDim, _1{})));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accV, thrV.partition_C(gO));
}

// ---------------------------------------------------------------------------
// dK kernel: grid (S/64, H). resident K,V; stream Q, dO, Qt.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kNThreads, 1)
dk64_kernel(CUTE_GRID_CONSTANT ParamsBwd64 const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDK*>(smem_raw);
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int const n = blockIdx.x, h = blockIdx.y;
  int const MT128 = p.S / 128;
  int const MT = p.S / kBQ;

  TiledMmaK128 mmaS; TiledMmaK64 mmaK;
  auto thrS = mmaS.get_thread_slice(tid);
  auto thrK = mmaK.get_thread_slice(tid);
  auto tsS = tile_shape(mmaS); auto tsK = tile_shape(mmaK);

  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
  Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQD{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQD{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.sQt.begin()), SmemLayoutDt{});
  Tensor sDSt = make_tensor(make_smem_ptr(ss.sDSt.begin()), SmemLayoutPt{});

  {
    int const kv0 = n * kBK;
    load_tile64<SmemLayoutKV>(p.K + (size_t(h) * p.S + kv0) * kHeadDim, kHeadDim, kBK, kHeadDim, ss.sK.begin(), tid);
    load_tile64<SmemLayoutKV>(p.V + (size_t(h) * p.S + kv0) * kHeadDim, kHeadDim, kBK, kHeadDim, ss.sV.begin(), tid);
    load_sf64(p.sfK + (size_t(h) * MT128 + n / 2) * 512, ss.sSFK, tid);
    load_sf64(p.sfV + (size_t(h) * MT128 + n / 2) * 512, ss.sSFV, tid);
  }
  __syncthreads();

  Tensor tSrQ  = thrS.partition_fragment_A(sQ);
  Tensor tSrD  = thrS.partition_fragment_A(sD);
  Tensor tSrK  = thrS.partition_fragment_B(sK);
  Tensor tSrV  = thrS.partition_fragment_B(sV);
  Tensor tSrQt  = thrK.partition_fragment_B(sQt);
  Tensor tSrDSt = thrK.partition_fragment_A(sDSt);
  Tensor sSFQ_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQ)), SmemLayoutSF_M64{});
  Tensor sSFD_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFD)), SmemLayoutSF_M64{});
  Tensor sSFK_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFK)), SmemLayoutSF_M64{});
  Tensor sSFV_v  = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFV)), SmemLayoutSF_M64{});
  Tensor sSFQt_v = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQt)), SmemLayoutSF_K64{});
  Tensor sSFDSt_v = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFDSt)), SmemLayoutSF_64{});
  Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ_v, thrS);
  Tensor tSrSFD = mxfp8::partition_fragment_SFA(sSFD_v, thrS);
  Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK_v, thrS);
  Tensor tSrSFV = mxfp8::partition_fragment_SFB(sSFV_v, thrS);
  Tensor tSrSFQt = mxfp8::partition_fragment_SFB(sSFQt_v, thrK);
  Tensor tSrSFDSt = mxfp8::partition_fragment_SFA(sSFDSt_v, thrK);
  auto scAS = make_tiled_copy_A(SmemCopyAtomData{}, mmaS); auto tscAS = scAS.get_thread_slice(tid);
  auto scBS = make_tiled_copy_B(SmemCopyAtomData{}, mmaS); auto tscBS = scBS.get_thread_slice(tid);
  auto scAK = make_tiled_copy_A(SmemCopyAtomData{}, mmaK); auto tscAK = scAK.get_thread_slice(tid);
  auto scBK = make_tiled_copy_B(SmemCopyAtomData{}, mmaK); auto tscBK = scBK.get_thread_slice(tid);
  auto scSFAS = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mmaS), make_shape(size<0>(tsS), size<2>(tsS)));
  auto scSFBS = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mmaS), make_shape(size<1>(tsS), size<2>(tsS)));
  auto scSFAK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mmaK), make_shape(size<0>(tsK), size<2>(tsK)));
  auto scSFBK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mmaK), make_shape(size<1>(tsK), size<2>(tsK)));
  auto tscSFAS = scSFAS.get_thread_slice(tid); auto tscSFBS = scSFBS.get_thread_slice(tid);
  auto tscSFAK = scSFAK.get_thread_slice(tid); auto tscSFBK = scSFBK.get_thread_slice(tid);

  copy(scBS, tscBS.partition_S(as_position_independent_swizzle_tensor(sK)), tscBS.retile_D(tSrK));
  copy(scBS, tscBS.partition_S(as_position_independent_swizzle_tensor(sV)), tscBS.retile_D(tSrV));

  Tensor accK = partition_fragment_C(mmaK, Shape<Int<kBK>, Int<kHeadDim>>{});
  clear(accK);
  Tensor accS  = partition_fragment_C(mmaS, Shape<Int<kBQ>, Int<kBK>>{});
  Tensor accDP = partition_fragment_C(mmaS, Shape<Int<kBQ>, Int<kBK>>{});
  auto rc_view = [](auto& f) {
    return make_tensor(f.data(), make_layout(
        make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
        make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
  };
  Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
  constexpr int kNRow = 2, kNCol = kBK / 4;
  int const row0 = warp * 16 + lane / 4;
  int const col0 = (lane % 4) * 2;
  int const grp = warp / 2;

  for (int m = 0; m < MT; ++m) {
    int const mh = m & 1;
    load_tile64<SmemLayoutQD>(p.Q + (size_t(h) * p.S + m * kBQ) * kHeadDim, kHeadDim, kBQ, kHeadDim, ss.sQ.begin(), tid);
    load_tile64<SmemLayoutQD>(p.D + (size_t(h) * p.S + m * kBQ) * kHeadDim, kHeadDim, kBQ, kHeadDim, ss.sD.begin(), tid);
    load_tile64<SmemLayoutDt>(p.Qt + size_t(h) * kHeadDim * p.S + size_t(m) * kBQ, p.S, kHeadDim, kBQ, ss.sQt.begin(), tid);
    load_sf64(p.sfQ + (size_t(h) * MT128 + m / 2) * 512, ss.sSFQ, tid);
    load_sf64(p.sfD + (size_t(h) * MT128 + m / 2) * 512, ss.sSFD, tid);
    load_sf64(p.sfQt + (size_t(h) * MT128 + m / 2) * 512, ss.sSFQt, tid);
    __syncthreads();
    Tensor sSFQ_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQ) + 8 * mh), SmemLayoutSF_M64{});
    Tensor sSFD_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFD) + 8 * mh), SmemLayoutSF_M64{});
    Tensor sSFK_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFK) + 8 * (n & 1)), SmemLayoutSF_M64{});
    Tensor sSFV_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFV) + 8 * (n & 1)), SmemLayoutSF_M64{});
    Tensor sSFQt_h = make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(ss.sSFQt) + 2 * mh), SmemLayoutSF_K64{});
    copy(scAS, tscAS.partition_S(as_position_independent_swizzle_tensor(sQ)), tscAS.retile_D(tSrQ));
    copy(scSFAS, tscSFAS.partition_S(as_position_independent_swizzle_tensor(sSFQ_h)), tscSFAS.retile_D(tSrSFQ));
    copy(scSFBS, tscSFBS.partition_S(as_position_independent_swizzle_tensor(sSFK_h)), tscSFBS.retile_D(tSrSFK));
    clear(accS);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrQ); ++k)
      cute::gemm(mmaS, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                 make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accS);
    copy(scAS, tscAS.partition_S(as_position_independent_swizzle_tensor(sD)), tscAS.retile_D(tSrD));
    copy(scSFAS, tscSFAS.partition_S(as_position_independent_swizzle_tensor(sSFD_h)), tscSFAS.retile_D(tSrSFD));
    copy(scSFBS, tscSFBS.partition_S(as_position_independent_swizzle_tensor(sSFV_h)), tscSFBS.retile_D(tSrSFV));
    clear(accDP);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrD); ++k)
      cute::gemm(mmaS, make_zip_tensor(tSrD(_, _, k), tSrSFD(_, _, k)),
                 make_zip_tensor(tSrV(_, _, k), tSrSFV(_, _, k)), accDP);
    __syncthreads();

    // dS = P*(dP - delta); amax per (kv col, q32 group)
    float lse_r[kNRow], dlt_r[kNRow];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kNRow; ++mi) {
      lse_r[mi] = p.lse[size_t(h) * p.S + m * kBQ + row0 + mi * 8];
      dlt_r[mi] = p.delta[size_t(h) * p.S + m * kBQ + row0 + mi * 8];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < kNCol; ++ni) {
      float as = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_r[mi]) * kLog2e);
        float dsv = pv * (accDP_rc(mi, ni) - dlt_r[mi]);
        accDP_rc(mi, ni) = dsv;
        as = fmaxf(as, fabsf(dsv));
      }
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 4));
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 8));
      as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 16));
      int c = (ni / 2) * 8 + col0 + (ni % 2);
      ss.sAm[warp][c] = as;
    }
    __syncthreads();
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < kNCol; ++ni) {
      int c = (ni / 2) * 8 + col0 + (ni % 2);
      float as = fmaxf(ss.sAm[2 * grp][c], ss.sAm[2 * grp + 1][c]);
      int ses = mx_scale_exp(as);
      if (lane / 4 == 0)
        ss.sSFDSt[16 * (c % 32) + 4 * (c / 32) + grp] = uint8_t(ses + 127);
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi)
        sDSt(c, row0 + mi * 8) = quant_e4m3(accDP_rc(mi, ni), ses);
    }
    __syncthreads();
    // dK += dS^T Qt
    copy(scAK, tscAK.partition_S(as_position_independent_swizzle_tensor(sDSt)), tscAK.retile_D(tSrDSt));
    copy(scSFAK, tscSFAK.partition_S(as_position_independent_swizzle_tensor(sSFDSt_v)), tscSFAK.retile_D(tSrSFDSt));
    copy(scBK, tscBK.partition_S(as_position_independent_swizzle_tensor(sQt)), tscBK.retile_D(tSrQt));
    copy(scSFBK, tscSFBK.partition_S(as_position_independent_swizzle_tensor(sSFQt_h)), tscSFBK.retile_D(tSrSFQt));
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrDSt); ++k)
      cute::gemm(mmaK, make_zip_tensor(tSrDSt(_, _, k), tSrSFDSt(_, _, k)),
                 make_zip_tensor(tSrQt(_, _, k), tSrSFQt(_, _, k)), accK);
    __syncthreads();
  }

  Tensor gO = make_tensor(make_gmem_ptr(p.dK + (size_t(h) * p.S + size_t(n) * kBK) * kHeadDim),
                          make_layout(make_shape(kBK, kHeadDim), make_stride(kHeadDim, _1{})));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accK, thrK.partition_C(gO));
}

}  // namespace s3b64
