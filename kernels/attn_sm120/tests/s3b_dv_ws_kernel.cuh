#pragma once
// S3B-DV-WS: warp-specialized TMA-pipelined MXFP8 attention dV kernel (sm_120a).
// dv_ws = dk_ws minus the dP gemm/delta: block = (kv tile n of 128, head).
// Resident: K (+SF). Ring over q tiles of 64: Q (natural) + Dt (transposed dO).
// Per m: S'=KQ^T, P'=exp(S'*sm-lse) quantized per-32-along-q, dV += P' * Dt.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

namespace s3bdvws {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockN = 128, kBlockM = 64, SFVecSize = 32, kStages = 2;
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;      // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
// S'/dP': (M=kv 128, N=q 64, K=d 128). dK: (M=kv 128, N=d 128, K=q 64).
using TiledMmaK128 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _128>{}));
using TiledMmaK64 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _64>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomKV = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockM>>());
// ring tiles: Q/dO natural [q, d] 64x128 ; Qt transposed [d, q] 128x64
using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kHeadDim>, Int<kBlockM>, Int<kStages>>{}));
// dS' produced tile [kv, q] 128x64
using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kBlockM>>{}));

// canonical 128x128 SF atom (512B) — full-atom loads for Q/dO/Qt SF
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaK128::AtomShape_MNK{}) / SFVecSize;
using Blk_MN = typename BlkSF::Blk_MN; using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>; using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>; using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK128 = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{}, _1{}), kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(_1{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFT = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK128{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
static_assert(cosize_v<SmemLayoutSFT> == 512, "SF atom 512B");
// (128 mn, 64 k) half-atom view: byte off = 16*(mn%32) + 4*(mn/32) + kb, kb in {0,1}
using SmemLayoutSFK64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _4{}), make_shape(_32{}, _2{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

// gmem SF tiled layouts (head mode L): Q/dO natural (rows, d) SFA; Qt (d, rows) SFB.
using LayoutSFQ = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(0, int(kBlockM), int(kHeadDim), 0)));
using LayoutSFQt = decltype(BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockN), int(kHeadDim), 0, 0)));

using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(0, int(kHeadDim), 0), make_stride(int(kHeadDim), _1{}, int(kHeadDim))),
    SmemLayoutQ{}(_, _, _0{}), make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
using TMA_Qt = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kHeadDim), 0, 0), make_stride(0, _1{}, 0)),
    SmemLayoutQt{}(_, _, _0{}), make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), _1{}));
using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFQ{}),
    SmemLayoutSFT{}, make_shape(Int<128>{}, Int<kHeadDim>{}), _1{}));
using TMA_SFQt = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFQt{}),
    SmemLayoutSFT{}, make_shape(Int<kHeadDim>{}, Int<128>{}), _1{}));

using PipeQD  = cutlass::PipelineTmaAsync<kStages>;    // Q + dO + SFQ + SFD
using PipeQt  = cutlass::PipelineTmaAsync<kStages>;    // Qt + SFQt
using StateQD = cutlass::PipelineState<kStages>;
using StateQt = cutlass::PipelineState<kStages>;

constexpr int TmaBytesQD = kBlockM * kHeadDim + 512;             // Q data + SF atom
constexpr int TmaBytesQt = kHeadDim * kBlockM + 512;            // Dt + SFDt

struct ParamsDv {
  TMA_Q tma_q; TMA_Qt tma_dt; TMA_SFQ tma_sfq; TMA_SFQt tma_sfdt;
  LayoutSFQ layout_sfq;      // (S, 64, 128, H)
  LayoutSFQt layout_sfdt;    // (128, 128, S, H)
  const uint8_t *K;          // resident natural [H, S, 128]
  const uint8_t *sfK;        // flat 512B tiles
  const float* lse;          // [H, S] indexed by q
  float* dV;                 // [H, S, 128]
  int S, H;
  float sm_scale;
};

struct SharedStorageDv {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt>> sDt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutDS>> sDS;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFK;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFQ, sSFDt;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFDS;
  alignas(16) float sLse[64];   // current q tile
  alignas(8) typename PipeQD::SharedStorage pipeline_qd;
  alignas(8) typename PipeQt::SharedStorage pipeline_qt;
};

__device__ __forceinline__ Element quant_e4m3(float v, int se) {
  return Element(v * exp2f(float(-se)));
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (amax <= 0.f) return -126;
  int e = (int)ceilf(log2f(amax)) - 8;
  return max(-126, min(127, e));
}

__global__ void __launch_bounds__(kNThreads, 1)
dv_ws_kernel(CUTE_GRID_CONSTANT ParamsDv const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDv*>(smem_raw);
  int const n = blockIdx.x, h = blockIdx.y;
  int const NT128 = p.S / kBlockN;           // kv 128-tiles (resident SF atoms)
  int const MT = p.S / kBlockM;              // q 64-tiles streamed

  int const wg = cutlass::canonical_warp_group_idx();
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  typename PipeQD::Params pqd;
  pqd.role = (wg == 0) ? PipeQD::ThreadCategory::Producer : PipeQD::ThreadCategory::Consumer;
  pqd.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pqd.num_consumers = NumMmaThreads;
  pqd.transaction_bytes = TmaBytesQD;
  PipeQD pipeline_qd(ss.pipeline_qd, pqd, Shape<_1, _1, _1>{});
  typename PipeQt::Params pqt;
  pqt.role = pqd.role; pqt.is_leader = pqd.is_leader; pqt.num_consumers = NumMmaThreads;
  pqt.transaction_bytes = TmaBytesQt;
  PipeQt pipeline_qt(ss.pipeline_qt, pqt, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sDt = make_tensor(make_smem_ptr(ss.sDt.begin()), SmemLayoutQt{});
  Tensor sDS = make_tensor(make_smem_ptr(ss.sDS.begin()), SmemLayoutDS{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFT{});

  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mQ3d = p.tma_q.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mDt3d = p.tma_dt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
      Tensor mSFQ3d = p.tma_sfq.get_tma_tensor(shape(p.layout_sfq));
      Tensor mSFDt3d = p.tma_sfdt.get_tma_tensor(shape(p.layout_sfdt));
      auto bq = p.tma_q.get_slice(_0{}); auto bdt = p.tma_dt.get_slice(_0{});
      auto bsq = p.tma_sfq.get_slice(_0{}); auto bsdt = p.tma_sfdt.get_slice(_0{});
      Tensor mQ = mQ3d(_, _, h); Tensor mDt = mDt3d(_, _, h);
      Tensor mSFQ = mSFQ3d(_, _, h); Tensor mSFDt = mSFDt3d(_, _, h);
      Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gDt = local_tile(mDt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
      Tensor gSFQ = local_tile(mSFQ, make_shape(Int<128>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gSFDt = local_tile(mSFDt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
      Tensor tQgQ = group_modes<0, 3>(bq.partition_S(gQ));
      Tensor tQsQ = group_modes<0, 3>(bq.partition_D(sQ));
      Tensor tDtgDt = group_modes<0, 3>(bdt.partition_S(gDt));
      Tensor tDtsDt = group_modes<0, 3>(bdt.partition_D(sDt));
      Tensor tQgSFQ = group_modes<0, 3>(bsq.partition_S(gSFQ));
      Tensor tQsSFQ = group_modes<0, 3>(bsq.partition_D(
          make_tensor(make_smem_ptr(ss.sSFQ.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      Tensor tDtgSFDt = group_modes<0, 3>(bsdt.partition_S(gSFDt));
      Tensor tDtsSFDt = group_modes<0, 3>(bsdt.partition_D(
          make_tensor(make_smem_ptr(ss.sSFDt.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      auto sQD = cutlass::make_producer_start_state<PipeQD>();
      auto sDt_ = cutlass::make_producer_start_state<PipeQt>();
      for (int m = 0; m < MT; ++m) {
        int const sfatom = m / 2;
        pipeline_qd.producer_acquire(sQD);
        copy(p.tma_q.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgQ(_, m), tQsQ(_, sQD.index()));
        copy(p.tma_sfq.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgSFQ(_, sfatom), tQsSFQ(_, sQD.index()));
        ++sQD;
        pipeline_qt.producer_acquire(sDt_);
        copy(p.tma_dt.with(*pipeline_qt.producer_get_barrier(sDt_), 0), tDtgDt(_, m), tDtsDt(_, sDt_.index()));
        copy(p.tma_sfdt.with(*pipeline_qt.producer_get_barrier(sDt_), 0), tDtgSFDt(_, sfatom), tDtsSFDt(_, sDt_.index()));
        ++sDt_;
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;
    int const warp = tid / 32, lane = tid % 32;

    // resident K: cooperative vectorized load
    {
      auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                                Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
      auto tgc = gc.get_thread_slice(tid);
      auto nat = make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{}));
      Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      copy(gc, tgc.partition_S(gKn), tgc.partition_D(sK));
      for (int i = tid; i < 128; i += NumMmaThreads)
        reinterpret_cast<uint32_t*>(ss.sSFK.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfK + (size_t(h) * NT128 + n) * 512)[i];
    }
    cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier + 1).sync();

    TiledMmaK128 mma128; TiledMmaK64 mma64;
    auto thr128 = mma128.get_thread_slice(tid);
    auto thr64 = mma64.get_thread_slice(tid);
    auto ts128 = tile_shape(mma128);

    // S' operands: A = K resident, B = Q streamed
    Tensor tSrK  = thr128.partition_fragment_A(sK);
    Tensor tSrQ  = thr128.partition_fragment_B(sQ(_, _, _0{}));
    Tensor tSrSFK = mxfp8::partition_fragment_SFA(sSFK, thr128);
    Tensor tSrSFQ = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFT{}), thr128);
    // dV operands (K=q=64): A = P' [kv,q], B = Dt [d,q]
    Tensor tOrDS = thr64.partition_fragment_A(sDS);
    Tensor tOrDt = thr64.partition_fragment_B(sDt(_, _, _0{}));
    Tensor tOrSFDS = mxfp8::partition_fragment_SFA(
        make_tensor(make_smem_ptr(ss.sSFDS.begin()), SmemLayoutSFK64{}), thr64);
    Tensor tOrSFDt = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFDt.begin()), SmemLayoutSFK64{}), thr64);
    Tensor sfpA_coord = mxfp8::partition_SFA(
        make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);
    Tensor sfpB_coord = mxfp8::partition_SFB(
        make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);

    auto scA128 = make_tiled_copy_A(SmemCopyAtomData{}, mma128); auto tscA128 = scA128.get_thread_slice(tid);
    auto scB128 = make_tiled_copy_B(SmemCopyAtomData{}, mma128); auto tscB128 = scB128.get_thread_slice(tid);
    auto scSFA128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma128), make_shape(size<0>(ts128), size<2>(ts128)));
    auto scSFB128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma128), make_shape(size<1>(ts128), size<2>(ts128)));
    auto tscSFA128 = scSFA128.get_thread_slice(tid); auto tscSFB128 = scSFB128.get_thread_slice(tid);
    auto scA64 = make_tiled_copy_A(SmemCopyAtomData{}, mma64); auto tscA64 = scA64.get_thread_slice(tid);
    auto scB64 = make_tiled_copy_B(SmemCopyAtomData{}, mma64); auto tscB64 = scB64.get_thread_slice(tid);

    copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sK)), tscA128.retile_D(tSrK));
    copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFK)), tscSFA128.retile_D(tSrSFK));

    auto subSF = [](auto const& f, auto hc) {
      auto m1 = get<1>(f.layout()); auto a = get<0>(m1); auto b = get<1>(m1);
      auto nb = shape(b); auto sb = stride(b);
      auto t = make_tensor(f.data(), make_layout(get<0>(f.layout()),
          make_layout(make_shape(shape(a), make_shape(nb / _2{}, _2{})),
                      make_stride(stride(a), make_stride(sb, sb * (nb / _2{})))),
          get<2>(f.layout())))(_, make_coord(_, make_coord(_, hc)), _);
      return group_modes<1, 3>(t);
    };

    Tensor accV  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    clear(accV);
    Tensor accS  = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
    auto rc_view = [](auto& f) {
      return make_tensor(f.data(), make_layout(
          make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
          make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
    };
    Tensor accS_rc = rc_view(accS);
    constexpr int kNRow = 2, kNCol = kBlockM / 4;   // 16 cols (q) per thread
    int const row0 = warp * 16 + lane / 4;          // kv row
    int const col0 = (lane % 4) * 2;                // q col base

    StateQD rqd; StateQt rdt;
    auto step = [&](int m, auto hc) {
      // lse for this q tile: cooperative 64-float smem load, issued pre-wait
      if (tid < 64)
        ss.sLse[tid] = p.lse[size_t(h) * p.S + m * kBlockM + tid];
      // ---- S' = K Q^T ----
      { auto t = pipeline_qd.consumer_try_wait(rqd); pipeline_qd.consumer_wait(rqd, t);
        int stage = rqd.index();
        Tensor sSFQst = make_tensor(make_smem_ptr(ss.sSFQ.begin() + stage * 512), SmemLayoutSFT{});
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sQ(_, _, stage))), tscB128.retile_D(tSrQ));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFQst)), tscSFB128.retile_D(tSrSFQ));
      }
      auto tSrSFQ_h = subSF(tSrSFQ, hc);
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrK); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)),
                   make_zip_tensor(tSrQ(_, _, k), tSrSFQ_h(_, _, k)), accS);
      pipeline_qd.consumer_release(rqd); ++rqd;

      // ---- P' quantize per-32-along-q -> sDS [kv, q] + sSFDS ----
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // prev m's readers done; sLse visible
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int kv = row0 + mi * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < kBlockM / SFVecSize; ++kb) {
          float as = 0.f;
          float pv[8];
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            int c = (ni / 2) * 8 + col0 + (ni % 2);
            pv[j] = exp2f((accS_rc(mi, ni) * p.sm_scale - ss.sLse[c]) * kLog2e);
            as = fmaxf(as, fabsf(pv[j]));
          }
          as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 1));
          as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 2));
          int ses = mx_scale_exp(as);
          if ((lane % 4) == 0)
            ss.sSFDS.begin()[16 * (kv % 32) + 4 * (kv / 32) + kb] = ElementSF::bitcast(uint8_t(ses + 127));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            int c = (ni / 2) * 8 + col0 + (ni % 2);
            sDS(kv, c) = quant_e4m3(pv[j], ses);
          }
        }
      }
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // DS/SF visible

      // ---- dV += P' Dt ----
      { auto t = pipeline_qt.consumer_try_wait(rdt); pipeline_qt.consumer_wait(rdt, t);
        int stage = rdt.index();
        constexpr int hh = decltype(hc)::value;
        const uint8_t* sfDt_base = reinterpret_cast<const uint8_t*>(ss.sSFDt.begin()) + stage * 512 + 2 * hh;
        copy(scA64, tscA64.partition_S(as_position_independent_swizzle_tensor(sDS)), tscA64.retile_D(tOrDS));
        copy(scB64, tscB64.partition_S(as_position_independent_swizzle_tensor(sDt(_, _, stage))), tscB64.retile_D(tOrDt));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrDS); ++k) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDS(_, _, k)); ++i) {
            auto c = sfpA_coord(_, _, k)(i);
            int kv = int(get<0>(c)), q = int(get<1>(c));
            tOrSFDS(_, _, k)(i) = ElementSF::bitcast(ss.sSFDS.begin()[16 * (kv % 32) + 4 * (kv / 32) + q / 32].storage);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDt(_, _, k)); ++i) {
            auto c = sfpB_coord(_, _, k)(i);
            int d = int(get<0>(c)), q = int(get<1>(c));
            tOrSFDt(_, _, k)(i) = ElementSF::bitcast(sfDt_base[16 * (d % 32) + 4 * (d / 32) + q / 32]);
          }
        }
        pipeline_qt.consumer_release(rdt); ++rdt; }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrDS); ++k)
        cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, k), tOrSFDS(_, _, k)),
                   make_zip_tensor(tOrDt(_, _, k), tOrSFDt(_, _, k)), accV);
    };
    for (int m = 0; m < MT; m += 2) {
      step(m, cute::Int<0>{});
      if (m + 1 < MT) step(m + 1, cute::Int<1>{});
    }

    // ---- epilogue: dV[h, n-tile] ----
    Tensor gO = make_tensor(make_gmem_ptr(p.dV + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                            make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accV, thr64.partition_C(gO));
  }
}

}  // namespace s3bdvws
