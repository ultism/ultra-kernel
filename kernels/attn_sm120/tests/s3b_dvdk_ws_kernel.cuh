#pragma once
// S3B-DVDK-WS: fused warp-specialized TMA-pipelined MXFP8 attention dV+dK kernel
// (sm_120a). kv-stationary: block = (kv tile n of 64, head). Resident K, V.
// Ring over q tiles of 64: Q, dO (natural) + Qt, Dt (transposed) + SF.
// Per m: S'=KQ^T and dP'=V dO^T computed ONCE (shared), then P' and
// dS'=P'*(dP'-delta) quantized per-32-along-q in ONE fused pass, dV += P' Dt,
// dK += dS' Qt. 4 MMA gemms/iter vs 5 for separate dv_ws+dk_ws.
// 256 consumer threads: atom layout (4,2,1); C frag: row0=(warp%4)*16+lane/4,
// col=(warp/4)*8+(lane%4)*2+parity+16*rep; 32-q quant groups span warp pairs
// (w, w+4) -> sAm cross-warp combine (dk64 pattern).
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

namespace s3bdvdk {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockN = 64, kBlockM = 64, SFVecSize = 32, kStages = 2;
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;      // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;
constexpr int kAmaxBarrier = 2;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
// S'/dP': (M=kv 64, N=q 64, K=d 128). dV/dK: (M=kv 64, N=d 128, K=q 64).
using TiledMmaK128 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_4, _2, _1>>{}, Tile<_64, _32, _128>{}));
using TiledMmaK64 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_4, _2, _1>>{}, Tile<_64, _32, _64>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtom128 = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutAtom64  = decltype(ccd::sm120_rr_smem_selector<Element, Int<64>>());
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom128{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
// ring: Q/dO natural [q, d] 64x128 x2st ; Qt/Dt transposed [d, q] 128x64 x2st
using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom128{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtom64{}, Shape<Int<kHeadDim>, Int<kBlockM>, Int<kStages>>{}));
// produced P'/dS' tiles [kv, q] 64x64
using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtom64{}, Shape<Int<kBlockN>, Int<kBlockM>>{}));

// canonical 128x128 SF atom (512B) — full-atom loads for streamed Q/dO/Qt/Dt SF
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
// (128 mn, 64 k) half-atom view for the dV/dK B-side SF gathers
using SmemLayoutSFK64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _4{}), make_shape(_32{}, _2{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));
// (64 mn, 64 k) produced-SF view
using SmemLayoutSF64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _2{}), make_shape(_32{}, _2{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));
// (64 mn, 128 k) resident K/V SF half-atom view (byte off +8*(n&1))
using SmemLayoutSFA64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _2{}), make_shape(_32{}, _4{})),
    make_stride(make_stride(_0{}, _16{}, _4{}), make_stride(_0{}, _1{}))));

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

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
using PipeTt  = cutlass::PipelineTmaAsync<kStages>;    // Qt + Dt + SFQt + SFDt
using StateQD = cutlass::PipelineState<kStages>;
using StateTt = cutlass::PipelineState<kStages>;

constexpr int TmaBytesQD = kBlockM * kHeadDim * 2 + 512 * 2;
constexpr int TmaBytesTt = kHeadDim * kBlockM * 2 + 512 * 2;

struct ParamsDvDK {
  TMA_Q tma_q; TMA_Q tma_d; TMA_Qt tma_qt; TMA_Qt tma_dt;
  TMA_SFQ tma_sfq; TMA_SFQ tma_sfd; TMA_SFQt tma_sfqt; TMA_SFQt tma_sfdt;
  LayoutSFQ layout_sfq;
  LayoutSFQt layout_sfqt;
  const uint8_t *K, *V;      // resident natural [H, S, 128]
  const uint8_t *sfK, *sfV;  // flat 512B tiles
  const float* lse;          // [H, S] indexed by q
  const float* delta;        // [H, S] indexed by q
  float* dV;                 // [H, S, 128]
  float* dK;                 // [H, S, 128]
  int S, H;
  float sm_scale;
  uint8_t* dbg = nullptr;  // [2][2*64*64 + 1024] P/DS tiles + SF
};

struct SharedStorageDvDK {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sD;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt>> sQt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt>> sDt;

  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFK, sSFV;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFQ, sSFD, sSFQt, sSFDt;
  alignas(16) float sLse[64];
  alignas(16) float sDlt[64];
  alignas(16) float sAmP[8][32];   // [warp][(mi*8 + lane/4)*2 + kb]
  alignas(16) float sAmD[8][32];
  alignas(8) typename PipeQD::SharedStorage pipeline_qd;
  alignas(8) typename PipeTt::SharedStorage pipeline_tt;
};

__device__ __forceinline__ Element quant_e4m3(float v, int se) {
  return Element(v * exp2f(float(-se)));
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (amax <= 0.f) return -126;
  int e = (int)ceilf(log2f(amax)) - 8;
  return max(-126, min(127, e));
}

__global__ void __launch_bounds__(kNThreads)
dvdk_ws_kernel(CUTE_GRID_CONSTANT ParamsDvDK const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDvDK*>(smem_raw);
  int const n = blockIdx.x, h = blockIdx.y;
  int const NT128 = p.S / 128;               // resident SF atoms are 128-row
  int const MT = p.S / kBlockM;

  int const wg = cutlass::canonical_warp_group_idx();
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  typename PipeQD::Params pqd;
  pqd.role = (wg == 0) ? PipeQD::ThreadCategory::Producer : PipeQD::ThreadCategory::Consumer;
  pqd.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pqd.num_consumers = NumMmaThreads;
  pqd.transaction_bytes = TmaBytesQD;
  PipeQD pipeline_qd(ss.pipeline_qd, pqd, Shape<_1, _1, _1>{});
  typename PipeTt::Params ptt;
  ptt.role = pqd.role; ptt.is_leader = pqd.is_leader; ptt.num_consumers = NumMmaThreads;
  ptt.transaction_bytes = TmaBytesTt;
  PipeTt pipeline_tt(ss.pipeline_tt, ptt, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
  Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQ{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.sQt.begin()), SmemLayoutQt{});
  Tensor sDt = make_tensor(make_smem_ptr(ss.sDt.begin()), SmemLayoutQt{});


  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mQ3d = p.tma_q.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mD3d = p.tma_d.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mQt3d = p.tma_qt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
      Tensor mDt3d = p.tma_dt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
      Tensor mSFQ3d = p.tma_sfq.get_tma_tensor(shape(p.layout_sfq));
      Tensor mSFQt3d = p.tma_sfqt.get_tma_tensor(shape(p.layout_sfqt));
      auto bq = p.tma_q.get_slice(_0{}); auto bqt = p.tma_qt.get_slice(_0{});
      auto bsq = p.tma_sfq.get_slice(_0{}); auto bsqt = p.tma_sfqt.get_slice(_0{});
      Tensor mQ = mQ3d(_, _, h); Tensor mD = mD3d(_, _, h);
      Tensor mQt = mQt3d(_, _, h); Tensor mDt = mDt3d(_, _, h);
      Tensor mSFQ = mSFQ3d(_, _, h); Tensor mSFQt = mSFQt3d(_, _, h);
      Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gD = local_tile(mD, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
      Tensor gDt = local_tile(mDt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
      Tensor gSFQ = local_tile(mSFQ, make_shape(Int<128>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gSFQt = local_tile(mSFQt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
      Tensor tQgQ = group_modes<0, 3>(bq.partition_S(gQ));
      Tensor tQsQ = group_modes<0, 3>(bq.partition_D(sQ));
      Tensor tQgD = group_modes<0, 3>(bq.partition_S(gD));
      Tensor tQsD = group_modes<0, 3>(bq.partition_D(sD));
      Tensor tQtgQt = group_modes<0, 3>(bqt.partition_S(gQt));
      Tensor tQtsQt = group_modes<0, 3>(bqt.partition_D(sQt));
      Tensor tDtgDt = group_modes<0, 3>(bqt.partition_S(gDt));
      Tensor tDtsDt = group_modes<0, 3>(bqt.partition_D(sDt));
      Tensor tQgSFQ = group_modes<0, 3>(bsq.partition_S(gSFQ));
      auto sfStage = [](ElementSF* base) {
        return make_tensor(make_smem_ptr(base),
                           make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                       append(stride(SmemLayoutSFT{}), Int<512>{})));
      };
      Tensor tQsSFQ  = group_modes<0, 3>(bsq.partition_D(sfStage(ss.sSFQ.begin())));
      Tensor tQsSFD  = group_modes<0, 3>(bsq.partition_D(sfStage(ss.sSFD.begin())));
      Tensor tQtgSFQt = group_modes<0, 3>(bsqt.partition_S(gSFQt));
      Tensor tQtsSFQt = group_modes<0, 3>(bsqt.partition_D(sfStage(ss.sSFQt.begin())));
      Tensor tQtsSFDt = group_modes<0, 3>(bsqt.partition_D(sfStage(ss.sSFDt.begin())));
      auto sQD = cutlass::make_producer_start_state<PipeQD>();
      auto sTt = cutlass::make_producer_start_state<PipeTt>();
      for (int m = 0; m < MT; ++m) {
        int const sfatom = m / 2;
        pipeline_qd.producer_acquire(sQD);
        copy(p.tma_q.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgQ(_, m), tQsQ(_, sQD.index()));
        copy(p.tma_d.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgD(_, m), tQsD(_, sQD.index()));
        copy(p.tma_sfq.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgSFQ(_, sfatom), tQsSFQ(_, sQD.index()));
        copy(p.tma_sfd.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgSFQ(_, sfatom), tQsSFD(_, sQD.index()));
        ++sQD;
        pipeline_tt.producer_acquire(sTt);
        copy(p.tma_qt.with(*pipeline_tt.producer_get_barrier(sTt), 0), tQtgQt(_, m), tQtsQt(_, sTt.index()));
        copy(p.tma_dt.with(*pipeline_tt.producer_get_barrier(sTt), 0), tDtgDt(_, m), tDtsDt(_, sTt.index()));
        copy(p.tma_sfqt.with(*pipeline_tt.producer_get_barrier(sTt), 0), tQtgSFQt(_, sfatom), tQtsSFQt(_, sTt.index()));
        copy(p.tma_sfdt.with(*pipeline_tt.producer_get_barrier(sTt), 0), tQtgSFQt(_, sfatom), tQtsSFDt(_, sTt.index()));
        ++sTt;
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<240>();
    int const tid = threadIdx.x - NumCopyThreads;
    int const warp = tid / 32, lane = tid % 32;

    // resident K, V (+SF): cooperative vectorized load
    {
      auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                                Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
      auto tgc = gc.get_thread_slice(tid);
      auto nat = make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{}));
      Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      Tensor gVn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      copy(gc, tgc.partition_S(gKn), tgc.partition_D(sK));
      copy(gc, tgc.partition_S(gVn), tgc.partition_D(sV));
      // resident SF: 64-row half of the 128-row atom (byte half +8*(n&1))
      const uint8_t* srcK = p.sfK + (size_t(h) * NT128 + n / 2) * 512;
      const uint8_t* srcV = p.sfV + (size_t(h) * NT128 + n / 2) * 512;
      for (int i = tid; i < 128; i += NumMmaThreads) {
        reinterpret_cast<uint32_t*>(ss.sSFK.begin())[i] = reinterpret_cast<const uint32_t*>(srcK)[i];
        reinterpret_cast<uint32_t*>(ss.sSFV.begin())[i] = reinterpret_cast<const uint32_t*>(srcV)[i];
      }
    }
    cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier + 1).sync();

    TiledMmaK128 mma128; TiledMmaK64 mma64;
    auto thr128 = mma128.get_thread_slice(tid);
    auto thr64 = mma64.get_thread_slice(tid);
    auto ts128 = tile_shape(mma128); auto ts64 = tile_shape(mma64);

    // S'/dP' fragments: A = K/V resident, B = Q/dO streamed
    Tensor tSrK  = thr128.partition_fragment_A(sK);
    Tensor tSrV  = thr128.partition_fragment_A(sV);
    Tensor tSrQ  = thr128.partition_fragment_B(sQ(_, _, _0{}));
    Tensor tSrD  = thr128.partition_fragment_B(sD(_, _, _0{}));
    // resident SF views: 64-row half-atom at byte +8*(n&1), layout (64 mn, 128 k)
    Tensor sSFK_v = make_tensor(make_smem_ptr(ss.sSFK.begin() + 8 * (n & 1)), SmemLayoutSFA64{});
    Tensor sSFV_v = make_tensor(make_smem_ptr(ss.sSFV.begin() + 8 * (n & 1)), SmemLayoutSFA64{});
    Tensor tSrSFK = mxfp8::partition_fragment_SFA(sSFK_v, thr128);
    Tensor tSrSFV = mxfp8::partition_fragment_SFA(sSFV_v, thr128);
    Tensor tSrSFQ = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFT{}), thr128);
    Tensor tSrSFD = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFD.begin()), SmemLayoutSFT{}), thr128);
    // dV/dK fragments (K=q=64): A = sP/sDS [kv,q], B = Dt/Qt [d,q]
    Tensor sProto = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutDS{});
    Tensor tOrP  = thr64.partition_fragment_A(sProto);
    Tensor tOrDS = thr64.partition_fragment_A(sProto);
    Tensor tOrDt = thr64.partition_fragment_B(sDt(_, _, _0{}));
    Tensor tOrQt = thr64.partition_fragment_B(sQt(_, _, _0{}));
    Tensor tOrSFP = mxfp8::partition_fragment_SFA(
        make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSF64{}), thr64);
    Tensor tOrSFDS = mxfp8::partition_fragment_SFA(
        make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSF64{}), thr64);
    Tensor tOrSFDt = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFDt.begin()), SmemLayoutSFK64{}), thr64);
    Tensor tOrSFQt = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFQt.begin()), SmemLayoutSFK64{}), thr64);
    Tensor sfpA_coord = mxfp8::partition_SFA(
        make_identity_tensor(make_shape(Int<kBlockN>{}, Int<64>{})), thr64);
    Tensor sfpB_coord = mxfp8::partition_SFB(
        make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<64>{})), thr64);

    auto scA128 = make_tiled_copy_A(SmemCopyAtomData{}, mma128); auto tscA128 = scA128.get_thread_slice(tid);
    auto scB128 = make_tiled_copy_B(SmemCopyAtomData{}, mma128); auto tscB128 = scB128.get_thread_slice(tid);
    auto scSFA128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma128), make_shape(size<0>(ts128), size<2>(ts128)));
    auto scSFB128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma128), make_shape(size<1>(ts128), size<2>(ts128)));
    auto tscSFA128 = scSFA128.get_thread_slice(tid); auto tscSFB128 = scSFB128.get_thread_slice(tid);
    auto scA64 = make_tiled_copy_A(SmemCopyAtomData{}, mma64); auto tscA64 = scA64.get_thread_slice(tid);
    auto scB64 = make_tiled_copy_B(SmemCopyAtomData{}, mma64); auto tscB64 = scB64.get_thread_slice(tid);


    // S9: static SF half via even/odd step pairs (streamed 128-row atoms)
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
    Tensor accK  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    clear(accV); clear(accK);
    Tensor accS  = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor accDP = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
    auto rc_view = [](auto& f) {
      return make_tensor(f.data(), make_layout(
          make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
          make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
    };
    Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
    constexpr int kNRow = 2, kNCol = 8;             // per-thread cols (q)
    int const row0 = (warp % 4) * 16 + lane / 4;    // kv row
    int const col0 = (warp / 4) * 8 + (lane % 4) * 2;  // q col base

    StateQD rqd; StateTt rtt;
    auto step = [&](int m, auto hc) {
      constexpr int hh0 = decltype(hc)::value;
      // produced P'/dS' tiles ALIAS the consumed QD ring stage (data dead after S'/dP';
      // stage release is delayed past the output gemms, so the producer can't refill).
      Tensor sP  = make_tensor(make_smem_ptr(ss.sQ.begin() + hh0 * 64 * 128), SmemLayoutDS{});
      Tensor sDS = make_tensor(make_smem_ptr(ss.sD.begin() + hh0 * 64 * 128), SmemLayoutDS{});
      ElementSF* sSFP_p  = ss.sSFQ.begin() + hh0 * 512;
      ElementSF* sSFDS_p = ss.sSFD.begin() + hh0 * 512;
      // lse/delta for this q tile: cooperative smem load, issued pre-wait
      if (tid < 64) {
        ss.sLse[tid] = p.lse[size_t(h) * p.S + m * kBlockM + tid];
        ss.sDlt[tid] = p.delta[size_t(h) * p.S + m * kBlockM + tid];
      }
      // ---- S' = K Q^T ; dP' = V dO^T (shared) ----
      { auto t = pipeline_qd.consumer_try_wait(rqd); pipeline_qd.consumer_wait(rqd, t);
        int stage = rqd.index();
        Tensor sSFQst = make_tensor(make_smem_ptr(ss.sSFQ.begin() + stage * 512), SmemLayoutSFT{});
        Tensor sSFDst = make_tensor(make_smem_ptr(ss.sSFD.begin() + stage * 512), SmemLayoutSFT{});
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sQ(_, _, stage))), tscB128.retile_D(tSrQ));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFQst)), tscSFB128.retile_D(tSrSFQ));
        copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sK)), tscA128.retile_D(tSrK));
        copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFK_v)), tscSFA128.retile_D(tSrSFK));
      }
      auto tSrSFQ_h = subSF(tSrSFQ, hc);
#ifdef DVDK_DEBUG
      if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && m < 2)
        printf("m=%d Q0=%f K0=%f SFQh0=%d SFK0=%d | t128 sees nothing here\n", m, float(Element::bitcast(tSrQ(0))), float(Element::bitcast(tSrK(0))),
               int(uint8_t(tSrSFQ_h(0).storage)), int(uint8_t(tSrSFK(0).storage)));
      if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 128 && m < 2)
        printf("m=%d t128 Q0=%f SFQh0=%d\n", m, float(Element::bitcast(tSrQ(0))), int(uint8_t(tSrSFQ_h(0).storage)));
#endif
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrK); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)),
                   make_zip_tensor(tSrQ(_, _, k), tSrSFQ_h(_, _, k)), accS);
#ifdef DVDK_DEBUG
      if (m == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        printf("accS00=%f accS13=%f\n", accS_rc(0,0), accS_rc(1,3));
#endif
      // ---- P' quantize per-32-along-q (accS dead after this) ----
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // prev m's readers done
      {
        float asP[kNRow][2] = {};
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) {
            int c = col0 + (ni & 1) + 16 * (ni / 2);
            float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - ss.sLse[c]) * kLog2e);
            accS_rc(mi, ni) = pv;
            asP[mi][ni / 4] = fmaxf(asP[mi][ni / 4], fabsf(pv));
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi)
          CUTLASS_PRAGMA_UNROLL
          for (int kb = 0; kb < 2; ++kb) {
            asP[mi][kb] = fmaxf(asP[mi][kb], __shfl_xor_sync(uint32_t(-1), asP[mi][kb], 1));
            asP[mi][kb] = fmaxf(asP[mi][kb], __shfl_xor_sync(uint32_t(-1), asP[mi][kb], 2));
            ss.sAmP[warp][(mi * 8 + lane / 4) * 2 + kb] = asP[mi][kb];
          }
        cutlass::arch::NamedBarrier(NumMmaThreads, kAmaxBarrier).sync();
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          int kv = row0 + mi * 8;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) {
            int c = col0 + (ni & 1) + 16 * (ni / 2);
            int kb = ni / 4;
            int slot = (mi * 8 + lane / 4) * 2 + kb;
            int ses = mx_scale_exp(fmaxf(ss.sAmP[warp][slot], ss.sAmP[warp ^ 4][slot]));
            if (warp < 4 && (lane % 4) == 0 && (ni & 3) == 0)
              sSFP_p[16 * (kv % 32) + 4 * (kv / 32) + kb] = ElementSF::bitcast(uint8_t(ses + 127));
            sP(kv, c) = quant_e4m3(accS_rc(mi, ni), ses);
          }
        }
      }
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // sP/SF visible

      // ---- dP' = V dO^T (accS's registers reusable now) ----
      { int stage = rqd.index();
        Tensor sSFDst = make_tensor(make_smem_ptr(ss.sSFD.begin() + stage * 512), SmemLayoutSFT{});
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sD(_, _, stage))), tscB128.retile_D(tSrD));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFDst)), tscSFB128.retile_D(tSrSFD));
        copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sV)), tscA128.retile_D(tSrV));
        copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFV_v)), tscSFA128.retile_D(tSrSFV));
      }
      auto tSrSFD_h = subSF(tSrSFD, hc);
      clear(accDP);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrV); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrV(_, _, k), tSrSFV(_, _, k)),
                   make_zip_tensor(tSrD(_, _, k), tSrSFD_h(_, _, k)), accDP);
      // ---- dS' from P_q8 readback ; dS' quantize ----
      float asD[kNRow][2] = {};
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < kNCol; ++ni) {
          int c = col0 + (ni & 1) + 16 * (ni / 2);
          int kv = row0 + mi * 8;
          float pv8 = float(Element::bitcast(sP(kv, c).storage)) *
                      exp2f(float(int(uint8_t(sSFP_p[16 * (kv % 32) + 4 * (kv / 32) + c / 32].storage)) - 127));
          float dsv = pv8 * (accDP_rc(mi, ni) - ss.sDlt[c]);
          accDP_rc(mi, ni) = dsv;
          asD[mi][ni / 4] = fmaxf(asD[mi][ni / 4], fabsf(dsv));
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi)
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < 2; ++kb) {
          asD[mi][kb] = fmaxf(asD[mi][kb], __shfl_xor_sync(uint32_t(-1), asD[mi][kb], 1));
          asD[mi][kb] = fmaxf(asD[mi][kb], __shfl_xor_sync(uint32_t(-1), asD[mi][kb], 2));
          ss.sAmD[warp][(mi * 8 + lane / 4) * 2 + kb] = asD[mi][kb];
        }
      cutlass::arch::NamedBarrier(NumMmaThreads, kAmaxBarrier).sync();
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int kv = row0 + mi * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < kNCol; ++ni) {
          int c = col0 + (ni & 1) + 16 * (ni / 2);
          int kb = ni / 4;
          int slot = (mi * 8 + lane / 4) * 2 + kb;
          int ses = mx_scale_exp(fmaxf(ss.sAmD[warp][slot], ss.sAmD[warp ^ 4][slot]));
          if (warp < 4 && (lane % 4) == 0 && (ni & 3) == 0)
            sSFDS_p[16 * (kv % 32) + 4 * (kv / 32) + kb] = ElementSF::bitcast(uint8_t(ses + 127));
          sDS(kv, c) = quant_e4m3(accDP_rc(mi, ni), ses);
        }
      }
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // P/DS/SF visible

      // ---- dV += P' Dt ; dK += dS' Qt ----
      { auto t = pipeline_tt.consumer_try_wait(rtt); pipeline_tt.consumer_wait(rtt, t);
        int stage = rtt.index();
        constexpr int hh = decltype(hc)::value;
        const uint8_t* sfDt_base = reinterpret_cast<const uint8_t*>(ss.sSFDt.begin()) + stage * 512 + 2 * hh;
        const uint8_t* sfQt_base = reinterpret_cast<const uint8_t*>(ss.sSFQt.begin()) + stage * 512 + 2 * hh;
        copy(scA64, tscA64.partition_S(as_position_independent_swizzle_tensor(sP)), tscA64.retile_D(tOrP));
        copy(scB64, tscB64.partition_S(as_position_independent_swizzle_tensor(sDt(_, _, stage))), tscB64.retile_D(tOrDt));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP(_, _, k)); ++i) {
            auto c = sfpA_coord(_, _, k)(i);
            int kv = int(get<0>(c)), q = int(get<1>(c));
            tOrSFP(_, _, k)(i) = ElementSF::bitcast(sSFP_p[16 * (kv % 32) + 4 * (kv / 32) + q / 32].storage);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDt(_, _, k)); ++i) {
            auto c = sfpB_coord(_, _, k)(i);
            int d = int(get<0>(c)), q = int(get<1>(c));
            tOrSFDt(_, _, k)(i) = ElementSF::bitcast(sfDt_base[16 * (d % 32) + 4 * (d / 32) + q / 32]);
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k)
          cute::gemm(mma64, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                     make_zip_tensor(tOrDt(_, _, k), tOrSFDt(_, _, k)), accV);
        copy(scA64, tscA64.partition_S(as_position_independent_swizzle_tensor(sDS)), tscA64.retile_D(tOrDS));
        copy(scB64, tscB64.partition_S(as_position_independent_swizzle_tensor(sQt(_, _, stage))), tscB64.retile_D(tOrQt));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrDS); ++k) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDS(_, _, k)); ++i) {
            auto c = sfpA_coord(_, _, k)(i);
            int kv = int(get<0>(c)), q = int(get<1>(c));
            tOrSFDS(_, _, k)(i) = ElementSF::bitcast(sSFDS_p[16 * (kv % 32) + 4 * (kv / 32) + q / 32].storage);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFQt(_, _, k)); ++i) {
            auto c = sfpB_coord(_, _, k)(i);
            int d = int(get<0>(c)), q = int(get<1>(c));
            tOrSFQt(_, _, k)(i) = ElementSF::bitcast(sfQt_base[16 * (d % 32) + 4 * (d / 32) + q / 32]);
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrDS); ++k)
          cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, k), tOrSFDS(_, _, k)),
                     make_zip_tensor(tOrQt(_, _, k), tOrSFQt(_, _, k)), accK);
        pipeline_tt.consumer_release(rtt); ++rtt;
        pipeline_qd.consumer_release(rqd); ++rqd;   // sQ/sD stages (aliased sP/sDS) dead now
      }
#ifdef DVDK_DEBUG
      if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        auto accV_rc = rc_view(accV);
        printf("accV row0 after m=%d: ", m);
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < 8; ++ni) printf("%.4f ", accV_rc(0, ni));
        printf("\n");
      }
#endif
    };
    for (int m = 0; m < MT; m += 2) {
      step(m, cute::Int<0>{});
      if (m + 1 < MT) step(m + 1, cute::Int<1>{});
    }

#ifdef DVDK_DEBUG
    if (p.dbg && blockIdx.x == 0 && blockIdx.y == 0) {
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier + 1).sync();
      for (int st = 0; st < 2; ++st) {
        Tensor sPd = make_tensor(make_smem_ptr(ss.sQ.begin() + st * 64 * 128), SmemLayoutDS{});
        Tensor sDSd = make_tensor(make_smem_ptr(ss.sD.begin() + st * 64 * 128), SmemLayoutDS{});
        const int STR = 2 * 64 * 64 + 1024;
        for (int i = tid; i < 64 * 64; i += NumMmaThreads) {
          p.dbg[st * STR + i] = sPd(i / 64, i % 64).storage;
          p.dbg[st * STR + 4096 + i] = sDSd(i / 64, i % 64).storage;
        }
        for (int i = tid; i < 512; i += NumMmaThreads) {
          p.dbg[st * STR + 8192 + i] = (ss.sSFQ.begin() + st * 512)[i].storage;
          p.dbg[st * STR + 8704 + i] = (ss.sSFD.begin() + st * 512)[i].storage;
        }
      }
    }
#endif
    // ---- epilogue: dV/dK [h, n-tile] ----
    Tensor gV = make_tensor(make_gmem_ptr(p.dV + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                            make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
    Tensor gK = make_tensor(make_gmem_ptr(p.dK + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                            make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accV, thr64.partition_C(gV));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accK, thr64.partition_C(gK));
  }
}

}  // namespace s3bdvdk
