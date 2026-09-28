#pragma once
// S3B-DVDK2: fused dv+dk experiment at kv=128, WS and non-WS variants.
// Template<bool kWS>: kWS=true  -> 384 threads, WG0 producer (TMA) @24 regs,
//                                 consumers @232 (existing dk/dv structure).
//                     kWS=false -> 256 homogeneous threads, 255 regs/thread,
//                                 warp-0 elected lane issues TMA inline
//                                 (ProducerConsumer), issue-ahead after the
//                                 per-step pipe releases.
// Question under test: does ptxas lifetime-overlap squeeze the fused kernel
// into the budget without spills (FA2 d=128 bwd: 250-255 regs, 0 spills).
// Loop per q-tile of 64: S'=KQ^T, dP'=VdO^T, P=exp2(...), dS=P*(dP-delta),
// P' quant const-scale (se=-8) -> shfl -> dV += P'*Dt ; dS' quant dynamic ->
// shfl -> dK += dS'*Qt. Register-shfl quant (c1), paired cvt (c9), TMA
// lse/delta (c11), const-scale P' (c12) all ported.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cuda_fp8.h>
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

namespace s3bdvdk2 {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockN = 128, kBlockM = 64, SFVecSize = 32, kStages = 2;
constexpr float kLog2e = 1.4426950408889634f;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TiledMmaK128 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _128>{}));
using TiledMmaK64 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _64>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomKV = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockM>>());
using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kHeadDim>, Int<kBlockM>, Int<kStages>>{}));
// TT ring (Qt/Dt) is SINGLE-stage: 2-stage would overflow the 99KB smem optin
// (K,V 32KB resident + 4x8KB x2 rings + SF + lse ~= 102KB > 101376B). The TT
// wait sits after the quant block, so TMA latency is still partially covered.
using SmemLayoutQt1 = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kHeadDim>, Int<kBlockM>, Int<1>>{}));
using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kBlockM>>{}));

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
using SmemLayoutSFK64 = decltype(make_layout(
    make_shape(make_shape(_1{}, _32{}, _4{}), make_shape(_32{}, _2{})),
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
using TMA_Lse = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)),
        make_shape(0), make_stride(_1{})),
    Layout<Shape<_256>>{}, make_shape(Int<256>{}), _1{}));

using PipeQD  = cutlass::PipelineTmaAsync<kStages>;    // Q + dO + SFQ + SFD (lse/delta via LDG)
using PipeTT  = cutlass::PipelineTmaAsync<1>;           // Qt + SFQt + Dt + SFDt (single-stage ring)
using StateQD = cutlass::PipelineState<kStages>;
using StateTT = cutlass::PipelineState<1>;

constexpr int TmaBytesQD = kBlockM * kHeadDim * 2 + 512 * 2;
constexpr int TmaBytesTT = (kHeadDim * kBlockM + 512) * 2;

struct ParamsDvdk {
  TMA_Q tma_q; TMA_Q tma_d; TMA_Qt tma_qt; TMA_Qt tma_dt;
  TMA_SFQ tma_sfq; TMA_SFQ tma_sfd; TMA_SFQt tma_sfqt; TMA_SFQt tma_sfdt;
  TMA_Lse tma_lse; TMA_Lse tma_dlt;
  LayoutSFQ layout_sfq;       // (S, 64, 128, H)
  LayoutSFQt layout_sfqt;     // (128, 128, S, H)  [Qt]
  LayoutSFQt layout_sfdt;     // (128, 128, S, H)  [Dt]
  const uint8_t *K, *V;
  const uint8_t *sfK, *sfV;
  const float *lse_raw, *dlt_raw;   // LDG fallback path
  float* dK;
  float* dV;
  int S, H;
  float sm_scale;
};

struct SharedStorageDvdk {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sD;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt1>> sQt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt1>> sDt;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFK, sSFV;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFQ, sSFD;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFQt, sSFDt;
  alignas(16) float sLse[kStages][64];
  alignas(16) float sDlt[kStages][64];
  alignas(8) typename PipeQD::SharedStorage pipeline_qd;
  alignas(8) typename PipeTT::SharedStorage pipeline_tt;
};

__device__ __forceinline__ int mx_scale_exp(float amax) {
  // c15: ceil(log2(x))-8 without MUFU: for finite x>0,
  // ceil(log2(x)) == unbiased_exp + (mantissa != 0), exactly.
  uint32_t b = __float_as_uint(amax);
  if ((b & 0x7fffffffu) == 0 || (b & 0x7f800000u) == 0) return -126;   // zero/subnormal
  if ((b & 0x7f800000u) == 0x7f800000u) return 127;                    // inf/nan
  int e = int(b >> 23) - 127 + ((b & 0x007fffffu) != 0 ? 1 : 0) - 8;
  return max(-126, min(127, e));
}

template <bool kWS>
__global__ void __launch_bounds__(kWS ? 384 : 256, 1)
dvdk2_kernel(CUTE_GRID_CONSTANT ParamsDvdk const p) {
  constexpr int NumMma = 256;
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDvdk*>(smem_raw);
  int const n = blockIdx.x, h = blockIdx.y;
  int const NT128 = p.S / kBlockN;
  int const MT = p.S / kBlockM;

  int const wg = cutlass::canonical_warp_group_idx();
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();
  // non-WS: ALL of warp 0 plays producer (warp-converged producer_acquire —
  // a divergent elected lane would race ahead of its warp's LDSM/shfl);
  // the TMA copies inside issue() are elect-guarded.
  bool const is_producer = kWS ? (wg == 0 && warp_in_wg == 0 && elect)
                               : (threadIdx.x / 32 == 0);

  typename PipeQD::Params pqd;
  if constexpr (kWS) {
    pqd.role = (wg == 0) ? PipeQD::ThreadCategory::Producer : PipeQD::ThreadCategory::Consumer;
    pqd.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  } else {
    pqd.role = is_producer ? PipeQD::ThreadCategory::ProducerConsumer
                           : PipeQD::ThreadCategory::Consumer;
    pqd.is_leader = (threadIdx.x == 0);
  }
  pqd.num_consumers = NumMma;
  pqd.transaction_bytes = TmaBytesQD;
  typename PipeTT::Params ptt;
  ptt.role = (pqd.role == PipeQD::ThreadCategory::Producer) ? PipeTT::ThreadCategory::Producer
             : (pqd.role == PipeQD::ThreadCategory::ProducerConsumer) ? PipeTT::ThreadCategory::ProducerConsumer
             : PipeTT::ThreadCategory::Consumer;
  ptt.is_leader = pqd.is_leader; ptt.num_consumers = NumMma;
  ptt.transaction_bytes = TmaBytesTT;
  PipeQD pipeline_qd(ss.pipeline_qd, pqd, Shape<_1, _1, _1>{});
  PipeTT pipeline_tt(ss.pipeline_tt, ptt, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
  Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQ{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.sQt.begin()), SmemLayoutQt1{});
  Tensor sDt = make_tensor(make_smem_ptr(ss.sDt.begin()), SmemLayoutQt1{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFT{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFT{});

  // TMA partition setup; returns issue(m, do_qd, do_tt) issuing each pipe's
  // tile m independently (QD is 2-stage/deep-ahead, TT is 1-stage/in-step).
  // Producer states live in kernel scope (safe capture by reference).
  auto sQDprod = cutlass::make_producer_start_state<PipeQD>();
  auto sTTprod = cutlass::make_producer_start_state<PipeTT>();
  auto make_issue = [&]() {
    Tensor mQ3d = p.tma_q.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
    Tensor mD3d = p.tma_d.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
    Tensor mQt3d = p.tma_qt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
    Tensor mDt3d = p.tma_dt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
    Tensor mSFQ3d = p.tma_sfq.get_tma_tensor(shape(p.layout_sfq));
    Tensor mSFQt3d = p.tma_sfqt.get_tma_tensor(shape(p.layout_sfqt));
    Tensor mSFDt3d = p.tma_sfdt.get_tma_tensor(shape(p.layout_sfdt));
    auto bq = p.tma_q.get_slice(_0{}); auto bd = p.tma_d.get_slice(_0{});
    auto bqt = p.tma_qt.get_slice(_0{}); auto bdt = p.tma_dt.get_slice(_0{});
    auto bsq = p.tma_sfq.get_slice(_0{}); auto bsqt = p.tma_sfqt.get_slice(_0{});
    auto bsdt = p.tma_sfdt.get_slice(_0{});
    Tensor mQ = mQ3d(_, _, h); Tensor mD = mD3d(_, _, h);
    Tensor mQt = mQt3d(_, _, h); Tensor mDt = mDt3d(_, _, h);
    Tensor mSFQ = mSFQ3d(_, _, h); Tensor mSFQt = mSFQt3d(_, _, h); Tensor mSFDt = mSFDt3d(_, _, h);
    Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
    Tensor gD = local_tile(mD, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
    Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
    Tensor gDt = local_tile(mDt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
    Tensor gSFQ = local_tile(mSFQ, make_shape(Int<128>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
    Tensor gSFQt = local_tile(mSFQt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
    Tensor gSFDt = local_tile(mSFDt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
    auto tQgQ = group_modes<0, 3>(bq.partition_S(gQ));
    auto tQsQ = group_modes<0, 3>(bq.partition_D(sQ));
    auto tQgD = group_modes<0, 3>(bd.partition_S(gD));
    auto tQsD = group_modes<0, 3>(bd.partition_D(sD));
    auto tQtgQt = group_modes<0, 3>(bqt.partition_S(gQt));
    auto tQtsQt = group_modes<0, 3>(bqt.partition_D(sQt));
    auto tDtgDt = group_modes<0, 3>(bdt.partition_S(gDt));
    auto tDtsDt = group_modes<0, 3>(bdt.partition_D(sDt));
    auto tQgSFQ = group_modes<0, 3>(bsq.partition_S(gSFQ));
    auto sSFQ2 = make_tensor(make_smem_ptr(ss.sSFQ.begin()),
        make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}), append(stride(SmemLayoutSFT{}), Int<512>{})));
    auto sSFD2 = make_tensor(make_smem_ptr(ss.sSFD.begin()),
        make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}), append(stride(SmemLayoutSFT{}), Int<512>{})));
    auto sSFQt2 = make_tensor(make_smem_ptr(ss.sSFQt.begin()),
        make_layout(append(shape(SmemLayoutSFT{}), Int<1>{}), append(stride(SmemLayoutSFT{}), Int<512>{})));
    auto sSFDt2 = make_tensor(make_smem_ptr(ss.sSFDt.begin()),
        make_layout(append(shape(SmemLayoutSFT{}), Int<1>{}), append(stride(SmemLayoutSFT{}), Int<512>{})));
    auto tQsSFQ = group_modes<0, 3>(bsq.partition_D(sSFQ2));
    auto tQsSFD = group_modes<0, 3>(bsq.partition_D(sSFD2));
    auto tQtgSFQt = group_modes<0, 3>(bsqt.partition_S(gSFQt));
    auto tQtsSFQt = group_modes<0, 3>(bsqt.partition_D(sSFQt2));
    auto tDtgSFDt = group_modes<0, 3>(bsdt.partition_S(gSFDt));
    auto tDtsSFDt = group_modes<0, 3>(bsdt.partition_D(sSFDt2));
    Tensor mLse1d = p.tma_lse.get_tma_tensor(make_shape(int(p.S) * int(p.H) * 4));
    Tensor mDlt1d = p.tma_dlt.get_tma_tensor(make_shape(int(p.S) * int(p.H) * 4));
    auto blse = p.tma_lse.get_slice(_0{}); auto bdlt = p.tma_dlt.get_slice(_0{});
    Tensor gLse = local_tile(mLse1d, make_shape(Int<256>{}), make_coord(_));
    Tensor gDlt = local_tile(mDlt1d, make_shape(Int<256>{}), make_coord(_));
    Tensor sLseT = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(ss.sLse)),
        make_layout(make_shape(Int<256>{}, Int<kStages>{}), make_stride(_1{}, Int<256>{})));
    Tensor sDltT = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(ss.sDlt)),
        make_layout(make_shape(Int<256>{}, Int<kStages>{}), make_stride(_1{}, Int<256>{})));
    auto tLgL = group_modes<0, 2>(blse.partition_S(gLse));
    auto tLsL = group_modes<0, 2>(blse.partition_D(sLseT));
    auto tLgD = group_modes<0, 2>(bdlt.partition_S(gDlt));
    auto tLsD = group_modes<0, 2>(bdlt.partition_D(sDltT));
    // NOTE: capture the partition tensors BY VALUE — they are make_issue locals
    // and would dangle if captured by reference into the escaping lambda.
    return [=, &p, &pipeline_qd, &pipeline_tt, &sQDprod, &sTTprod](int m, bool do_qd, bool do_tt) {
      int const sfatom = m / 2;
      if (do_qd) {
        pipeline_qd.producer_acquire(sQDprod);
        if (elect) {
          copy(p.tma_q.with(*pipeline_qd.producer_get_barrier(sQDprod), 0), tQgQ(_, m), tQsQ(_, sQDprod.index()));
          copy(p.tma_d.with(*pipeline_qd.producer_get_barrier(sQDprod), 0), tQgD(_, m), tQsD(_, sQDprod.index()));
          copy(p.tma_sfq.with(*pipeline_qd.producer_get_barrier(sQDprod), 0), tQgSFQ(_, sfatom), tQsSFQ(_, sQDprod.index()));
          copy(p.tma_sfd.with(*pipeline_qd.producer_get_barrier(sQDprod), 0), tQgSFQ(_, sfatom), tQsSFD(_, sQDprod.index()));
        }
        ++sQDprod;
      }
      if (do_tt) {
        pipeline_tt.producer_acquire(sTTprod);
        if (elect) {
          copy(p.tma_qt.with(*pipeline_tt.producer_get_barrier(sTTprod), 0), tQtgQt(_, m), tQtsQt(_, sTTprod.index()));
          copy(p.tma_sfqt.with(*pipeline_tt.producer_get_barrier(sTTprod), 0), tQtgSFQt(_, sfatom), tQtsSFQt(_, sTTprod.index()));
          copy(p.tma_dt.with(*pipeline_tt.producer_get_barrier(sTTprod), 0), tDtgDt(_, m), tDtsDt(_, sTTprod.index()));
          copy(p.tma_sfdt.with(*pipeline_tt.producer_get_barrier(sTTprod), 0), tDtgSFDt(_, sfatom), tDtsSFDt(_, sTTprod.index()));
        }
        ++sTTprod;
      }
    };
  };

  if constexpr (kWS) {
    if (wg == 0) {
      cutlass::arch::warpgroup_reg_dealloc<24>();
      if (is_producer) {
        auto issue = make_issue();
        for (int m = 0; m < MT; ++m) issue(m, true, true);
      }
      return;
    }
    cutlass::arch::warpgroup_reg_alloc<232>();
  }

  // ---- consumers ----
  int const tid = kWS ? (threadIdx.x - 128) : threadIdx.x;
  int const lane = tid % 32;

  {
    auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                              Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
    auto tgc = gc.get_thread_slice(tid);
    auto nat = make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{}));
    Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
    Tensor gVn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
    copy(gc, tgc.partition_S(gKn), tgc.partition_D(sK));
    copy(gc, tgc.partition_S(gVn), tgc.partition_D(sV));
    for (int i = tid; i < 128; i += NumMma) {
      reinterpret_cast<uint32_t*>(ss.sSFK.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfK + (size_t(h) * NT128 + n) * 512)[i];
      reinterpret_cast<uint32_t*>(ss.sSFV.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfV + (size_t(h) * NT128 + n) * 512)[i];
    }
  }
  if constexpr (kWS) {
    cutlass::arch::NamedBarrier(NumMma, 1).sync();
  } else {
    __syncthreads();
  }

  TiledMmaK128 mma128; TiledMmaK64 mma64;
  auto thr128 = mma128.get_thread_slice(tid);
  auto thr64 = mma64.get_thread_slice(tid);
  auto ts128 = tile_shape(mma128);

  Tensor tSrK  = thr128.partition_fragment_A(sK);
  Tensor tSrV  = thr128.partition_fragment_A(sV);
  Tensor tSrQ  = thr128.partition_fragment_B(sQ(_, _, _0{}));
  Tensor tSrD  = thr128.partition_fragment_B(sD(_, _, _0{}));
  Tensor tSrSFK = mxfp8::partition_fragment_SFA(sSFK, thr128);
  Tensor tSrSFV = mxfp8::partition_fragment_SFA(sSFV, thr128);
  Tensor tSrSFQ = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFT{}), thr128);
  Tensor tSrSFD = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(ss.sSFD.begin()), SmemLayoutSFT{}), thr128);
  Tensor sDSshape = make_tensor(make_smem_ptr(static_cast<Element*>(nullptr)), SmemLayoutDS{});
  Tensor tOrP  = thr64.partition_fragment_A(sDSshape);
  Tensor tOrDS = thr64.partition_fragment_A(sDSshape);
  Tensor tOrQt = thr64.partition_fragment_B(sQt(_, _, _0{}));
  Tensor tOrDt = thr64.partition_fragment_B(sDt(_, _, _0{}));
  Tensor tOrSFP = mxfp8::partition_fragment_SFA(
      make_tensor(make_smem_ptr(static_cast<ElementSF*>(nullptr)), SmemLayoutSFK64{}), thr64);
  Tensor tOrSFDS = mxfp8::partition_fragment_SFA(
      make_tensor(make_smem_ptr(static_cast<ElementSF*>(nullptr)), SmemLayoutSFK64{}), thr64);
  Tensor tOrSFQt = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(ss.sSFQt.begin()), SmemLayoutSFK64{}), thr64);
  Tensor tOrSFDt = mxfp8::partition_fragment_SFB(
      make_tensor(make_smem_ptr(ss.sSFDt.begin()), SmemLayoutSFK64{}), thr64);
  Tensor sfpB_coord = mxfp8::partition_SFB(
      make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);

  auto scA128 = make_tiled_copy_A(SmemCopyAtomData{}, mma128); auto tscA128 = scA128.get_thread_slice(tid);
  auto scB128 = make_tiled_copy_B(SmemCopyAtomData{}, mma128); auto tscB128 = scB128.get_thread_slice(tid);
  auto scSFA128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma128), make_shape(size<0>(ts128), size<2>(ts128)));
  auto scSFB128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma128), make_shape(size<1>(ts128), size<2>(ts128)));
  auto tscSFA128 = scSFA128.get_thread_slice(tid); auto tscSFB128 = scSFB128.get_thread_slice(tid);
  auto scB64 = make_tiled_copy_B(SmemCopyAtomData{}, mma64); auto tscB64 = scB64.get_thread_slice(tid);

  // lifetime variant: DATA fragments are streamed per-k inside the gemm loops
  // (FA2 A_in_regs=false style); only the tiny SF A-frags stay resident.
  copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFK)), tscSFA128.retile_D(tSrSFK));
  copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFA128.retile_D(tSrSFV));

  auto subSF = [](auto const& f, auto hc) {
    auto m1 = get<1>(f.layout()); auto a = get<0>(m1); auto b = get<1>(m1);
    auto nb = shape(b); auto sb = stride(b);
    auto t = make_tensor(f.data(), make_layout(get<0>(f.layout()),
        make_layout(make_shape(shape(a), make_shape(nb / _2{}, _2{})),
                    make_stride(stride(a), make_stride(sb, sb * (nb / _2{})))),
        get<2>(f.layout())))(_, make_coord(_, make_coord(_, hc)), _);
    return group_modes<1, 3>(t);
  };

  Tensor accK  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
  Tensor accV  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
  clear(accK); clear(accV);
  Tensor accS  = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
  Tensor accDP = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
  auto rc_view = [](auto& f) {
    return make_tensor(f.data(), make_layout(
        make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
        make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
  };
  Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
  constexpr int kNRow = 2, kNCol = kBlockM / 4;
  int const col0 = (lane % 4) * 2;

  StateQD rqd; StateTT rtt;

  auto step = [&](int m, auto hc, auto&& issue_next) {
    int stage_qd;
    { auto t = pipeline_qd.consumer_try_wait(rqd); pipeline_qd.consumer_wait(rqd, t);
      stage_qd = rqd.index();
      Tensor sSFQst = make_tensor(make_smem_ptr(ss.sSFQ.begin() + stage_qd * 512), SmemLayoutSFT{});
      Tensor sSFDst = make_tensor(make_smem_ptr(ss.sSFD.begin() + stage_qd * 512), SmemLayoutSFT{});
      copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFQst)), tscSFB128.retile_D(tSrSFQ));
      copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFDst)), tscSFB128.retile_D(tSrSFD));
    }
    auto tSrSFQ_h = subSF(tSrSFQ, hc);
    auto tSrSFD_h = subSF(tSrSFD, hc);
    auto tscK = tscA128.partition_S(as_position_independent_swizzle_tensor(sK));
    auto tscV = tscA128.partition_S(as_position_independent_swizzle_tensor(sV));
    auto tscQ = tscB128.partition_S(as_position_independent_swizzle_tensor(sQ(_, _, stage_qd)));
    auto tscD = tscB128.partition_S(as_position_independent_swizzle_tensor(sD(_, _, stage_qd)));
    auto tcrK = tscA128.retile_D(tSrK); auto tcrV = tscA128.retile_D(tSrV);
    auto tcrQ = tscB128.retile_D(tSrQ); auto tcrD = tscB128.retile_D(tSrD);
    // c15: issue the lse/delta LDGs BEFORE the gemm bursts — 32 mma of
    // latency hide the global load; use site is the quant phase below.
    float lse_s[2], dlt_s[2];
    {
      int const c0 = (lane >> 2) + 8 * (lane & 3);
      const float* lb = p.lse_raw + size_t(h) * p.S + m * kBlockM + c0;
      const float* db = p.dlt_raw + size_t(h) * p.S + m * kBlockM + c0;
      lse_s[0] = lb[0]; lse_s[1] = lb[32];
      dlt_s[0] = db[0]; dlt_s[1] = db[32];
    }
    int const lsh_base = (lane & 3) * 8;
    clear(accS);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrK); ++k) {
      copy(scA128, tscK(_, _, k), tcrK(_, _, k));
      copy(scB128, tscQ(_, _, k), tcrQ(_, _, k));
      cute::gemm(mma128, make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)),
                 make_zip_tensor(tSrQ(_, _, k), tSrSFQ_h(_, _, k)), accS);
    }
    clear(accDP);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tSrV); ++k) {
      copy(scA128, tscV(_, _, k), tcrV(_, _, k));
      copy(scB128, tscD(_, _, k), tcrD(_, _, k));
      cute::gemm(mma128, make_zip_tensor(tSrV(_, _, k), tSrSFV(_, _, k)),
                 make_zip_tensor(tSrD(_, _, k), tSrSFD_h(_, _, k)), accDP);
    }

    // FA3-style distributed stats (ShuffleLSE/ShuffledPsum): 64 cols' lse/dlt
    // per tile are stored ONCE per warp instead of per-thread: lane l holds
    // cols (l/4) + 8*(l%4) + 32*k (k<2) -> 2+2 regs vs 16+16. Fetched via
    // shfl at use: col(ni) = 2*(l%4) + 8*(ni>>1) + (ni&1) is held by lane
    // L = (l%4)*8 + (ni&1)*4 + ((ni>>1)&3), slot ni>>3 (== kb, compile-time).
    pipeline_qd.consumer_release(rqd); ++rqd;
    issue_next(m, true, false);   // non-WS: QD tile m+kStages (2-stage ahead)

    // P into accS_rc (kept for P'), dsv into accDP_rc with dynamic amax
    int ses_r[kNRow][kBlockM / SFVecSize];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kNRow; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int kb = 0; kb < kBlockM / SFVecSize; ++kb) {
        float as = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 8; ++j) {
          int ni = kb * 8 + j;
          int const L = lsh_base + ((j & 1) << 2) + ((j >> 1) & 3);
          float const lse = __shfl_sync(0xffffffffu, lse_s[kb], L);
          float const dlt = __shfl_sync(0xffffffffu, dlt_s[kb], L);
          float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse) * kLog2e);
          accS_rc(mi, ni) = pv;
          float dsv = pv * (accDP_rc(mi, ni) - dlt);
          accDP_rc(mi, ni) = dsv;
          as = fmaxf(as, fabsf(dsv));
        }
        as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 1));
        as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 2));
        ses_r[mi][kb] = mx_scale_exp(as);
      }
    }
    uint32_t qw[kNRow][kNCol / 4];
    auto shfl_fill = [&](auto& tOr, uint32_t const (&q)[kNRow][kNCol / 4]) {
      Tensor t_u32 = recast<uint32_t>(tOr);
      int const qb = lane & ~3, off = 2 * (lane & 1), half = (lane >> 1) & 1;
      CUTLASS_PRAGMA_UNROLL
      for (int mk = 0; mk < size<2>(t_u32); ++mk) {
        CUTLASS_PRAGMA_UNROLL
        for (int e2 = 0; e2 < 2; ++e2) {
          int const g = e2 + 2 * mk;
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < kNRow; ++r) {
            uint32_t wlo = __shfl_sync(0xffffffffu, q[r][g], qb + off);
            uint32_t whi = __shfl_sync(0xffffffffu, q[r][g], qb + off + 1);
            uint32_t lo = half ? (wlo >> 16) : (wlo & 0xffffu);
            uint32_t hi = half ? (whi >> 16) : (whi & 0xffffu);
            t_u32(make_coord(_0{}, r, e2), _0{}, mk) = lo | (hi << 16);
          }
        }
      }
    };
    // P' quant (const scale 2^8) -> tOrP ; accS_rc dies after this
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kNRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int g = 0; g < kNCol / 4; ++g) {
        uint32_t lo = __nv_cvt_float2_to_fp8x2(
            make_float2(accS_rc(r, 4 * g) * 256.f, accS_rc(r, 4 * g + 1) * 256.f),
            __NV_SATFINITE, __NV_E4M3);
        uint32_t hi = __nv_cvt_float2_to_fp8x2(
            make_float2(accS_rc(r, 4 * g + 2) * 256.f, accS_rc(r, 4 * g + 3) * 256.f),
            __NV_SATFINITE, __NV_E4M3);
        qw[r][g] = lo | (hi << 16);
      }
    }
    shfl_fill(tOrP, qw);
    // dS' quant (dynamic ses) -> tOrDS ; accDP_rc dies after this
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kNRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int g = 0; g < kNCol / 4; ++g) {
        float const sc = exp2f(float(-ses_r[r][g >> 1]));
        uint32_t lo = __nv_cvt_float2_to_fp8x2(
            make_float2(accDP_rc(r, 4 * g) * sc, accDP_rc(r, 4 * g + 1) * sc),
            __NV_SATFINITE, __NV_E4M3);
        uint32_t hi = __nv_cvt_float2_to_fp8x2(
            make_float2(accDP_rc(r, 4 * g + 2) * sc, accDP_rc(r, 4 * g + 3) * sc),
            __NV_SATFINITE, __NV_E4M3);
        qw[r][g] = lo | (hi << 16);
      }
    }
    shfl_fill(tOrDS, qw);

    int stage_tt;
    { auto t = pipeline_tt.consumer_try_wait(rtt); pipeline_tt.consumer_wait(rtt, t);
      int stage = rtt.index();
      stage_tt = stage;
      constexpr int hh = decltype(hc)::value;
      const uint8_t* sfQt_base = reinterpret_cast<const uint8_t*>(ss.sSFQt.begin()) + stage * 512 + 2 * hh;
      const uint8_t* sfDt_base = reinterpret_cast<const uint8_t*>(ss.sSFDt.begin()) + stage * 512 + 2 * hh;

      {
        ElementSF const b = ElementSF::bitcast(uint8_t(-8 + 127));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrSFP); ++k)
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP(_, _, k)); ++i) tOrSFP(_, _, k)(i) = b;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrSFDS); ++k) {
        int const sel = (lane & 1) ? ses_r[1][k] : ses_r[0][k];
        ElementSF const b = ElementSF::bitcast(uint8_t(sel + 127));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tOrSFDS(_, _, k)); ++i) tOrSFDS(_, _, k)(i) = b;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrSFQt); ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < size(tOrSFQt(_, _, k)) / 32; ++r) {
          auto c = sfpB_coord(_, _, k)(32 * r);
          int d = int(get<0>(c)), kv = int(get<1>(c));
          ElementSF const b = ElementSF::bitcast(sfQt_base[16 * (d % 32) + 4 * (d / 32) + kv / 32]);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 32; ++i) tOrSFQt(_, _, k)(32 * r + i) = b;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrSFDt); ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < size(tOrSFDt(_, _, k)) / 32; ++r) {
          auto c = sfpB_coord(_, _, k)(32 * r);
          int d = int(get<0>(c)), kv = int(get<1>(c));
          ElementSF const b = ElementSF::bitcast(sfDt_base[16 * (d % 32) + 4 * (d / 32) + kv / 32]);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 32; ++i) tOrSFDt(_, _, k)(32 * r + i) = b;
        }
      }
    }   // TT stage NOT released yet: the per-k LDSM streams below still read it
    auto tscQt = tscB64.partition_S(as_position_independent_swizzle_tensor(sQt(_, _, stage_tt)));
    auto tscDt = tscB64.partition_S(as_position_independent_swizzle_tensor(sDt(_, _, stage_tt)));
    auto tcrQt = tscB64.retile_D(tOrQt); auto tcrDt = tscB64.retile_D(tOrDt);

    // c14: single fused loop — both LDSM batches issue before the mma bursts,
    // and the 16 mma run with 8 independent acc chains (accV | accK) instead
    // of two serialized 4-chain bursts.
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tOrP); ++k) {
      copy(scB64, tscDt(_, _, k), tcrDt(_, _, k));
      copy(scB64, tscQt(_, _, k), tcrQt(_, _, k));
      cute::gemm(mma64, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                 make_zip_tensor(tOrDt(_, _, k), tOrSFDt(_, _, k)), accV);
      cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, k), tOrSFDS(_, _, k)),
                 make_zip_tensor(tOrQt(_, _, k), tOrSFQt(_, _, k)), accK);
    }
    pipeline_tt.consumer_release(rtt); ++rtt;
    issue_next(m, false, true);   // non-WS: TT tile m+1 (single-stage ring)
  };

  if constexpr (!kWS) {
    auto issue = make_issue();
    // QD tile m is issued at step m-2; TT tile m at step m-1 (1-stage ring).
    auto issuew = [&](int m, bool qd, bool tt) {
      if (!is_producer) return;
      if (qd && m + kStages < MT) issue(m + kStages, true, false);
      if (tt && m + 1 < MT) issue(m + 1, false, true);
    };
    if (is_producer) { issue(0, true, true); if (MT > 1) issue(1, true, false); }
    for (int m = 0; m < MT; m += 2) {
      step(m, cute::Int<0>{}, issuew);
      if (m + 1 < MT) step(m + 1, cute::Int<1>{}, issuew);
    }
  } else {
    auto noissue = [](int, bool, bool) {};
    for (int m = 0; m < MT; m += 2) {
      step(m, cute::Int<0>{}, noissue);
      if (m + 1 < MT) step(m + 1, cute::Int<1>{}, noissue);
    }
  }

  Tensor gK = make_tensor(make_gmem_ptr(p.dK + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                          make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
  Tensor gV = make_tensor(make_gmem_ptr(p.dV + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                          make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accK, thr64.partition_C(gK));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accV, thr64.partition_C(gV));
}

}  // namespace s3bdvdk2
