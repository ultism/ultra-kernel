#pragma once
// S3B-DK-WS: warp-specialized TMA-pipelined MXFP8 attention dK kernel (sm_120a).
// Mirror of s3b_dq_ws_kernel.cuh with roles swapped: block = (kv tile n of 128,
// head). Resident: K, V (+SF). Ring over q tiles of 64: Q, dO (natural),
// Qt (host-transposed [d, S]) + SF. Per m: S'=KQ^T, dP'=V dO^T,
// dS'=P'*(dP'-delta) quantized per-32-along-q, dK += dS' * Qt.
// SF atoms 128-row aligned: Q/dO/Qt stream the 128-row atom, consumer slices
// the (m&1) half (S9 subSF pattern, even/odd pairs -> compile-time h).
// lse/delta are indexed by q (streamed): per-column LDG, prefetched pre-gemm.
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

namespace s3bdkws {

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

constexpr int TmaBytesQD = kBlockM * kHeadDim * 2 + 512 * 2;      // Q,dO data + SF atoms
constexpr int TmaBytesQt = kHeadDim * kBlockM + 512;

struct ParamsDk {
  TMA_Q tma_q; TMA_Q tma_d; TMA_Qt tma_qt; TMA_SFQ tma_sfq; TMA_SFQ tma_sfd; TMA_SFQt tma_sfqt;
  LayoutSFQ layout_sfq;      // (S, 64, 128, H)
  LayoutSFQt layout_sfqt;    // (128, 128, S, H)
  const uint8_t *K, *V;      // resident natural tiles [H, S, 128]
  const uint8_t *sfK, *sfV;  // flat 512B tiles, tile (h, nt) at ((h*NT128 + nt))*512
  const float* lse;          // [H, S] indexed by q
  const float* delta;        // [H, S] indexed by q
  float* dK;                 // [H, S, 128]
  int S, H;
  float sm_scale;
};

struct SharedStorageDk {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sD;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt>> sQt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutDS>> sDS;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFK, sSFV;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFQ, sSFD, sSFQt;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFDS;
  alignas(16) float sLse[64];   // current q tile
  alignas(16) float sDlt[64];
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
dk_ws_kernel(CUTE_GRID_CONSTANT ParamsDk const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDk*>(smem_raw);
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
  Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutKV{});
  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQ{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.sQt.begin()), SmemLayoutQt{});
  Tensor sDS = make_tensor(make_smem_ptr(ss.sDS.begin()), SmemLayoutDS{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFT{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFT{});

  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mQ3d = p.tma_q.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mD3d = p.tma_d.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mQt3d = p.tma_qt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
      Tensor mSFQ3d = p.tma_sfq.get_tma_tensor(shape(p.layout_sfq));
      Tensor mSFQt3d = p.tma_sfqt.get_tma_tensor(shape(p.layout_sfqt));
      auto bq = p.tma_q.get_slice(_0{}); auto bd = p.tma_d.get_slice(_0{}); auto bqt = p.tma_qt.get_slice(_0{});
      auto bsq = p.tma_sfq.get_slice(_0{}); auto bsqt = p.tma_sfqt.get_slice(_0{});
      Tensor mQ = mQ3d(_, _, h); Tensor mD = mD3d(_, _, h); Tensor mQt = mQt3d(_, _, h);
      Tensor mSFQ = mSFQ3d(_, _, h); Tensor mSFQt = mSFQt3d(_, _, h);
      Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));    // (M,K,mb)
      Tensor gD = local_tile(mD, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));  // (d,M,mb)
      Tensor gSFQ = local_tile(mSFQ, make_shape(Int<128>{}, Int<kHeadDim>{}), make_coord(_, _0{}));    // (128-row atom,hd,mb/2)
      Tensor gSFQt = local_tile(mSFQt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
      Tensor tQgQ = group_modes<0, 3>(bq.partition_S(gQ));
      Tensor tQsQ = group_modes<0, 3>(bq.partition_D(sQ));
      Tensor tQgD = group_modes<0, 3>(bd.partition_S(gD));
      Tensor tQsD = group_modes<0, 3>(bd.partition_D(sD));
      Tensor tQtgQt = group_modes<0, 3>(bqt.partition_S(gQt));
      Tensor tQtsQt = group_modes<0, 3>(bqt.partition_D(sQt));
      Tensor tQgSFQ = group_modes<0, 3>(bsq.partition_S(gSFQ));
      Tensor tQsSFQ = group_modes<0, 3>(bsq.partition_D(
          make_tensor(make_smem_ptr(ss.sSFQ.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      Tensor tQsSFD = group_modes<0, 3>(bsq.partition_D(
          make_tensor(make_smem_ptr(ss.sSFD.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      Tensor tQtgSFQt = group_modes<0, 3>(bsqt.partition_S(gSFQt));
      Tensor tQtsSFQt = group_modes<0, 3>(bsqt.partition_D(
          make_tensor(make_smem_ptr(ss.sSFQt.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      auto sQD = cutlass::make_producer_start_state<PipeQD>();
      auto sQt_ = cutlass::make_producer_start_state<PipeQt>();
      for (int m = 0; m < MT; ++m) {
        int const sfatom = m / 2;
        pipeline_qd.producer_acquire(sQD);
        copy(p.tma_q.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgQ(_, m), tQsQ(_, sQD.index()));
        copy(p.tma_d.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgD(_, m), tQsD(_, sQD.index()));
        copy(p.tma_sfq.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgSFQ(_, sfatom), tQsSFQ(_, sQD.index()));
        copy(p.tma_sfd.with(*pipeline_qd.producer_get_barrier(sQD), 0), tQgSFQ(_, sfatom), tQsSFD(_, sQD.index()));
        ++sQD;
        pipeline_qt.producer_acquire(sQt_);
        copy(p.tma_qt.with(*pipeline_qt.producer_get_barrier(sQt_), 0), tQtgQt(_, m), tQtsQt(_, sQt_.index()));
        copy(p.tma_sfqt.with(*pipeline_qt.producer_get_barrier(sQt_), 0), tQtgSFQt(_, sfatom), tQtsSFQt(_, sQt_.index()));
        ++sQt_;
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;
    int const warp = tid / 32, lane = tid % 32;

    // resident K, V: cooperative vectorized load by all consumer threads
    {
      auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                                Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
      auto tgc = gc.get_thread_slice(tid);
      auto nat = make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{}));
      Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      Tensor gVn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      copy(gc, tgc.partition_S(gKn), tgc.partition_D(sK));
      copy(gc, tgc.partition_S(gVn), tgc.partition_D(sV));
      for (int i = tid; i < 128; i += NumMmaThreads) {
        reinterpret_cast<uint32_t*>(ss.sSFK.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfK + (size_t(h) * NT128 + n) * 512)[i];
        reinterpret_cast<uint32_t*>(ss.sSFV.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfV + (size_t(h) * NT128 + n) * 512)[i];
      }
    }
    cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier + 1).sync();

    TiledMmaK128 mma128; TiledMmaK64 mma64;
    auto thr128 = mma128.get_thread_slice(tid);
    auto thr64 = mma64.get_thread_slice(tid);
    auto ts128 = tile_shape(mma128);

    // S'/dP' operand fragments (K=d=128): A = K/V resident, B = Q/dO streamed
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
    // dK operand fragments (K=q=64): A = dS' [kv,q], B = Qt [d,q]
    Tensor tOrDS = thr64.partition_fragment_A(sDS);
    Tensor tOrQt = thr64.partition_fragment_B(sQt(_, _, _0{}));
    Tensor tOrSFDS = mxfp8::partition_fragment_SFA(
        make_tensor(make_smem_ptr(ss.sSFDS.begin()), SmemLayoutSFK64{}), thr64);
    Tensor tOrSFQt = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFQt.begin()), SmemLayoutSFK64{}), thr64);
    // identity-coord gather views for the two in-kernel/half-atom SFs
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

    // resident K/V fragments + SF: load once
    copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sK)), tscA128.retile_D(tSrK));
    copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sV)), tscA128.retile_D(tSrV));
    copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFK)), tscSFA128.retile_D(tSrSFK));
    copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFA128.retile_D(tSrSFV));

    // S9: static SF half via even/odd step pairs
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
    clear(accK);
    Tensor accS  = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor accDP = partition_fragment_C(mma128, Shape<Int<kBlockN>, Int<kBlockM>>{});
    auto rc_view = [](auto& f) {
      return make_tensor(f.data(), make_layout(
          make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
          make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
    };
    Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
    constexpr int kNRow = 2, kNCol = kBlockM / 4;   // 16 cols (q) per thread
    int const row0 = warp * 16 + lane / 4;          // kv row
    int const col0 = (lane % 4) * 2;                // q col base

    StateQD rqd; StateQt rqt;
    auto step = [&](int m, auto hc) {
      // lse/delta for this q tile: cooperative 64-float smem load, issued pre-wait;
      // read in the quant phase (guarded by the kQuantBarrier pair, single buffer).
      if (tid < 64) {
        ss.sLse[tid] = p.lse[size_t(h) * p.S + m * kBlockM + tid];
        ss.sDlt[tid] = p.delta[size_t(h) * p.S + m * kBlockM + tid];
      }
      float lse_c[kNCol], dlt_c[kNCol];
      // ---- S' = K Q^T ; dP' = V dO^T ----
      { auto t = pipeline_qd.consumer_try_wait(rqd); pipeline_qd.consumer_wait(rqd, t);
        int stage = rqd.index();
        Tensor sSFQst = make_tensor(make_smem_ptr(ss.sSFQ.begin() + stage * 512), SmemLayoutSFT{});
        Tensor sSFDst = make_tensor(make_smem_ptr(ss.sSFD.begin() + stage * 512), SmemLayoutSFT{});
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sQ(_, _, stage))), tscB128.retile_D(tSrQ));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFQst)), tscSFB128.retile_D(tSrSFQ));
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sD(_, _, stage))), tscB128.retile_D(tSrD));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFDst)), tscSFB128.retile_D(tSrSFD));
      }
      auto tSrSFQ_h = subSF(tSrSFQ, hc);
      auto tSrSFD_h = subSF(tSrSFD, hc);
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrK); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)),
                   make_zip_tensor(tSrQ(_, _, k), tSrSFQ_h(_, _, k)), accS);
      clear(accDP);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrV); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrV(_, _, k), tSrSFV(_, _, k)),
                   make_zip_tensor(tSrD(_, _, k), tSrSFD_h(_, _, k)), accDP);
      pipeline_qd.consumer_release(rqd); ++rqd;

      // ---- dS' quantize per-32-along-q -> sDS [kv, q] + sSFDS ----
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // prev m's readers done
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < kNCol; ++ni) {
        int c = (ni / 2) * 8 + col0 + (ni % 2);
        lse_c[ni] = ss.sLse[c];
        dlt_c[ni] = ss.sDlt[c];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int kv = row0 + mi * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < kBlockM / SFVecSize; ++kb) {
          float as = 0.f;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_c[ni]) * kLog2e);
            float dsv = pv * (accDP_rc(mi, ni) - dlt_c[ni]);
            accDP_rc(mi, ni) = dsv;
            as = fmaxf(as, fabsf(dsv));
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
            sDS(kv, c) = quant_e4m3(accDP_rc(mi, ni), ses);
          }
        }
      }
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // DS/SF visible

      // ---- dK += dS' Qt ----
      { auto t = pipeline_qt.consumer_try_wait(rqt); pipeline_qt.consumer_wait(rqt, t);
        int stage = rqt.index();
        constexpr int hh = decltype(hc)::value;
        const uint8_t* sfQt_base = reinterpret_cast<const uint8_t*>(ss.sSFQt.begin()) + stage * 512 + 2 * hh;
        copy(scA64, tscA64.partition_S(as_position_independent_swizzle_tensor(sDS)), tscA64.retile_D(tOrDS));
        copy(scB64, tscB64.partition_S(as_position_independent_swizzle_tensor(sQt(_, _, stage))), tscB64.retile_D(tOrQt));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrDS); ++k) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDS(_, _, k)); ++i) {
            auto c = sfpA_coord(_, _, k)(i);
            int kv = int(get<0>(c)), q = int(get<1>(c));
            tOrSFDS(_, _, k)(i) = ElementSF::bitcast(ss.sSFDS.begin()[16 * (kv % 32) + 4 * (kv / 32) + q / 32].storage);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFQt(_, _, k)); ++i) {
            auto c = sfpB_coord(_, _, k)(i);
            int d = int(get<0>(c)), q = int(get<1>(c));
            tOrSFQt(_, _, k)(i) = ElementSF::bitcast(sfQt_base[16 * (d % 32) + 4 * (d / 32) + q / 32]);
          }
        }
        pipeline_qt.consumer_release(rqt); ++rqt; }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrDS); ++k)
        cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, k), tOrSFDS(_, _, k)),
                   make_zip_tensor(tOrQt(_, _, k), tOrSFQt(_, _, k)), accK);
    };
    for (int m = 0; m < MT; m += 2) {
      step(m, cute::Int<0>{});
      if (m + 1 < MT) step(m + 1, cute::Int<1>{});
    }

    // ---- epilogue: dK[h, n-tile] ----
    Tensor gO = make_tensor(make_gmem_ptr(p.dK + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                            make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accK, thr64.partition_C(gO));
  }
}

}  // namespace s3bdkws
