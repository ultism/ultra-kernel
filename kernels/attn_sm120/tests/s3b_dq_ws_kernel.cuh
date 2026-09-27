#pragma once
// S3B-WS: warp-specialized TMA-pipelined MXFP8 attention dq kernel (sm_120a).
// Mirrors s3_kernel.cuh structure: 1 producer warpgroup (TMA) + 2 consumer
// warpgroups (256 MMA threads), 2-stage ring over kv tiles of 64.
//   block = (m_tile of 128 q, head). Resident: Q, dO (+SF). Ring: K, V, Kt
//   (host-transposed K [d, S]) + SF. Per n: S=QK^T, dP=dO V^T, dS=P*(dP-delta)
//   quantized per-32-along-kv (fwd S5 quad_reduce pattern) to smem,
//   dQ += dS * Kt.  SF atoms are 128-key aligned: K/V/Kt stream the 128-key
//   atom and the consumer slices the (n&1) 64-key half (S9 subSF pattern,
//   driven in even/odd pairs so h is compile-time).
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

namespace s3bws {

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 64, SFVecSize = 32, kStages = 2;
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;      // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
// S / dP: (M=q 128, N=kv 64, K=d 128). dQ: (M=q 128, N=d 128, K=kv 64).
using TiledMmaK128 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _128>{}));
using TiledMmaK64 = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _64>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutAtomKV = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockN>>());
// ring tiles: K/V natural [kv, d] 64x128 ; Kt transposed [d, kv] 128x64
using SmemLayoutK  = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutKt = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{}));
// dS produced tile [q, kv] 128x64
using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomKV{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));

// canonical 128x128 SF atom (512B) — full-atom loads for K/V/Kt SF
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

// gmem SF tiled layouts (head mode L): K/V natural (keys, d) SFA; Kt (d, keys) SFB.
using LayoutSFK = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(0, int(kBlockN), int(kHeadDim), 0)));
using LayoutSFKt = decltype(BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), 0, 0)));

using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(0, int(kHeadDim), 0), make_stride(int(kHeadDim), _1{}, int(kHeadDim))),
    SmemLayoutK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{}));
using TMA_Kt = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kHeadDim), 0, 0), make_stride(0, _1{}, 0)),
    SmemLayoutKt{}(_, _, _0{}), make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{}));
using TMA_SFK = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFK{}),
    SmemLayoutSFT{}, make_shape(Int<128>{}, Int<kHeadDim>{}), _1{}));
using TMA_SFKt = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFKt{}),
    SmemLayoutSFT{}, make_shape(Int<kHeadDim>{}, Int<128>{}), _1{}));

using PipeKV  = cutlass::PipelineTmaAsync<kStages>;    // K + V + SFK + SFV
using PipeKt  = cutlass::PipelineTmaAsync<kStages>;    // Kt + SFKt
using StateKV = cutlass::PipelineState<kStages>;
using StateKt = cutlass::PipelineState<kStages>;

constexpr int TmaBytesKV = kBlockN * kHeadDim * 2 + 512 * 2;      // K,V data + SF atoms
constexpr int TmaBytesKt = kHeadDim * kBlockN + 512;

struct ParamsDq {
  TMA_K tma_k; TMA_K tma_v; TMA_Kt tma_kt; TMA_SFK tma_sfk; TMA_SFK tma_sfv; TMA_SFKt tma_sfkt;
  LayoutSFK layout_sfk;      // (S, 64, 128, H)
  LayoutSFKt layout_sfkt;    // (128, 128, S, H)
  const uint8_t *Q, *D;      // resident natural tiles [H, S, 128]
  const uint8_t *sfQ, *sfD;  // flat 512B tiles, tile (h, m) at ((h*MT + m))*512
  const float* lse;          // [H, S]
  const float* delta;        // [H, S]
  float* dQ;                 // [H, S, 128]
  int S, H;
  float sm_scale;
  float* dbg;   // [64] debug
};

struct SharedStorageDq {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sD;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>>  sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>>  sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKt>> sKt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutDS>> sDS;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFQ, sSFD;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFK, sSFV, sSFKt;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFDS;
  alignas(8) typename PipeKV::SharedStorage pipeline_kv;
  alignas(8) typename PipeKt::SharedStorage pipeline_kt;
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
dq_ws_kernel(CUTE_GRID_CONSTANT ParamsDq const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDq*>(smem_raw);
  int const m = blockIdx.x, h = blockIdx.y;
  int const MT = p.S / kBlockM;              // q tiles = kv 128-atoms... kv 64-tiles: S/64
  int const NT = p.S / kBlockN;

  int const wg = cutlass::canonical_warp_group_idx();
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  typename PipeKV::Params pkv;
  pkv.role = (wg == 0) ? PipeKV::ThreadCategory::Producer : PipeKV::ThreadCategory::Consumer;
  pkv.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pkv.num_consumers = NumMmaThreads;
  pkv.transaction_bytes = TmaBytesKV;
  PipeKV pipeline_kv(ss.pipeline_kv, pkv, Shape<_1, _1, _1>{});
  typename PipeKt::Params pkt;
  pkt.role = pkv.role; pkt.is_leader = pkv.is_leader; pkt.num_consumers = NumMmaThreads;
  pkt.transaction_bytes = TmaBytesKt;
  PipeKt pipeline_kt(ss.pipeline_kt, pkt, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQ{});
  Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutK{});
  Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutK{});
  Tensor sKt = make_tensor(make_smem_ptr(ss.sKt.begin()), SmemLayoutKt{});
  Tensor sDS = make_tensor(make_smem_ptr(ss.sDS.begin()), SmemLayoutDS{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFT{});
  Tensor sSFD = make_tensor(make_smem_ptr(ss.sSFD.begin()), SmemLayoutSFT{});


  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mK3d = p.tma_k.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mV3d = p.tma_v.get_tma_tensor(make_shape(int(p.S), int(kHeadDim), int(p.H)));
      Tensor mKt3d = p.tma_kt.get_tma_tensor(make_shape(int(kHeadDim), int(p.S), int(p.H)));
      Tensor mSFK3d = p.tma_sfk.get_tma_tensor(shape(p.layout_sfk));
      Tensor mSFKt3d = p.tma_sfkt.get_tma_tensor(shape(p.layout_sfkt));
      auto bk = p.tma_k.get_slice(_0{}); auto bv = p.tma_v.get_slice(_0{}); auto bkt = p.tma_kt.get_slice(_0{});
      auto bsk = p.tma_sfk.get_slice(_0{}); auto bskt = p.tma_sfkt.get_slice(_0{});
      Tensor mK = mK3d(_, _, h); Tensor mV = mV3d(_, _, h); Tensor mKt = mKt3d(_, _, h);
      Tensor mSFK = mSFK3d(_, _, h); Tensor mSFKt = mSFKt3d(_, _, h);
      Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}));    // (N,K,nb)
      Tensor gV = local_tile(mV, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}));
      Tensor gKt = local_tile(mKt, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _));  // (d,N,nb)
      Tensor gSFK = local_tile(mSFK, make_shape(Int<128>{}, Int<kHeadDim>{}), make_coord(_, _0{}));    // (128-key atom,hd,nb/2)
      Tensor gSFKt = local_tile(mSFKt, make_shape(Int<kHeadDim>{}, Int<128>{}), make_coord(_0{}, _));
      Tensor tKgK = group_modes<0, 3>(bk.partition_S(gK));
      Tensor tKsK = group_modes<0, 3>(bk.partition_D(sK));
      Tensor tKgV = group_modes<0, 3>(bv.partition_S(gV));
      Tensor tKsV = group_modes<0, 3>(bv.partition_D(sV));
      Tensor tKtgKt = group_modes<0, 3>(bkt.partition_S(gKt));
      Tensor tKtsKt = group_modes<0, 3>(bkt.partition_D(sKt));
      Tensor tKgSFK = group_modes<0, 3>(bsk.partition_S(gSFK));
      Tensor tKsSFK = group_modes<0, 3>(bsk.partition_D(
          make_tensor(make_smem_ptr(ss.sSFK.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      Tensor tKsSFV = group_modes<0, 3>(bsk.partition_D(
          make_tensor(make_smem_ptr(ss.sSFV.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      Tensor tKtgSFKt = group_modes<0, 3>(bskt.partition_S(gSFKt));
      Tensor tKtsSFKt = group_modes<0, 3>(bskt.partition_D(
          make_tensor(make_smem_ptr(ss.sSFKt.begin()),
                      make_layout(append(shape(SmemLayoutSFT{}), Int<kStages>{}),
                                  append(stride(SmemLayoutSFT{}), Int<512>{})))));
      StateKV wkv; StateKt wkt;
      auto sKV = cutlass::make_producer_start_state<PipeKV>();
      auto sKt_ = cutlass::make_producer_start_state<PipeKt>();
      for (int n = 0; n < NT; ++n) {
        int const sfatom = n / 2;
        pipeline_kv.producer_acquire(sKV);
        copy(p.tma_k.with(*pipeline_kv.producer_get_barrier(sKV), 0), tKgK(_, n), tKsK(_, sKV.index()));
        copy(p.tma_v.with(*pipeline_kv.producer_get_barrier(sKV), 0), tKgV(_, n), tKsV(_, sKV.index()));
        copy(p.tma_sfk.with(*pipeline_kv.producer_get_barrier(sKV), 0), tKgSFK(_, sfatom), tKsSFK(_, sKV.index()));
        copy(p.tma_sfv.with(*pipeline_kv.producer_get_barrier(sKV), 0), tKgSFK(_, sfatom), tKsSFV(_, sKV.index()));
        ++sKV;
        pipeline_kt.producer_acquire(sKt_);
        copy(p.tma_kt.with(*pipeline_kt.producer_get_barrier(sKt_), 0), tKtgKt(_, n), tKtsKt(_, sKt_.index()));
        copy(p.tma_sfkt.with(*pipeline_kt.producer_get_barrier(sKt_), 0), tKtgSFKt(_, sfatom), tKtsSFKt(_, sKt_.index()));
        ++sKt_;
      }
      (void)wkv; (void)wkt;
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;
    int const warp = tid / 32, lane = tid % 32;

    // resident Q, dO: cooperative vectorized load by all consumer threads
    {
      auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                                Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
      auto tgc = gc.get_thread_slice(tid);
      auto nat = make_layout(make_shape(kBlockM, kHeadDim), make_stride(kHeadDim, _1{}));
      Tensor gQm = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.Q) + (size_t(h) * p.S + size_t(m) * kBlockM) * kHeadDim), nat);
      Tensor gDm = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.D) + (size_t(h) * p.S + size_t(m) * kBlockM) * kHeadDim), nat);
      copy(gc, tgc.partition_S(gQm), tgc.partition_D(sQ));
      copy(gc, tgc.partition_S(gDm), tgc.partition_D(sD));
      for (int i = tid; i < 128; i += NumMmaThreads) {
        reinterpret_cast<uint32_t*>(ss.sSFQ.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfQ + (size_t(h) * MT + m) * 512)[i];
        reinterpret_cast<uint32_t*>(ss.sSFD.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfD + (size_t(h) * MT + m) * 512)[i];
      }
    }
    cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier + 1).sync();

    TiledMmaK128 mma128; TiledMmaK64 mma64;
    auto thr128 = mma128.get_thread_slice(tid);
    auto thr64 = mma64.get_thread_slice(tid);
    auto ts128 = tile_shape(mma128); auto ts64 = tile_shape(mma64);

    // S/dP operand fragments (K=d=128)
    Tensor tSrQ  = thr128.partition_fragment_A(sQ);
    Tensor tSrD  = thr128.partition_fragment_A(sD);
    Tensor tSrK  = thr128.partition_fragment_B(sK(_, _, _0{}));
    Tensor tSrV  = thr128.partition_fragment_B(sV(_, _, _0{}));
    Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr128);
    Tensor tSrSFD = mxfp8::partition_fragment_SFA(sSFD, thr128);
    Tensor tSrSFK = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFT{}), thr128);
    Tensor tSrSFV = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFT{}), thr128);
    // dQ operand fragments (K=kv=64): A = dS [q,kv], B = Kt [d,kv]
    Tensor tOrDS = thr64.partition_fragment_A(sDS);
    Tensor tOrKt = thr64.partition_fragment_B(sKt(_, _, _0{}));
    Tensor tOrSFDS = mxfp8::partition_fragment_SFA(
        make_tensor(make_smem_ptr(ss.sSFDS.begin()), SmemLayoutSFK64{}), thr64);
    Tensor tOrSFKt = mxfp8::partition_fragment_SFB(
        make_tensor(make_smem_ptr(ss.sSFKt.begin()), SmemLayoutSFK64{}), thr64);
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

    // resident Q/dO fragments + SF: load once
    copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sQ)), tscA128.retile_D(tSrQ));
    copy(scA128, tscA128.partition_S(as_position_independent_swizzle_tensor(sD)), tscA128.retile_D(tSrD));
    copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFQ)), tscSFA128.retile_D(tSrSFQ));
    copy(scSFA128, tscSFA128.partition_S(as_position_independent_swizzle_tensor(sSFD)), tscSFA128.retile_D(tSrSFD));

    // S9: static SF half via even/odd step pairs
    auto subSFK = [](auto const& f, auto hc) {
      auto m1 = get<1>(f.layout()); auto a = get<0>(m1); auto b = get<1>(m1);
      auto nb = shape(b); auto sb = stride(b);
      auto t = make_tensor(f.data(), make_layout(get<0>(f.layout()),
          make_layout(make_shape(shape(a), make_shape(nb / _2{}, _2{})),
                      make_stride(stride(a), make_stride(sb, sb * (nb / _2{})))),
          get<2>(f.layout())))(_, make_coord(_, make_coord(_, hc)), _);
      return group_modes<1, 3>(t);
    };

    Tensor accQ  = partition_fragment_C(mma64, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(accQ);
    Tensor accS  = partition_fragment_C(mma128, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor accDP = partition_fragment_C(mma128, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto rc_view = [](auto& f) {
      return make_tensor(f.data(), make_layout(
          make_layout(get<0, 1>(f.layout()), get<1>(f.layout())),
          make_layout(get<0, 0>(f.layout()), get<2>(f.layout()))));
    };
    Tensor accS_rc = rc_view(accS); Tensor accDP_rc = rc_view(accDP);
    constexpr int kNRow = 2, kNCol = kBlockN / 4;   // 16
    int const row0 = warp * 16 + lane / 4;
    int const col0 = (lane % 4) * 2;

    float lse_r[kNRow], dlt_r[kNRow];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kNRow; ++mi) {
      int q = m * kBlockM + row0 + mi * 8;
      lse_r[mi] = p.lse[size_t(h) * p.S + q];
      dlt_r[mi] = p.delta[size_t(h) * p.S + q];
    }

    StateKV rkv; StateKt rkt;
    auto step = [&](int n, auto hc) {
      // ---- S = Q K^T ; dP = dO V^T ----
      { auto t = pipeline_kv.consumer_try_wait(rkv); pipeline_kv.consumer_wait(rkv, t);
        int stage = rkv.index();
        Tensor sSFKst = make_tensor(make_smem_ptr(ss.sSFK.begin() + stage * 512), SmemLayoutSFT{});
        Tensor sSFVst = make_tensor(make_smem_ptr(ss.sSFV.begin() + stage * 512), SmemLayoutSFT{});
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sK(_, _, stage))), tscB128.retile_D(tSrK));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFKst)), tscSFB128.retile_D(tSrSFK));
        copy(scB128, tscB128.partition_S(as_position_independent_swizzle_tensor(sV(_, _, stage))), tscB128.retile_D(tSrV));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFVst)), tscSFB128.retile_D(tSrSFV));
      }
      auto tSrSFK_h = subSFK(tSrSFK, hc);
      auto tSrSFV_h = subSFK(tSrSFV, hc);
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrQ); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                   make_zip_tensor(tSrK(_, _, k), tSrSFK_h(_, _, k)), accS);
      if (p.dbg && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        p.dbg[0] = accS_rc(0, 0); p.dbg[1] = accS_rc(1, 5);
        p.dbg[2] = float(Element::bitcast(tSrQ(0))); p.dbg[3] = float(Element::bitcast(tSrK(0)));
        p.dbg[4] = float(uint8_t(tSrSFQ(0).storage)); p.dbg[5] = float(uint8_t(tSrSFK_h(0).storage));
        p.dbg[6] = lse_r[0]; p.dbg[7] = dlt_r[0];
      }
      clear(accDP);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrD); ++k)
        cute::gemm(mma128, make_zip_tensor(tSrD(_, _, k), tSrSFD(_, _, k)),
                   make_zip_tensor(tSrV(_, _, k), tSrSFV_h(_, _, k)), accDP);
      pipeline_kv.consumer_release(rkv); ++rkv;

      // ---- dS quantize per-32-along-kv -> sDS [q, kv] + sSFDS ----
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // prev n's readers done
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int q = row0 + mi * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < kBlockN / SFVecSize; ++kb) {
          float as = 0.f;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            float pv = exp2f((accS_rc(mi, ni) * p.sm_scale - lse_r[mi]) * kLog2e);
            float dsv = pv * (accDP_rc(mi, ni) - dlt_r[mi]);
            accDP_rc(mi, ni) = dsv;
            as = fmaxf(as, fabsf(dsv));
          }
          as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 1));
          as = fmaxf(as, __shfl_xor_sync(uint32_t(-1), as, 2));
          int ses = mx_scale_exp(as);
          if ((lane % 4) == 0)
            ss.sSFDS.begin()[16 * (q % 32) + 4 * (q / 32) + kb] = ElementSF::bitcast(uint8_t(ses + 127));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) {
            int ni = kb * 8 + j;
            int c = (ni / 2) * 8 + col0 + (ni % 2);
            sDS(q, c) = quant_e4m3(accDP_rc(mi, ni), ses);
          }
        }
      }
      if (p.dbg && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        p.dbg[8] = accDP_rc(0, 0); p.dbg[9] = float(Element::bitcast(sDS(0, 0).storage));
        p.dbg[10] = float(Element::bitcast(sDS(row0, col0).storage));
        p.dbg[11] = float(uint8_t(ss.sSFDS.begin()[0].storage));
      }
      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // DS/SF visible

      // ---- dQ += dS Kt ----
      { auto t = pipeline_kt.consumer_try_wait(rkt); pipeline_kt.consumer_wait(rkt, t);
        int stage = rkt.index();
        constexpr int hh = decltype(hc)::value;
        const uint8_t* sfKt_base = reinterpret_cast<const uint8_t*>(ss.sSFKt.begin()) + stage * 512 + 2 * hh;
        copy(scA64, tscA64.partition_S(as_position_independent_swizzle_tensor(sDS)), tscA64.retile_D(tOrDS));
        copy(scB64, tscB64.partition_S(as_position_independent_swizzle_tensor(sKt(_, _, stage))), tscB64.retile_D(tOrKt));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrDS); ++k) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFDS(_, _, k)); ++i) {
            auto c = sfpA_coord(_, _, k)(i);
            int q = int(get<0>(c)), kv = int(get<1>(c));
            tOrSFDS(_, _, k)(i) = ElementSF::bitcast(ss.sSFDS.begin()[16 * (q % 32) + 4 * (q / 32) + kv / 32].storage);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFKt(_, _, k)); ++i) {
            auto c = sfpB_coord(_, _, k)(i);
            int d = int(get<0>(c)), kv = int(get<1>(c));
            tOrSFKt(_, _, k)(i) = ElementSF::bitcast(sfKt_base[16 * (d % 32) + 4 * (d / 32) + kv / 32]);
          }
        }
        pipeline_kt.consumer_release(rkt); ++rkt; }
      if (p.dbg && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        p.dbg[22] = float(uint8_t(tOrSFDS(0).storage));
      if (p.dbg && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        auto c0 = sfpA_coord(_, _, _0{})(0);
        p.dbg[17] = float(int(get<0>(c0))); p.dbg[18] = float(int(get<1>(c0)));
        p.dbg[19] = float(size(tOrSFDS)); p.dbg[20] = float(size(sfpA_coord));
        p.dbg[21] = float(uint8_t(ss.sSFDS.begin()[0].storage));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrDS); ++k)
        cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, k), tOrSFDS(_, _, k)),
                   make_zip_tensor(tOrKt(_, _, k), tOrSFKt(_, _, k)), accQ);
    };
    for (int n = 0; n < NT; n += 2) {
      step(n, cute::Int<0>{});
      if (n + 1 < NT) step(n + 1, cute::Int<1>{});
    }

    if (p.dbg && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
      p.dbg[12] = accQ(0); p.dbg[13] = float(Element::bitcast(tOrDS(0))); p.dbg[14] = float(Element::bitcast(tOrKt(0)));
      p.dbg[15] = float(uint8_t(tOrSFDS(0).storage)); p.dbg[16] = float(uint8_t(tOrSFKt(0).storage));
    }
    // ---- epilogue: dQ[h, m-tile] ----
    Tensor gO = make_tensor(make_gmem_ptr(p.dQ + (size_t(h) * p.S + size_t(m) * kBlockM) * kHeadDim),
                            make_layout(make_shape(kBlockM, kHeadDim), make_stride(kHeadDim, _1{})));
    copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accQ, thr64.partition_C(gO));
  }
}

}  // namespace s3bws
