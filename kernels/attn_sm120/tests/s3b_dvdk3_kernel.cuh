// S3B-DVDK3: transient/acc warp-group split prototype (c21).
// 512 threads: WG0/1 = transient (inline TMA + S'/dP gemm + quant + STS),
//              WG2/3 = acc       (LDS P'/dS' from staging + output gemms
//                                 into accV/accK + epilogue STG).
// Hypothesis under test: quant(m) on transient WGs overlaps output-gemm(m-1)
// on acc WGs, filling the tensor bubble that the lockstep monolith (dvdk2)
// cannot. P'/dS' cross the WG boundary through an 8KB smem staging buffer
// (single-buffered in 2 chunks of [128,32] = one output k-atom each) since
// sm_120 has no TMEM; ses exponents ride along in a 256B sideband.
// Register split: 256*104 (transient) + 256*152 (acc) = 65536.

#pragma once

#include "s3b_dvdk2_kernel.cuh"

namespace s3bdvdk3 {

using namespace s3bdvdk2;

constexpr int kRegTransient = 104, kRegAcc = 152;
static_assert(256 * (kRegTransient + kRegAcc) <= 65536);

// Named barrier ids (bar 0 reserved for __syncthreads in pipeline ctors).
constexpr int kBarFull0 = 1, kBarEmpty0 = 2, kBarFull1 = 3, kBarEmpty1 = 4;
constexpr int kBarInitT = 6;   // transient-only init sync (256 thr)

// staging uses a flat [128,32] fp8 layout per chunk (elementwise STS/LDS,
// no LDSM) — see wrap64 lambdas below.
struct SharedStorageDvdk3 {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutKV>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sD;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt1>> sQt;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQt1>> sDt;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFK, sSFV;
  alignas(128) cute::ArrayEngine<ElementSF, 512 * kStages> sSFQ, sSFD;
  alignas(128) cute::ArrayEngine<ElementSF, 512> sSFQt, sSFDt;
  // cross-WG staging: one chunk = one output k-atom = [128,32] fp8 per tensor.
  alignas(1024) cute::ArrayEngine<Element, kBlockN * 32> sPS;
  alignas(1024) cute::ArrayEngine<Element, kBlockN * 32> sDS;
  alignas(16) uint8_t sSesDS[2][128];   // per-kb, per-kv-row ue8m0 (ses+127)
  alignas(8) typename PipeQD::SharedStorage pipeline_qd;
  alignas(8) typename PipeTT::SharedStorage pipeline_tt;
};

__global__ void __launch_bounds__(512, 1)
dvdk3_kernel(CUTE_GRID_CONSTANT ParamsDvdk const p) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorageDvdk3*>(smem_raw);
  int const n = blockIdx.x, h = blockIdx.y;
  int const NT128 = p.S / kBlockN;
  int const MT = p.S / kBlockM;

  int const wg = cutlass::canonical_warp_group_idx();
  int const elect = cute::elect_one_sync();
  bool const is_transient = wg < 2;
  bool const is_producer = (threadIdx.x / 32 == 0);   // transient warp0, whole warp

  typename PipeQD::Params pqd;
  pqd.role = is_producer ? PipeQD::ThreadCategory::ProducerConsumer
                         : PipeQD::ThreadCategory::Consumer;
  pqd.is_leader = (threadIdx.x == 0);
  pqd.num_consumers = 256;   // transient WGs only
  pqd.transaction_bytes = TmaBytesQD;
  typename PipeTT::Params ptt;
  ptt.role = is_producer ? PipeTT::ThreadCategory::ProducerConsumer
                         : PipeTT::ThreadCategory::Consumer;
  ptt.is_leader = pqd.is_leader;
  ptt.num_consumers = 256;   // acc WGs only
  ptt.transaction_bytes = TmaBytesTT;
  PipeQD pipeline_qd(ss.pipeline_qd, pqd, Shape<_1, _1, _1>{});
  PipeTT pipeline_tt(ss.pipeline_tt, ptt, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ  = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sD  = make_tensor(make_smem_ptr(ss.sD.begin()), SmemLayoutQ{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.sQt.begin()), SmemLayoutQt1{});
  Tensor sDt = make_tensor(make_smem_ptr(ss.sDt.begin()), SmemLayoutQt1{});

  // TMA partition setup (identical to dvdk2; evaluated per-role below).
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

  if (is_transient) {
    // ---------------- transient WGs: input gemms + quant + STS ----------------
    cutlass::arch::warpgroup_reg_dealloc<kRegTransient>();
    int const tid = threadIdx.x;   // 0..255
    int const lane = tid % 32;

    Tensor sK  = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutKV{});
    Tensor sV  = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutKV{});
    Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFT{});
    Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFT{});
    {
      auto gc = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
                                Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{});
      auto tgc = gc.get_thread_slice(tid);
      auto nat = make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{}));
      Tensor gKn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.K) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      Tensor gVn = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(p.V) + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim), nat);
      copy(gc, tgc.partition_S(gKn), tgc.partition_D(sK));
      copy(gc, tgc.partition_S(gVn), tgc.partition_D(sV));
      for (int i = tid; i < 128; i += 256) {
        reinterpret_cast<uint32_t*>(ss.sSFK.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfK + (size_t(h) * NT128 + n) * 512)[i];
        reinterpret_cast<uint32_t*>(ss.sSFV.begin())[i] = reinterpret_cast<const uint32_t*>(p.sfV + (size_t(h) * NT128 + n) * 512)[i];
      }
    }
    cutlass::arch::NamedBarrier::sync(256, kBarInitT);

    auto issue = make_issue();

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

    auto scA128 = make_tiled_copy_A(SmemCopyAtomData{}, mma128); auto tscA128 = scA128.get_thread_slice(tid);
    auto scB128 = make_tiled_copy_B(SmemCopyAtomData{}, mma128); auto tscB128 = scB128.get_thread_slice(tid);
    auto scSFA128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma128), make_shape(size<0>(ts128), size<2>(ts128)));
    auto scSFB128 = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma128), make_shape(size<1>(ts128), size<2>(ts128)));
    auto tscSFA128 = scSFA128.get_thread_slice(tid); auto tscSFB128 = scSFB128.get_thread_slice(tid);

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

    // staging views: RAW per-thread dump — thread t's k-atom A-frag (16B) at
    // buf + t*16. Same-tid write/read round-trips exactly (validated by
    // s3b_stage_probe: cute elementwise copy on these nested fp8 frag layouts
    // mis-vectorizes; raw uint4 dump is exact and conflict-free).
    StateQD rqd;
    if (is_producer) { issue(0, true, false); if (MT > 1) issue(1, true, false); }
    int const warp_id = tid / 32;

    for (int m = 0; m < MT; ++m) {
      if (is_producer) issue(m, false, true);   // TT(m) at step top (1-stage ring)
      int stage_qd;
      { auto t = pipeline_qd.consumer_try_wait(rqd); pipeline_qd.consumer_wait(rqd, t);
        stage_qd = rqd.index();
        Tensor sSFQst = make_tensor(make_smem_ptr(ss.sSFQ.begin() + stage_qd * 512), SmemLayoutSFT{});
        Tensor sSFDst = make_tensor(make_smem_ptr(ss.sSFD.begin() + stage_qd * 512), SmemLayoutSFT{});
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFQst)), tscSFB128.retile_D(tSrSFQ));
        copy(scSFB128, tscSFB128.partition_S(as_position_independent_swizzle_tensor(sSFDst)), tscSFB128.retile_D(tSrSFD));
      }
      auto tSrSFQ_h = subSF(tSrSFQ, m & 1);
      auto tSrSFD_h = subSF(tSrSFD, m & 1);
      auto tscK = tscA128.partition_S(as_position_independent_swizzle_tensor(sK));
      auto tscV = tscA128.partition_S(as_position_independent_swizzle_tensor(sV));
      auto tscQ = tscB128.partition_S(as_position_independent_swizzle_tensor(sQ(_, _, stage_qd)));
      auto tscD = tscB128.partition_S(as_position_independent_swizzle_tensor(sD(_, _, stage_qd)));
      auto tcrK = tscA128.retile_D(tSrK); auto tcrV = tscA128.retile_D(tSrV);
      auto tcrQ = tscB128.retile_D(tSrQ); auto tcrD = tscB128.retile_D(tSrD);
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
      pipeline_qd.consumer_release(rqd); ++rqd;
      if (is_producer && m + kStages < MT) issue(m + kStages, true, false);

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

      // hand off chunk kb = output k-atom kb; ses sideband rides per chunk.
      // SINGLE buffer: STS c1 must wait acc's LDS of c0 -> EmptyA/EmptyB chain:
      //   transient: sync EmptyA -> STS c0 -> Full0 -> sync EmptyB -> STS c1 -> Full1
      //   acc:       Full0 -> LDS c0 -> arrive EmptyB -> Full1 -> LDS c1 -> arrive EmptyA
      cutlass::arch::NamedBarrier::sync(512, kBarEmpty0);
      *reinterpret_cast<uint4*>(ss.sPS.begin() + tid * 16) =
          *reinterpret_cast<uint4 const*>(&tOrP(_0{}, _0{}, _0{}));
      *reinterpret_cast<uint4*>(ss.sDS.begin() + tid * 16) =
          *reinterpret_cast<uint4 const*>(&tOrDS(_0{}, _0{}, _0{}));
      if ((lane & 3) == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi)
          ss.sSesDS[0][warp_id * 16 + (lane >> 2) + 8 * mi] = uint8_t(ses_r[mi][0] + 127);
      }
      cutlass::arch::NamedBarrier::arrive(512, kBarFull0);
      cutlass::arch::NamedBarrier::sync(512, kBarEmpty1);
      *reinterpret_cast<uint4*>(ss.sPS.begin() + tid * 16) =
          *reinterpret_cast<uint4 const*>(&tOrP(_0{}, _0{}, _1{}));
      *reinterpret_cast<uint4*>(ss.sDS.begin() + tid * 16) =
          *reinterpret_cast<uint4 const*>(&tOrDS(_0{}, _0{}, _1{}));
      if ((lane & 3) == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi)
          ss.sSesDS[1][warp_id * 16 + (lane >> 2) + 8 * mi] = uint8_t(ses_r[mi][1] + 127);
      }
      cutlass::arch::NamedBarrier::arrive(512, kBarFull1);
    }
    return;
  }

  // ---------------- acc WGs: staging LDS + output gemms + epilogue ----------------
  cutlass::arch::warpgroup_reg_alloc<kRegAcc>();
  int const tid = threadIdx.x - 256;   // 0..255
  int const lane = tid % 32;

  TiledMmaK64 mma64;
  auto thr64 = mma64.get_thread_slice(tid);

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
  Tensor sfpA_coord = mxfp8::partition_SFA(
      make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);
  auto scB64 = make_tiled_copy_B(SmemCopyAtomData{}, mma64); auto tscB64 = scB64.get_thread_slice(tid);

  Tensor accK  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
  Tensor accV  = partition_fragment_C(mma64, Shape<Int<kBlockN>, Int<kHeadDim>>{});
  clear(accK); clear(accV);

  {
    ElementSF const b = ElementSF::bitcast(uint8_t(-8 + 127));
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(tOrSFP); ++k)
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tOrSFP(_, _, k)); ++i) tOrSFP(_, _, k)(i) = b;
  }

  // pre-arm: staging starts empty — arm EmptyA once (EmptyB arms after c0 LDS)
  cutlass::arch::NamedBarrier::arrive(512, kBarEmpty0);

  StateTT rtt;
  for (int m = 0; m < MT; ++m) {
    int stage_tt;
    { auto t = pipeline_tt.consumer_try_wait(rtt); pipeline_tt.consumer_wait(rtt, t);
      stage_tt = rtt.index();
      const uint8_t* sfQt_base = reinterpret_cast<const uint8_t*>(ss.sSFQt.begin()) + stage_tt * 512 + 2 * (m & 1);
      const uint8_t* sfDt_base = reinterpret_cast<const uint8_t*>(ss.sSFDt.begin()) + stage_tt * 512 + 2 * (m & 1);
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
    }
    auto tscQt = tscB64.partition_S(as_position_independent_swizzle_tensor(sQt(_, _, stage_tt)));
    auto tscDt = tscB64.partition_S(as_position_independent_swizzle_tensor(sDt(_, _, stage_tt)));
    auto tcrQt = tscB64.retile_D(tOrQt); auto tcrDt = tscB64.retile_D(tOrDt);

    CUTLASS_PRAGMA_UNROLL
    for (int kb = 0; kb < 2; ++kb) {
      // wait transient's chunk, pull A-frags + ses, then free the buffer:
      // c0 LDS -> arrive EmptyB (unblocks STS c1); c1 LDS -> arrive EmptyA
      // (unblocks next tile's STS c0).
      cutlass::arch::NamedBarrier::sync(512, kb ? kBarFull1 : kBarFull0);
      *reinterpret_cast<uint4*>(&tOrP(_0{}, _0{}, kb)) =
          *reinterpret_cast<uint4 const*>(ss.sPS.begin() + tid * 16);
      *reinterpret_cast<uint4*>(&tOrDS(_0{}, _0{}, kb)) =
          *reinterpret_cast<uint4 const*>(ss.sDS.begin() + tid * 16);
      {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < size(tOrSFDS(_, _, kb)) / 32; ++r) {
          auto c = sfpA_coord(_, _, kb)(32 * r);
          int row = int(get<0>(c));
          ElementSF const b = ElementSF::bitcast(ss.sSesDS[kb][row]);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 32; ++i) tOrSFDS(_, _, kb)(32 * r + i) = b;
        }
      }
      cutlass::arch::NamedBarrier::arrive(512, kb ? kBarEmpty0 : kBarEmpty1);
      copy(scB64, tscDt(_, _, kb), tcrDt(_, _, kb));
      copy(scB64, tscQt(_, _, kb), tcrQt(_, _, kb));
      cute::gemm(mma64, make_zip_tensor(tOrP(_, _, kb), tOrSFP(_, _, kb)),
                 make_zip_tensor(tOrDt(_, _, kb), tOrSFDt(_, _, kb)), accV);
      cute::gemm(mma64, make_zip_tensor(tOrDS(_, _, kb), tOrSFDS(_, _, kb)),
                 make_zip_tensor(tOrQt(_, _, kb), tOrSFQt(_, _, kb)), accK);
    }
    pipeline_tt.consumer_release(rtt); ++rtt;
  }

  Tensor gK = make_tensor(make_gmem_ptr(p.dK + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                          make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
  Tensor gV = make_tensor(make_gmem_ptr(p.dV + (size_t(h) * p.S + size_t(n) * kBlockN) * kHeadDim),
                          make_layout(make_shape(kBlockN, kHeadDim), make_stride(kHeadDim, _1{})));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accK, thr64.partition_C(gK));
  copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accV, thr64.partition_C(gV));
}

}  // namespace s3bdvdk3
