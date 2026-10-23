// Launchers for the S3M masked MXFP8 attention kernels (block-causal via per-row
// key limits). New TU; forks mxfp8_ragged_mx_kernel.cu (fwd) and the mode-1
// bwd launchers (mxfp8_bwd_kernel.cu). All buffers are owned by the caller.
//
// The fwd launcher takes the work plan directly (built host-side per forward);
// row_kv_limit is the per-row key prefix: row r attends keys [0, limit[r]).
// bwd launchers take the same row_kv_limit (+ L) and mask the recomputed P/dS.
#include <cstdint>
#include <cuda_runtime.h>
#include "../s3m_kernel.cuh"
#include "../s3m_dq_ws_kernel.cuh"
#include "../s3m_dk_ws_kernel.cuh"
#include "../s3m_dv_ws_kernel.cuh"

using namespace flashinfer;
using cute::Int;
using cute::make_coord;
using cute::make_shape;
using cute::make_stride;

// ---------------------------------------------------------------------------
// fwd: S3M ragged prefill with per-row key limits (plan built by the caller).
// ---------------------------------------------------------------------------
extern "C" void s3m_ragged_mx_launch(
    const void* Qd, const void* Kd, const void* Vt,   // e4m3 uint8: [Sq,Hq,D] [Sk,Hkv,D] [Hkv,D,Sk]
    const void* sfQ, const void* sfK, const void* sfV, // packed ue8m0
    int Sq_pad, int Sk_pad, int Hq, int Hkv, int group,
    float sm_scale,
    void* out_O_, float* out_lse, float* out_l,   // out_O_: bf16 [Sq,Hq,D]
    int* work_indptr, int* head_indices, int* qo_tile_indices,
    int* qo_indptr, int* kv_indptr, int* qo_lens, int* kv_lens, int* batch_indices,
    const int* row_kv_limit,   // [Sq_pad] per-row key limit (required)
    int num_sm, uintptr_t stream_) {
  const int HD = kHeadDim;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

  Element* dQ = reinterpret_cast<Element*>(const_cast<void*>(Qd));
  Element* dK = reinterpret_cast<Element*>(const_cast<void*>(Kd));
  Element* dV = reinterpret_cast<Element*>(const_cast<void*>(Vt));
  ElementSF* dSFQ = reinterpret_cast<ElementSF*>(const_cast<void*>(sfQ));
  ElementSF* dSFK = reinterpret_cast<ElementSF*>(const_cast<void*>(sfK));
  ElementSF* dSFV = reinterpret_cast<ElementSF*>(const_cast<void*>(sfV));

  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, Hq));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, Hkv));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, Hkv));
  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(Sq_pad, HD, Hq), make_stride(Hq * HD, _1{}, HD));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Sk_pad, HD, Hkv), make_stride(Hkv * HD, _1{}, HD));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, Sk_pad, Hkv), make_stride(Sk_pad, _1{}, HD * Sk_pad));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSFV), layoutSFV);

  ParamsM p;
  p.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  p.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  p.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  p.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{});
  p.layout_sfq = layoutSFQ; p.layout_sfv = layoutSFV;
  p.seqlen_q = Sq_pad; p.seqlen_k = Sk_pad; p.n_block_total = Sk_pad / kBlockN;
  p.sm_scale = sm_scale; p.o_scale = 1.0f;
  p.num_qo_heads = Hq; p.num_kv_heads = Hkv; p.tile_kv_len = nullptr;
  p.row_kv_limit = row_kv_limit;
  p.out_O = reinterpret_cast<cutlass::bfloat16_t*>(out_O_); p.out_lse = out_lse; p.out_l = out_l;
  p.out_Ppre = nullptr; p.out_Mnb = nullptr; p.out_dbg = nullptr;

  using Sched = BatchPrefillPersistentTileScheduler<int>;
  Sched::Arguments sa;
  sa.work_indptr = work_indptr; sa.head_indices = head_indices; sa.qo_tile_indices = qo_tile_indices;
  sa.qo_indptr = qo_indptr; sa.kv_indptr = kv_indptr; sa.qo_lens = qo_lens; sa.kv_lens = kv_lens;
  sa.batch_indices = batch_indices; sa.group_size_fastdiv = cutlass::FastDivmod(group);
  sa.num_qo_heads = Hq;
  dim3 grid = Sched::get_grid_dim(sa, num_sm);
  typename Sched::Params sp = Sched::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorageM));

  // row-limit mode is always non-causal (the limit array carries the structure)
  cudaFuncSetAttribute(s3m_kernel<Sched, false, SFSource::kMxFp8>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  s3m_kernel<Sched, false, SFSource::kMxFp8><<<grid, kNThreads, smem, stream>>>(p, sp);
}

// ---------------------------------------------------------------------------
// bwd: masked mode-1 kernels (dv_ws + dk_ws + dq_ws), same buffer contracts as
// mxfp8_bwd_kernel.cu. L = real q rows; row_kv_limit as above.
// ---------------------------------------------------------------------------
extern "C" void mxfp8_dv_ws_bc_launch(
    const void* Kd, const void* Qd, const void* Dt,
    const void* sfK, const void* sfQ, const void* sfDt,
    const float* lse, float* dV, int S, int H, int L,
    float sm_scale, const int* row_kv_limit, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3mdvws::dv_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3mdvws::SharedStorageDv)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3mdvws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3mdvws::kBlockM), D, H));
  auto layoutSFDt = s3mdvws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3mdvws::kBlockN), D, S, H));
  s3mdvws::ParamsDv p{};
  cute::Tensor mQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdvws::Element const*>(Qd)),
      cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  cute::Tensor mDt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdvws::Element const*>(Dt)),
      cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  cute::Tensor mSFQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdvws::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFDt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdvws::ElementSF const*>(sfDt)), layoutSFDt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3mdvws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3mdvws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3mdvws::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3mdvws::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3mdvws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3mdvws::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.layout_sfq = layoutSFQ;
  p.layout_sfdt = layoutSFDt;
  p.K = (const uint8_t*)Kd; p.sfK = (const uint8_t*)sfK;
  p.lse = lse; p.dV = dV;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
  p.row_kv_limit = row_kv_limit; p.L = L;
  dim3 grid(S / s3mdvws::kBlockN, H);
  s3mdvws::dv_ws_kernel<<<grid, s3mdvws::kNThreads, int(sizeof(s3mdvws::SharedStorageDv)), stream>>>(p);
}

extern "C" void mxfp8_dk_ws_bc_launch(
    const void* Kd, const void* Vd,
    const void* Qd, const void* Dd, const void* Qt,
    const void* sfK, const void* sfV,
    const void* sfQ, const void* sfD, const void* sfQt,
    const float* lse, const float* delta,
    float* dK, int S, int H, int L,
    float sm_scale, const int* row_kv_limit, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3mdkws::dk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3mdkws::SharedStorageDk)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3mdkws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3mdkws::kBlockM), D, H));
  auto layoutSFQt = s3mdkws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3mdkws::kBlockN), D, S, H));
  s3mdkws::ParamsDk p{};
  cute::Tensor mQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::Element const*>(Qd)),
      cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  cute::Tensor mD = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::Element const*>(Dd)), mQ.layout());
  cute::Tensor mQt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::Element const*>(Qt)),
      cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  cute::Tensor mSFQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFD = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::ElementSF const*>(sfD)), layoutSFQ);
  cute::Tensor mSFQt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdkws::ElementSF const*>(sfQt)), layoutSFQt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3mdkws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3mdkws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_d = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mD, s3mdkws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3mdkws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, s3mdkws::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3mdkws::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3mdkws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfd = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFD, s3mdkws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfqt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQt, s3mdkws::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  cute::Tensor mLse = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const uint8_t*>(lse)),
      cute::make_layout(cute::make_shape(S * H * 4), cute::make_stride(cute::_1{})));
  cute::Tensor mDlt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const uint8_t*>(delta)),
      cute::make_layout(cute::make_shape(S * H * 4), cute::make_stride(cute::_1{})));
  p.tma_lse = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mLse, cute::Layout<cute::Shape<cute::_256>>{},
                                  cute::make_shape(cute::Int<256>{}), cute::_1{});
  p.tma_dlt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDlt, cute::Layout<cute::Shape<cute::_256>>{},
                                  cute::make_shape(cute::Int<256>{}), cute::_1{});
  p.layout_sfq = layoutSFQ;
  p.layout_sfqt = layoutSFQt;
  p.K = (const uint8_t*)Kd; p.V = (const uint8_t*)Vd;
  p.sfK = (const uint8_t*)sfK; p.sfV = (const uint8_t*)sfV;
  p.dK = dK;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
  p.row_kv_limit = row_kv_limit; p.L = L;
  dim3 grid(S / s3mdkws::kBlockN, H);
  s3mdkws::dk_ws_kernel<<<grid, s3mdkws::kNThreads, int(sizeof(s3mdkws::SharedStorageDk)), stream>>>(p);
}

extern "C" void mxfp8_dq_ws_bc_launch(
    const void* Kd, const void* Vd, const void* Kt,
    const void* Qd, const void* Dd,
    const void* sfK, const void* sfV, const void* sfKt,
    const void* sfQ, const void* sfD,
    const float* lse, const float* delta,
    float* dQ, int S, int H, int L,
    float sm_scale, const int* row_kv_limit, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3mdqws::dq_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3mdqws::SharedStorageDq)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFK = s3mdqws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3mdqws::kBlockN), D, H));
  auto layoutSFKt = s3mdqws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3mdqws::kBlockM), D, S, H));
  s3mdqws::ParamsDq p{};
  cute::Tensor mK = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::Element const*>(Kd)),
      cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  cute::Tensor mV = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::Element const*>(Vd)), mK.layout());
  cute::Tensor mKt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::Element const*>(Kt)),
      cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  cute::Tensor mSFK = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::ElementSF const*>(sfK)), layoutSFK);
  cute::Tensor mSFV = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::ElementSF const*>(sfV)), layoutSFK);
  cute::Tensor mSFKt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3mdqws::ElementSF const*>(sfKt)), layoutSFKt);
  p.tma_k = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mK, s3mdqws::SmemLayoutK{}(_, _, cute::_0{}),
                                make_shape(Int<s3mdqws::kBlockN>{}, Int<D>{}), cute::_1{});
  p.tma_v = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mV, s3mdqws::SmemLayoutK{}(_, _, cute::_0{}),
                                make_shape(Int<s3mdqws::kBlockN>{}, Int<D>{}), cute::_1{});
  p.tma_kt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mKt, s3mdqws::SmemLayoutKt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3mdqws::kBlockN>{}), cute::_1{});
  p.tma_sfk = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFK, s3mdqws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfv = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFV, s3mdqws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfkt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFKt, s3mdqws::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.layout_sfk = layoutSFK;
  p.layout_sfkt = layoutSFKt;
  p.Q = (const uint8_t*)Qd; p.D = (const uint8_t*)Dd;
  p.sfQ = (const uint8_t*)sfQ; p.sfD = (const uint8_t*)sfD;
  p.lse = lse; p.delta = delta; p.dQ = dQ;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
  p.row_kv_limit = row_kv_limit; p.L = L;
  p.dbg = nullptr;
  dim3 grid(S / s3mdqws::kBlockM, H);
  s3mdqws::dq_ws_kernel<<<grid, s3mdqws::kNThreads, int(sizeof(s3mdqws::SharedStorageDq)), stream>>>(p);
}
