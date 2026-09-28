// Pure-CUDA launchers for the MXFP8 attention backward kernels (dv64/dk64/dq_ws).
// No torch headers in this TU (nvcc EDG rejects some c10 headers); pybind glue is in
// mxfp8_bwd_ext.cpp. All buffers are owned by the caller.
//
// Tensor contracts (all contiguous, S % 128 == 0, head_dim 128):
//   natural data:  [H, S, 128] e4m3 bytes;   SF: [H, S/128, 512] flat tile-atom packing
//   transposed:    [H, 128, S] e4m3 bytes;   SF: [H, S/128, 512] flat packing
//   lse/delta:     fp32 [H, S]
//   dQ/dK/dV out:  fp32 [H, S, 128]
#include <cstdint>
#include <cuda_runtime.h>
#include "../s3b64_kernel.cuh"
#include "../s3b_dq_ws_kernel.cuh"
#include "../s3b_dk_ws_kernel.cuh"
#include "../s3b_dv_ws_kernel.cuh"
#include "../s3b_dvdk_ws_kernel.cuh"
#include "../s3b_dvdk2_kernel.cuh"

using cute::Int;
uint8_t* g_dvdk_dbg = nullptr;
using cute::make_coord;
using cute::make_shape;
using cute::make_stride;

extern "C" void mxfp8_dk_ws_launch(
    const void* Kd, const void* Vd,
    const void* Qd, const void* Dd, const void* Qt,
    const void* sfK, const void* sfV,
    const void* sfQ, const void* sfD, const void* sfQt,
    const float* lse, const float* delta,
    float* dK, int S, int H, float sm_scale, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3bdkws::dk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3bdkws::SharedStorageDk)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3bdkws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdkws::kBlockM), D, H));
  auto layoutSFQt = s3bdkws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdkws::kBlockN), D, S, H));
  s3bdkws::ParamsDk p{};
  cute::Tensor mQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(Qd)),
      cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  cute::Tensor mD = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(Dd)), mQ.layout());
  cute::Tensor mQt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(Qt)),
      cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  cute::Tensor mSFQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFD = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(sfD)), layoutSFQ);
  cute::Tensor mSFQt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(sfQt)), layoutSFQt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdkws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdkws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_d = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mD, s3bdkws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdkws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, s3bdkws::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdkws::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdkws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfd = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFD, s3bdkws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfqt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQt, s3bdkws::SmemLayoutSFT{},
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
  dim3 grid(S / s3bdkws::kBlockN, H);
  s3bdkws::dk_ws_kernel<<<grid, s3bdkws::kNThreads, int(sizeof(s3bdkws::SharedStorageDk)), stream>>>(p);
}

extern "C" void mxfp8_dv_ws_launch(
    const void* Kd, const void* Qd, const void* Dt,
    const void* sfK, const void* sfQ, const void* sfDt,
    const float* lse, float* dV, int S, int H, float sm_scale, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3bdvws::dv_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3bdvws::SharedStorageDv)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3bdvws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvws::kBlockM), D, H));
  auto layoutSFDt = s3bdvws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvws::kBlockN), D, S, H));
  s3bdvws::ParamsDv p{};
  cute::Tensor mQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdvws::Element const*>(Qd)),
      cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  cute::Tensor mDt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdvws::Element const*>(Dt)),
      cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  cute::Tensor mSFQ = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdvws::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFDt = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<s3bdvws::ElementSF const*>(sfDt)), layoutSFDt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdvws::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdvws::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3bdvws::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdvws::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdvws::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3bdvws::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.layout_sfq = layoutSFQ;
  p.layout_sfdt = layoutSFDt;
  p.K = (const uint8_t*)Kd; p.sfK = (const uint8_t*)sfK;
  p.lse = lse; p.dV = dV;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
  dim3 grid(S / s3bdvws::kBlockN, H);
  s3bdvws::dv_ws_kernel<<<grid, s3bdvws::kNThreads, int(sizeof(s3bdvws::SharedStorageDv)), stream>>>(p);
}

extern "C" void mxfp8_dvdk_ws_launch(
    const void* Kd, const void* Vd,
    const void* Qd, const void* Dd, const void* Qt, const void* Dt,
    const void* sfK, const void* sfV, const void* sfQ, const void* sfD,
    const void* sfQt, const void* sfDt,
    const float* lse, const float* delta,
    float* dV, float* dK, int S, int H, float sm_scale, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(s3bdvdk::dvdk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3bdvdk::SharedStorageDvDK)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3bdvdk::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvdk::kBlockM), D, H));
  auto layoutSFQt = s3bdvdk::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvdk::kBlockN), D, S, H));
  s3bdvdk::ParamsDvDK p{};
  auto nat = [&](const void* x) {
    return cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(x)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  };
  auto trn = [&](const void* x) {
    return cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(x)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  };
  cute::Tensor mQ = nat(Qd); cute::Tensor mD = nat(Dd);
  cute::Tensor mQt = trn(Qt); cute::Tensor mDt = trn(Dt);
  cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(sfD)), layoutSFQ);
  cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(sfQt)), layoutSFQt);
  cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(sfDt)), layoutSFQt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdvdk::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdvdk::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_d = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mD, s3bdvdk::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdvdk::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, s3bdvdk::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdvdk::kBlockM>{}), cute::_1{});
  p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3bdvdk::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdvdk::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdvdk::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfd = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFD, s3bdvdk::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfqt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQt, s3bdvdk::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3bdvdk::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.layout_sfq = layoutSFQ; p.layout_sfqt = layoutSFQt;
  p.K = (const uint8_t*)Kd; p.V = (const uint8_t*)Vd;
  p.sfK = (const uint8_t*)sfK; p.sfV = (const uint8_t*)sfV;
  p.lse = lse; p.delta = delta; p.dV = dV; p.dK = dK;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
#ifdef DVDK_DEBUG
  { if (!g_dvdk_dbg) { cudaMalloc(&g_dvdk_dbg, 2 * (2*64*64 + 1024)); cudaMemset(g_dvdk_dbg, 0, 2 * (2*64*64 + 1024)); } p.dbg = g_dvdk_dbg; }
#endif
  dim3 grid(S / s3bdvdk::kBlockN, H);
  s3bdvdk::dvdk_ws_kernel<<<grid, s3bdvdk::kNThreads, int(sizeof(s3bdvdk::SharedStorageDvDK)), stream>>>(p);
}

extern "C" void mxfp8_dvdk2_launch(
    const void* Kd, const void* Vd,
    const void* Qd, const void* Dd, const void* Qt, const void* Dt,
    const void* sfK, const void* sfV, const void* sfQ, const void* sfD,
    const void* sfQt, const void* sfDt,
    const float* lse, const float* delta,
    float* dV, float* dK, int S, int H, float sm_scale, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute((const void*)s3bdvdk2::dvdk2_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(sizeof(s3bdvdk2::SharedStorageDvdk)));
    attr_done = true;
  }
  constexpr int D = 128;
  auto layoutSFQ = s3bdvdk2::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvdk2::kBlockM), D, H));
  auto layoutSFQt = s3bdvdk2::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvdk2::kBlockN), D, S, H));
  s3bdvdk2::ParamsDvdk p{};
  auto nat = [&](const void* x) {
    return cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(x)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
  };
  auto trn = [&](const void* x) {
    return cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(x)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
  };
  cute::Tensor mQ = nat(Qd); cute::Tensor mD = nat(Dd);
  cute::Tensor mQt = trn(Qt); cute::Tensor mDt = trn(Dt);
  cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(sfQ)), layoutSFQ);
  cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(sfD)), layoutSFQ);
  cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(sfQt)), layoutSFQt);
  cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(sfDt)), layoutSFQt);
  p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdvdk2::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdvdk2::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_d = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mD, s3bdvdk2::SmemLayoutQ{}(_, _, cute::_0{}),
                                make_shape(Int<s3bdvdk2::kBlockM>{}, Int<D>{}), cute::_1{});
  p.tma_qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, s3bdvdk2::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdvdk2::kBlockM>{}), cute::_1{});
  p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3bdvdk2::SmemLayoutQt{}(_, _, cute::_0{}),
                                 make_shape(Int<D>{}, Int<s3bdvdk2::kBlockM>{}), cute::_1{});
  p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdvdk2::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfd = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFD, s3bdvdk2::SmemLayoutSFT{},
                                            make_shape(Int<128>{}, Int<D>{}), cute::_1{});
  p.tma_sfqt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQt, s3bdvdk2::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3bdvdk2::SmemLayoutSFT{},
                                             make_shape(Int<D>{}, Int<128>{}), cute::_1{});
  cute::Tensor mLse = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const uint8_t*>(lse)),
      cute::make_layout(cute::make_shape(S * H * 4), cute::make_stride(cute::_1{})));
  cute::Tensor mDlt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const uint8_t*>(delta)),
      cute::make_layout(cute::make_shape(S * H * 4), cute::make_stride(cute::_1{})));
  p.tma_lse = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mLse, cute::Layout<cute::Shape<cute::_256>>{},
                                  cute::make_shape(cute::Int<256>{}), cute::_1{});
  p.tma_dlt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDlt, cute::Layout<cute::Shape<cute::_256>>{},
                                  cute::make_shape(cute::Int<256>{}), cute::_1{});
  p.layout_sfq = layoutSFQ; p.layout_sfqt = layoutSFQt; p.layout_sfdt = layoutSFQt;
  p.K = (const uint8_t*)Kd; p.V = (const uint8_t*)Vd;
  p.sfK = (const uint8_t*)sfK; p.sfV = (const uint8_t*)sfV;
  p.lse_raw = lse; p.dlt_raw = delta;
  p.dK = dK; p.dV = dV;
  p.S = S; p.H = H; p.sm_scale = sm_scale;
  dim3 grid(S / s3bdvdk2::kBlockN, H);
  s3bdvdk2::dvdk2_kernel<false><<<grid, 256, int(sizeof(s3bdvdk2::SharedStorageDvdk)), stream>>>(p);
}

extern "C" void mxfp8_bwd_launch(
    const void* Qd, const void* Kd, const void* Vd, const void* Dd,
    const void* Qt, const void* Kt, const void* Dt,
    const void* sfQ, const void* sfK, const void* sfV, const void* sfD,
    const void* sfQt, const void* sfKt, const void* sfDt,
    const float* lse, const float* delta,
    float* dQ, float* dK, float* dV,
    int S, int H, float sm_scale, int use_dk_ws, uintptr_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

  // ---- dv64 / dk64 ----
  {
    static bool attr_done = false;
    if (!attr_done) {
      cudaFuncSetAttribute(s3b64::dv64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(sizeof(s3b64::SharedStorageDV)));
      cudaFuncSetAttribute(s3b64::dk64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(sizeof(s3b64::SharedStorageDK)));
      attr_done = true;
    }
    s3b64::ParamsBwd64 p{};
    p.Q = (const uint8_t*)Qd; p.K = (const uint8_t*)Kd; p.V = (const uint8_t*)Vd;
    p.D = (const uint8_t*)Dd; p.Qt = (const uint8_t*)Qt; p.Kt = (const uint8_t*)Kt;
    p.Dt = (const uint8_t*)Dt;
    p.sfQ = (const uint8_t*)sfQ; p.sfK = (const uint8_t*)sfK; p.sfV = (const uint8_t*)sfV;
    p.sfD = (const uint8_t*)sfD; p.sfQt = (const uint8_t*)sfQt; p.sfKt = (const uint8_t*)sfKt;
    p.sfDt = (const uint8_t*)sfDt;
    p.lse = lse; p.delta = delta; p.dQ = dQ; p.dK = dK; p.dV = dV;
    p.S = S; p.H = H; p.sm_scale = sm_scale;
    if (!use_dk_ws) {
      dim3 grid(S / 64, H);
      s3b64::dv64_kernel<<<grid, s3b64::kNThreads, int(sizeof(s3b64::SharedStorageDV)), stream>>>(p);
      s3b64::dk64_kernel<<<grid, s3b64::kNThreads, int(sizeof(s3b64::SharedStorageDK)), stream>>>(p);
    }
  }
  if (use_dk_ws == 1) {
    mxfp8_dv_ws_launch(Kd, Qd, Dt, sfK, sfQ, sfDt, lse, dV, S, H, sm_scale, stream_);
    mxfp8_dk_ws_launch(Kd, Vd, Qd, Dd, Qt, sfK, sfV, sfQ, sfD, sfQt, lse, delta, dK,
                       S, H, sm_scale, stream_);
  } else if (use_dk_ws == 2) {
    mxfp8_dvdk_ws_launch(Kd, Vd, Qd, Dd, Qt, Dt, sfK, sfV, sfQ, sfD, sfQt, sfDt,
                         lse, delta, dV, dK, S, H, sm_scale, stream_);
  } else if (use_dk_ws == 3) {
    mxfp8_dvdk2_launch(Kd, Vd, Qd, Dd, Qt, Dt, sfK, sfV, sfQ, sfD, sfQt, sfDt,
                       lse, delta, dV, dK, S, H, sm_scale, stream_);
  }

  // ---- dq_ws ----
  {
    static bool attr_done = false;
    if (!attr_done) {
      cudaFuncSetAttribute(s3bws::dq_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(sizeof(s3bws::SharedStorageDq)));
      attr_done = true;
    }
    constexpr int D = 128;
    auto layoutSFK = s3bws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bws::kBlockN), D, H));
    auto layoutSFKt = s3bws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bws::kBlockM), D, S, H));
    s3bws::ParamsDq p{};
    cute::Tensor mK = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(Kd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mV = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(Vd)), mK.layout());
    cute::Tensor mKt = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(Kt)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mSFK = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(sfK)), layoutSFK);
    cute::Tensor mSFV = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(sfV)), layoutSFK);
    cute::Tensor mSFKt = cute::make_tensor(
        cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(sfKt)), layoutSFKt);
    p.tma_k = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mK, s3bws::SmemLayoutK{}(_, _, cute::_0{}),
                                  make_shape(Int<s3bws::kBlockN>{}, Int<D>{}), cute::_1{});
    p.tma_v = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mV, s3bws::SmemLayoutK{}(_, _, cute::_0{}),
                                  make_shape(Int<s3bws::kBlockN>{}, Int<D>{}), cute::_1{});
    p.tma_kt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mKt, s3bws::SmemLayoutKt{}(_, _, cute::_0{}),
                                   make_shape(Int<D>{}, Int<s3bws::kBlockN>{}), cute::_1{});
    p.tma_sfk = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFK, s3bws::SmemLayoutSFT{},
                                              make_shape(Int<128>{}, Int<D>{}), cute::_1{});
    p.tma_sfv = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFV, s3bws::SmemLayoutSFT{},
                                              make_shape(Int<128>{}, Int<D>{}), cute::_1{});
    p.tma_sfkt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFKt, s3bws::SmemLayoutSFT{},
                                               make_shape(Int<D>{}, Int<128>{}), cute::_1{});
    p.layout_sfk = layoutSFK;
    p.layout_sfkt = layoutSFKt;
    p.Q = (const uint8_t*)Qd; p.D = (const uint8_t*)Dd;
    p.sfQ = (const uint8_t*)sfQ; p.sfD = (const uint8_t*)sfD;
    p.lse = lse; p.delta = delta; p.dQ = dQ;
    p.S = S; p.H = H; p.sm_scale = sm_scale;
    p.dbg = nullptr;
    dim3 grid(S / s3bws::kBlockM, H);
    s3bws::dq_ws_kernel<<<grid, s3bws::kNThreads, int(sizeof(s3bws::SharedStorageDq)), stream>>>(p);
  }
}

// ---------------------------------------------------------------------------
// Fused MXFP8 quantization (bf16 [H,L,128] -> e4m3 + ue8m0, zero-pad L->S).
// e4m3 bytes via cutlass RNE-saturating conversion (matches MMA kernels).
// ---------------------------------------------------------------------------
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

__device__ __forceinline__ int q_e8m0(float amax) {
  float a = fmaxf(amax, 1.1754943508222875e-38f);  // 2^-126
  int b = (int)ceilf(log2f(a)) - 8 + 127;
  return max(1, min(254, b));
}
__device__ __forceinline__ uint8_t q_e4m3(float v) {
  return cutlass::float_e4m3_t(v).storage;
}

// one thread per 32-element group along d. grid covers H*S*4 threads.
// qT (nullable): second output in token-major [S,H,D] layout for the fwd kernel --
// written in the same pass, saves a separate 18MB-per-tensor permute+copy.
extern "C" __global__ void quant_nat_kernel(
    const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q,
    uint8_t* __restrict__ sf_raw, uint8_t* __restrict__ sf_pack,
    int H, int S, int L, uint8_t* __restrict__ qT = nullptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int groups = S * 4;
  if (idx >= H * groups) return;
  int h = idx / groups, rem = idx % groups, r = rem / 4, kb = rem & 3;
  size_t goff = ((size_t)h * S + r) * 128 + kb * 32;
  union { uint8_t b[32]; uint4 v[2]; } out;
  int b = 1;
  if (r < L) {
    const __nv_bfloat16* src = x + ((size_t)h * L + r) * 128 + kb * 32;
    float amax = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(__bfloat162float(src[j])));
    b = q_e8m0(amax);
    float s = exp2f(float(b - 127));
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 32; ++j) out.b[j] = q_e4m3(__bfloat162float(src[j]) / s);
  } else {
    out.v[0] = make_uint4(0, 0, 0, 0); out.v[1] = make_uint4(0, 0, 0, 0);
  }
  reinterpret_cast<uint4*>(q + goff)[0] = out.v[0];
  reinterpret_cast<uint4*>(q + goff)[1] = out.v[1];
  if (qT != nullptr) {
    size_t toff = ((size_t)r * H + h) * 128 + kb * 32;
    reinterpret_cast<uint4*>(qT + toff)[0] = out.v[0];
    reinterpret_cast<uint4*>(qT + toff)[1] = out.v[1];
  }
  sf_raw[((size_t)h * S + r) * 4 + kb] = uint8_t(b);
  sf_pack[(size_t)h * (S / 128) * 512 + (r / 128) * 512 + 16 * (r % 32) + 4 * ((r % 128) / 32) + kb] = uint8_t(b);
}

// fused bwd postprocess: out[i] = bf16(in[i] * scale) sliced to L rows, 3 tensors in one launch.
extern "C" __global__ void scale_slice_cast_kernel(
    const float* __restrict__ dq, const float* __restrict__ dk, const float* __restrict__ dv,
    __nv_bfloat16* __restrict__ oq, __nv_bfloat16* __restrict__ ok, __nv_bfloat16* __restrict__ ov,
    float sm, int H, int S, int L) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = int64_t(H) * L * 128;
  if (idx >= total) return;
  int d = int(idx % 128); int64_t hl = idx / 128;
  int l = int(hl % L), h = int(hl / L);
  size_t src = ((size_t)h * S + l) * 128 + d;
  oq[idx] = __float2bfloat16(dq[src] * sm);
  ok[idx] = __float2bfloat16(dk[src] * sm);
  ov[idx] = __float2bfloat16(dv[src]);
}

extern "C" void scale_slice_cast_launch(
    const void* dq, const void* dk, const void* dv, void* oq, void* ok, void* ov,
    float sm, int H, int S, int L, uintptr_t stream_) {
  int64_t total = int64_t(H) * L * 128;
  int blocks = int((total + 255) / 256);
  scale_slice_cast_kernel<<<blocks, 256, 0, (cudaStream_t)stream_>>>(
      (const float*)dq, (const float*)dk, (const float*)dv,
      (__nv_bfloat16*)oq, (__nv_bfloat16*)ok, (__nv_bfloat16*)ov, sm, H, S, L);
}

// block = one (head, 32-token group): smem tile 32x128, 128 threads.
extern "C" __global__ void __launch_bounds__(128) quant_trn_kernel(
    const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ qt,
    uint8_t* __restrict__ sf_raw, uint8_t* __restrict__ sf_pack,
    int H, int S, int L) {
  __shared__ __nv_bfloat16 tile[32][136];
  int const h = blockIdx.x, kb = blockIdx.y;
  int const tid = threadIdx.x;
  int const s0 = kb * 32;
  for (int i = tid; i < 32 * 128; i += 128) {
    int s = i / 128, d = i % 128;
    tile[s][d] = (s0 + s < L) ? x[((size_t)h * L + s0 + s) * 128 + d] : __float2bfloat16(0.f);
  }
  __syncthreads();
  int const d = tid;
  float amax = 0.f;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(__bfloat162float(tile[j][d])));
  int b = q_e8m0(amax);
  float s = exp2f(float(b - 127));
  union { uint8_t b[32]; uint4 v[2]; } out;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 32; ++j) out.b[j] = q_e4m3(__bfloat162float(tile[j][d]) / s);
  uint8_t* dst = qt + ((size_t)h * 128 + d) * S + s0;
  reinterpret_cast<uint4*>(dst)[0] = out.v[0];
  reinterpret_cast<uint4*>(dst)[1] = out.v[1];
  sf_raw[((size_t)h * 128 + d) * (S / 32) + kb] = uint8_t(b);
  sf_pack[(size_t)h * (S / 128) * 512 + (kb / 4) * 512 + 16 * (d % 32) + 4 * (d / 32) + (kb % 4)] = uint8_t(b);
}

extern "C" void quant_nat_launch(const void* x, void* q, void* sf_raw, void* sf_pack,
                                 int H, int S, int L, uintptr_t stream_, void* q_trn = nullptr) {
  int total = H * S * 4;
  quant_nat_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t)stream_>>>(
      (const __nv_bfloat16*)x, (uint8_t*)q, (uint8_t*)sf_raw, (uint8_t*)sf_pack, H, S, L,
      (uint8_t*)q_trn);
}
extern "C" void quant_trn_launch(const void* x, void* qt, void* sf_raw, void* sf_pack,
                                 int H, int S, int L, uintptr_t stream_) {
  dim3 grid(H, S / 32);
  quant_trn_kernel<<<grid, 128, 0, (cudaStream_t)stream_>>>(
      (const __nv_bfloat16*)x, (uint8_t*)qt, (uint8_t*)sf_raw, (uint8_t*)sf_pack, H, S, L);
}
