// pybind glue for the MXFP8 attention backward (nvcc-built kernels live in
// mxfp8_bwd_kernel.cu). Single op:
//   mxfp8_bwd(Qd,Kd,Vd,Dd,Qt,Kt,Dt, sfQ,sfK,sfV,sfD,sfQt,sfKt,sfDt, lse,delta, sm_scale)
//     -> (dQ, dK, dV) fp32 [H,S,128]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void mxfp8_bwd_launch(
    const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, float*, float*, int, int, float, int, uintptr_t);
extern "C" void mxfp8_dk_ws_launch(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, int, int, float, uintptr_t);
extern uint8_t* g_dvdk_dbg;
extern "C" void quant_nat_launch(const void*, void*, void*, void*, int, int, int, uintptr_t);
extern "C" void quant_trn_launch(const void*, void*, void*, void*, int, int, int, uintptr_t);
extern "C" void mxfp8_dvdk_ws_launch(
    const void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, const float*, float*, float*, int, int, float, uintptr_t);
extern "C" void mxfp8_dv_ws_launch(
    const void*, const void*, const void*, const void*, const void*, const void*,
    const float*, float*, int, int, float, uintptr_t);

static std::vector<at::Tensor> mxfp8_bwd(
    at::Tensor Qd, at::Tensor Kd, at::Tensor Vd, at::Tensor Dd,
    at::Tensor Qt, at::Tensor Kt, at::Tensor Dt,
    at::Tensor sfQ, at::Tensor sfK, at::Tensor sfV, at::Tensor sfD,
    at::Tensor sfQt, at::Tensor sfKt, at::Tensor sfDt,
    at::Tensor lse, at::Tensor delta, double sm_scale, int64_t use_dk_ws) {
  TORCH_CHECK(Qd.is_cuda() && Qd.dtype() == at::kByte && Qd.is_contiguous());
  const int H = Qd.size(0), S = Qd.size(1), D = Qd.size(2);
  TORCH_CHECK(D == 128 && S % 128 == 0, "mxfp8_bwd: need head_dim=128, S%128==0");
  auto opts = at::TensorOptions().dtype(at::kFloat).device(Qd.device());
  at::Tensor dQ = at::empty({H, S, D}, opts);
  at::Tensor dK = at::empty({H, S, D}, opts);
  at::Tensor dV = at::empty({H, S, D}, opts);
  mxfp8_bwd_launch(Qd.data_ptr(), Kd.data_ptr(), Vd.data_ptr(), Dd.data_ptr(),
                   Qt.data_ptr(), Kt.data_ptr(), Dt.data_ptr(),
                   sfQ.data_ptr(), sfK.data_ptr(), sfV.data_ptr(), sfD.data_ptr(),
                   sfQt.data_ptr(), sfKt.data_ptr(), sfDt.data_ptr(),
                   lse.data_ptr<float>(), delta.data_ptr<float>(),
                   dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
                   S, H, float(sm_scale), int(use_dk_ws),
                   uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dQ, dK, dV};
}

static std::vector<at::Tensor> quant_op(at::Tensor x, bool transposed) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.dim() == 3 && x.size(2) == 128);
  auto xc = x.contiguous();
  const int H = xc.size(0), L = xc.size(1);
  const int S = (L + 127) / 128 * 128;
  auto u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
  at::Tensor q, sf_raw, sf_pack;
  if (!transposed) {
    q = at::empty({H, S, 128}, u8);
    sf_raw = at::empty({H, S, 4}, u8);
    sf_pack = at::empty({H, S / 128, 512}, u8);
    quant_nat_launch(xc.data_ptr(), q.data_ptr(), sf_raw.data_ptr(), sf_pack.data_ptr(),
                     H, S, L, uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  } else {
    q = at::empty({H, 128, S}, u8);
    sf_raw = at::empty({H, 128, S / 32}, u8);
    sf_pack = at::empty({H, S / 128, 512}, u8);
    quant_trn_launch(xc.data_ptr(), q.data_ptr(), sf_raw.data_ptr(), sf_pack.data_ptr(),
                     H, S, L, uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, sf_raw, sf_pack};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_nat", [](at::Tensor x) { return quant_op(x, false); }, "fused mxfp8 natural quant (pads to S)");
  m.def("quant_trn", [](at::Tensor x) { return quant_op(x, true); }, "fused mxfp8 transposed quant (pads to S)"); m.def("mxfp8_bwd", &mxfp8_bwd, "mxfp8 attention bwd");
  m.def("dvdk_ws", [](at::Tensor Kd, at::Tensor Vd, at::Tensor Qd, at::Tensor Dd,
                      at::Tensor Qt, at::Tensor Dt, at::Tensor sfK, at::Tensor sfV,
                      at::Tensor sfQ, at::Tensor sfD, at::Tensor sfQt, at::Tensor sfDt,
                      at::Tensor lse, at::Tensor delta, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    auto f32 = at::TensorOptions().dtype(at::kFloat).device(Kd.device());
    at::Tensor dV = at::empty({H, S, 128}, f32);
    at::Tensor dK = at::empty({H, S, 128}, f32);
    mxfp8_dvdk_ws_launch(Kd.data_ptr(), Vd.data_ptr(), Qd.data_ptr(), Dd.data_ptr(),
                         Qt.data_ptr(), Dt.data_ptr(), sfK.data_ptr(), sfV.data_ptr(),
                         sfQ.data_ptr(), sfD.data_ptr(), sfQt.data_ptr(), sfDt.data_ptr(),
                         lse.data_ptr<float>(), delta.data_ptr<float>(),
                         dV.data_ptr<float>(), dK.data_ptr<float>(),
                         S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::vector<at::Tensor>{dV, dK};
  }, "fused dv+dk ws kernel (bench)");
  m.def("dvdk_dbg", []() {
    auto t = at::empty({2 * (2 * 64 * 64 + 1024)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    if (g_dvdk_dbg) cudaMemcpy(t.data_ptr(), g_dvdk_dbg, t.numel(), cudaMemcpyDeviceToDevice);
    return t;
  });
  m.def("dv_ws", [](at::Tensor Kd, at::Tensor Qd, at::Tensor Dt,
                    at::Tensor sfK, at::Tensor sfQ, at::Tensor sfDt,
                    at::Tensor lse, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    at::Tensor dV = at::empty({H, S, 128}, at::TensorOptions().dtype(at::kFloat).device(Kd.device()));
    mxfp8_dv_ws_launch(Kd.data_ptr(), Qd.data_ptr(), Dt.data_ptr(),
                       sfK.data_ptr(), sfQ.data_ptr(), sfDt.data_ptr(),
                       lse.data_ptr<float>(), dV.data_ptr<float>(),
                       S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dV;
  }, "dv ws kernel only (bench)");
  m.def("dk_ws", [](at::Tensor Kd, at::Tensor Vd, at::Tensor Qd, at::Tensor Dd, at::Tensor Qt,
                    at::Tensor sfK, at::Tensor sfV, at::Tensor sfQ, at::Tensor sfD, at::Tensor sfQt,
                    at::Tensor lse, at::Tensor delta, double sm_scale) {
    const int H = Kd.size(0), S = Kd.size(1);
    at::Tensor dK = at::empty({H, S, 128}, at::TensorOptions().dtype(at::kFloat).device(Kd.device()));
    mxfp8_dk_ws_launch(Kd.data_ptr(), Vd.data_ptr(), Qd.data_ptr(), Dd.data_ptr(), Qt.data_ptr(),
                       sfK.data_ptr(), sfV.data_ptr(), sfQ.data_ptr(), sfD.data_ptr(), sfQt.data_ptr(),
                       lse.data_ptr<float>(), delta.data_ptr<float>(), dK.data_ptr<float>(),
                       S, H, float(sm_scale), uintptr_t(at::cuda::getCurrentCUDAStream().stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dK;
  }, "dk ws kernel only (bench)"); }
