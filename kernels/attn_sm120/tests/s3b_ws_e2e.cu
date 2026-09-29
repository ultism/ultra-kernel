// S3B-WS unified harness: dq_ws + dk_ws + dv_ws correctness & bench (sm_120a).
// Correctness (default): H=1, S=S3B_S (256), fp64-dequant reference, rel-L2 for
// dQ/dK/dV. Bench (-DS3B_BENCH): H=32, S=16896, per-kernel event timing.
// Build (from /root/fa-blackwell):
//   nvcc -std=c++17 -O2 [-DS3B_BENCH] [-DS3B_S=1024] -Xptxas -v \
//     -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include kernels/attn_sm120/tests/s3b_ws_e2e.cu \
//     -o kernels/attn_sm120/tests/s3b_ws_e2e
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include "s3b_dq_ws_kernel.cuh"
#include "s3b_dq_ws_ref.cuh"
#include "s3b_dk_ws_kernel.cuh"

#include "s3b_dv_ws_kernel.cuh"

#include "s3b_dvdk2_kernel.cuh"
#include "s3b_dvdk3_kernel.cuh"

#include "s3b_dvdk_ws_kernel.cuh"

#include "s3b_dvdk221_kernel.cuh"
#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#ifndef S3B_BENCH
#ifndef S3B_S
#define S3B_S 256
#endif
static const int H = 1, S = S3B_S, D = 128;
#else
#ifndef BENCH_H
#define BENCH_H 32
#define BENCH_S 16896
#endif
static const int H = BENCH_H, S = BENCH_S, D = 128;
#endif
static const int MT = S / 128;   // 128-row SF tiles per head

static int e8m0_byte(double amax) {
  if (amax <= 0) return 1;
  double a = std::max(amax, std::ldexp(1.0, -126));
  return std::max(1, std::min(254, (int)std::ceil(std::log2(a)) - 8 + 127));
}
// natural quant: [S,128] -> data + flat sf tiles [S/128][512] (scales along d)
static void quant_nat(const float* x, uint8_t* q, uint8_t* sf) {
  for (int r = 0; r < S; ++r)
    for (int kb = 0; kb < D / 32; ++kb) {
      double amax = 0;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, (double)std::fabs(x[(size_t)r * D + kb * 32 + j]));
      int b = e8m0_byte(amax);
      double s = std::ldexp(1.0, b - 127);
      for (int j = 0; j < 32; ++j)
        q[(size_t)r * D + kb * 32 + j] = cutlass::float_e4m3_t(float(x[(size_t)r * D + kb * 32 + j] / s)).storage;
      sf[(r / 128) * 512 + 16 * (r % 32) + 4 * ((r % 128) / 32) + kb] = uint8_t(b);
    }
}
// transposed quant: xt[d,s]=x[s,d]; scales along s. flat sf tiles [S/128][512]
static void quant_trn(const float* x, uint8_t* q, uint8_t* sf) {
  for (int d = 0; d < D; ++d)
    for (int kb = 0; kb < S / 32; ++kb) {
      double amax = 0;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, (double)std::fabs(x[(size_t)(kb * 32 + j) * D + d]));
      int b = e8m0_byte(amax);
      double s = std::ldexp(1.0, b - 127);
      for (int j = 0; j < 32; ++j)
        q[(size_t)d * S + kb * 32 + j] = cutlass::float_e4m3_t(float(x[(size_t)(kb * 32 + j) * D + d] / s)).storage;
      sf[(kb / 4) * 512 + 16 * (d % 32) + 4 * (d / 32) + (kb % 4)] = uint8_t(b);
    }
}

using cute::Int;
using cute::make_coord;
using cute::make_shape;
using cute::make_stride;

int main() {
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> Q((size_t)H * S * D), K((size_t)H * S * D), V((size_t)H * S * D), dO((size_t)H * S * D);
  for (auto& v : Q) v = nd(rng); for (auto& v : K) v = nd(rng);
  for (auto& v : V) v = nd(rng); for (auto& v : dO) v = nd(rng);

  std::vector<uint8_t> qQ(H * S * D), qK(H * S * D), qV(H * S * D), qD(H * S * D),
      qQt(H * S * D), qKt(H * S * D), qDt(H * S * D);
  std::vector<uint8_t> sfQ(H * MT * 512), sfK(H * MT * 512), sfV(H * MT * 512), sfD(H * MT * 512),
      sfQt(H * MT * 512), sfKt(H * MT * 512), sfDt(H * MT * 512);
  for (int h = 0; h < H; ++h) {
    size_t o = (size_t)h * S * D;
    quant_nat(Q.data() + o, qQ.data() + o, sfQ.data() + (size_t)h * MT * 512);
    quant_nat(K.data() + o, qK.data() + o, sfK.data() + (size_t)h * MT * 512);
    quant_nat(V.data() + o, qV.data() + o, sfV.data() + (size_t)h * MT * 512);
    quant_nat(dO.data() + o, qD.data() + o, sfD.data() + (size_t)h * MT * 512);
    quant_trn(Q.data() + o, qQt.data() + o, sfQt.data() + (size_t)h * MT * 512);
    quant_trn(K.data() + o, qKt.data() + o, sfKt.data() + (size_t)h * MT * 512);
    quant_trn(dO.data() + o, qDt.data() + o, sfDt.data() + (size_t)h * MT * 512);
  }

  std::vector<float> lsef(H * S), dltf(H * S);
  double sm = 1.0 / std::sqrt(128.0);
#ifndef S3B_BENCH
  // fp64 reference on dequantized natural inputs
  std::vector<double> P(S * S), dS(S * S), lse(S), delta(S), Od(S * D);
  std::vector<double> rQ(S * D, 0), rK(S * D, 0), rV(S * D, 0);
  auto deq = [&](const std::vector<uint8_t>& q, const std::vector<uint8_t>& sf, int r, int c) {
    int b = sf[(r / 128) * 512 + 16 * (r % 32) + 4 * ((r % 128) / 32) + c / 32];
    return double(float(cutlass::float_e4m3_t::bitcast(q[(size_t)r * D + c]))) * std::ldexp(1.0, b - 127);
  };
  for (int i = 0; i < S; ++i) {
    double mx = -1e300;
    std::vector<double> row(S);
    for (int j = 0; j < S; ++j) {
      double s = 0;
      for (int d = 0; d < D; ++d) s += deq(qQ, sfQ, i, d) * deq(qK, sfK, j, d);
      row[j] = s * sm; mx = std::max(mx, row[j]);
    }
    double sum = 0;
    for (int j = 0; j < S; ++j) { P[i * S + j] = std::exp(row[j] - mx); sum += P[i * S + j]; }
    lse[i] = mx + std::log(sum);
    for (int j = 0; j < S; ++j) P[i * S + j] /= sum;
  }
  for (int i = 0; i < S; ++i)
    for (int d = 0; d < D; ++d) {
      double o = 0;
      for (int j = 0; j < S; ++j) o += P[i * S + j] * deq(qV, sfV, j, d);
      Od[i * D + d] = o;
    }
  for (int i = 0; i < S; ++i) {
    double dl = 0;
    for (int d = 0; d < D; ++d) dl += deq(qD, sfD, i, d) * Od[i * D + d];
    delta[i] = dl;
  }
  for (int i = 0; i < S; ++i)
    for (int j = 0; j < S; ++j) {
      double dp = 0;
      for (int d = 0; d < D; ++d) dp += deq(qD, sfD, i, d) * deq(qV, sfV, j, d);
      dS[i * S + j] = P[i * S + j] * (dp - delta[i]);
    }
  for (int i = 0; i < S; ++i)
    for (int d = 0; d < D; ++d) {
      double dq = 0, dk = 0, dv = 0;
      for (int j = 0; j < S; ++j) {
        dq += dS[i * S + j] * deq(qK, sfK, j, d);
        dk += dS[j * S + i] * deq(qQ, sfQ, j, d);
        dv += P[j * S + i] * deq(qD, sfD, j, d);
      }
      rQ[i * D + d] = dq; rK[i * D + d] = dk; rV[i * D + d] = dv;
    }
  for (int i = 0; i < S; ++i) { lsef[i] = float(lse[i]); dltf[i] = float(delta[i]); }
#else
  { std::mt19937 r2(1); std::normal_distribution<float> n2(3.f, 1.5f);
    for (auto& v : lsef) v = n2(r2); for (auto& v : dltf) v = n2(r2); }
#endif

  auto up = [&](const std::vector<uint8_t>& v) {
    uint8_t* p; CK(cudaMalloc(&p, v.size())); CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice)); return p;
  };
  uint8_t *dQd = up(qQ), *dKd = up(qK), *dVd = up(qV), *dDd = up(qD);
  uint8_t *dQtd = up(qQt), *dKtd = up(qKt), *dDtd = up(qDt);
  uint8_t *dsfQ = up(sfQ), *dsfK = up(sfK), *dsfV = up(sfV), *dsfD = up(sfD);
  uint8_t *dsfQt = up(sfQt), *dsfKt = up(sfKt), *dsfDt = up(sfDt);
  float *dLse, *dDlt; CK(cudaMalloc(&dLse, H * S * 4)); CK(cudaMalloc(&dDlt, H * S * 4));
  CK(cudaMemcpy(dLse, lsef.data(), H * S * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dDlt, dltf.data(), H * S * 4, cudaMemcpyHostToDevice));
  float *ddQ, *ddK, *ddV, *ddQref;
  CK(cudaMalloc(&ddQ, (size_t)H * S * D * 4)); CK(cudaMalloc(&ddK, (size_t)H * S * D * 4));
  CK(cudaMalloc(&ddV, (size_t)H * S * D * 4)); CK(cudaMalloc(&ddQref, (size_t)H * S * D * 4));
  float *ddK2, *ddV2;
  CK(cudaMalloc(&ddK2, (size_t)H * S * D * 4)); CK(cudaMalloc(&ddV2, (size_t)H * S * D * 4));

  // ---------------- dq_ws launch ----------------
  auto launch_dq = [&] {
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(s3bws::dq_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(s3bws::SharedStorageDq)))); attr = true; }
    auto layoutSFK = s3bws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bws::kBlockN), D, H));
    auto layoutSFKt = s3bws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bws::kBlockM), D, S, H));
    s3bws::ParamsDq p{};
    cute::Tensor mK = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(dKd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mV = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(dVd)), mK.layout());
    cute::Tensor mKt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::Element const*>(dKtd)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mSFK = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfK)), layoutSFK);
    cute::Tensor mSFV = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfV)), layoutSFK);
    cute::Tensor mSFKt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfKt)), layoutSFKt);
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
    p.layout_sfk = layoutSFK; p.layout_sfkt = layoutSFKt;
    p.Q = dQd; p.D = dDd; p.sfQ = dsfQ; p.sfD = dsfD;
    p.lse = dLse; p.delta = dDlt; p.dQ = ddQ;
    p.S = S; p.H = H; p.sm_scale = float(sm);
    s3bws::dq_ws_kernel<<<dim3(S / s3bws::kBlockM, H), s3bws::kNThreads,
                          int(sizeof(s3bws::SharedStorageDq))>>>(p);
    CK(cudaGetLastError());
  };

  // ---------------- dq_ws_ref launch (bitwise A/B baseline) ----------------
  auto launch_dq_ref = [&] {
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(s3bwsref::dq_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(s3bwsref::SharedStorageDq)))); attr = true; }
    auto layoutSFK = s3bwsref::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bwsref::kBlockN), D, H));
    auto layoutSFKt = s3bwsref::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bwsref::kBlockM), D, S, H));
    s3bwsref::ParamsDq p{};
    cute::Tensor mK = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::Element const*>(dKd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mV = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::Element const*>(dVd)), mK.layout());
    cute::Tensor mKt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::Element const*>(dKtd)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mSFK = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::ElementSF const*>(dsfK)), layoutSFK);
    cute::Tensor mSFV = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::ElementSF const*>(dsfV)), layoutSFK);
    cute::Tensor mSFKt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bwsref::ElementSF const*>(dsfKt)), layoutSFKt);
    p.tma_k = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mK, s3bwsref::SmemLayoutK{}(_, _, cute::_0{}),
                                  make_shape(Int<s3bwsref::kBlockN>{}, Int<D>{}), cute::_1{});
    p.tma_v = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mV, s3bwsref::SmemLayoutK{}(_, _, cute::_0{}),
                                  make_shape(Int<s3bwsref::kBlockN>{}, Int<D>{}), cute::_1{});
    p.tma_kt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mKt, s3bwsref::SmemLayoutKt{}(_, _, cute::_0{}),
                                   make_shape(Int<D>{}, Int<s3bwsref::kBlockN>{}), cute::_1{});
    p.tma_sfk = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFK, s3bwsref::SmemLayoutSFT{},
                                              make_shape(Int<128>{}, Int<D>{}), cute::_1{});
    p.tma_sfv = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFV, s3bwsref::SmemLayoutSFT{},
                                              make_shape(Int<128>{}, Int<D>{}), cute::_1{});
    p.tma_sfkt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFKt, s3bwsref::SmemLayoutSFT{},
                                               make_shape(Int<D>{}, Int<128>{}), cute::_1{});
    p.layout_sfk = layoutSFK; p.layout_sfkt = layoutSFKt;
    p.Q = dQd; p.D = dDd; p.sfQ = dsfQ; p.sfD = dsfD;
    p.lse = dLse; p.delta = dDlt; p.dQ = ddQref;
    p.S = S; p.H = H; p.sm_scale = float(sm);
    s3bwsref::dq_ws_kernel<<<dim3(S / s3bwsref::kBlockM, H), s3bwsref::kNThreads,
                             int(sizeof(s3bwsref::SharedStorageDq))>>>(p);
    CK(cudaGetLastError());
  };

  // ---------------- dk_ws launch ----------------
  auto launch_dk = [&] {
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(s3bdkws::dk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(s3bdkws::SharedStorageDk)))); attr = true; }
    auto layoutSFQ = s3bdkws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdkws::kBlockM), D, H));
    auto layoutSFQt = s3bdkws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdkws::kBlockN), D, S, H));
    s3bdkws::ParamsDk p{};
    cute::Tensor mQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(dQd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(dDd)), mQ.layout());
    cute::Tensor mQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::Element const*>(dQtd)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(dsfQ)), layoutSFQ);
    cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(dsfD)), layoutSFQ);
    cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdkws::ElementSF const*>(dsfQt)), layoutSFQt);
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
    cute::Tensor mLse = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(dLse)),
        cute::make_layout(make_shape(S * H * 4), make_stride(cute::_1{})));
    cute::Tensor mDlt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(dDlt)),
        cute::make_layout(make_shape(S * H * 4), make_stride(cute::_1{})));
    p.tma_lse = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mLse, cute::Layout<cute::Shape<cute::_256>>{},
                                    make_shape(Int<256>{}), cute::_1{});
    p.tma_dlt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDlt, cute::Layout<cute::Shape<cute::_256>>{},
                                    make_shape(Int<256>{}), cute::_1{});
    p.layout_sfq = layoutSFQ; p.layout_sfqt = layoutSFQt;
    p.K = dKd; p.V = dVd; p.sfK = dsfK; p.sfV = dsfV;
    p.dK = ddK;
    p.S = S; p.H = H; p.sm_scale = float(sm);
    s3bdkws::dk_ws_kernel<<<dim3(S / s3bdkws::kBlockN, H), s3bdkws::kNThreads,
                            int(sizeof(s3bdkws::SharedStorageDk))>>>(p);
    CK(cudaGetLastError());
  };

  // ---------------- dv_ws launch ----------------
  auto launch_dv = [&] {
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(s3bdvws::dv_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(s3bdvws::SharedStorageDv)))); attr = true; }
    auto layoutSFQ = s3bdvws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvws::kBlockM), D, H));
    auto layoutSFDt = s3bdvws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvws::kBlockN), D, S, H));
    s3bdvws::ParamsDv p{};
    cute::Tensor mQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvws::Element const*>(dQd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvws::Element const*>(dDtd)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvws::ElementSF const*>(dsfQ)), layoutSFQ);
    cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvws::ElementSF const*>(dsfDt)), layoutSFDt);
    p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdvws::SmemLayoutQ{}(_, _, cute::_0{}),
                                  make_shape(Int<s3bdvws::kBlockM>{}, Int<D>{}), cute::_1{});
    p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3bdvws::SmemLayoutQt{}(_, _, cute::_0{}),
                                   make_shape(Int<D>{}, Int<s3bdvws::kBlockM>{}), cute::_1{});
    p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdvws::SmemLayoutSFT{},
                                              make_shape(Int<128>{}, Int<D>{}), cute::_1{});
    p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3bdvws::SmemLayoutSFT{},
                                               make_shape(Int<D>{}, Int<128>{}), cute::_1{});
    p.layout_sfq = layoutSFQ; p.layout_sfdt = layoutSFDt;
    p.K = dKd; p.sfK = dsfK;
    p.lse = dLse; p.dV = ddV;
    p.S = S; p.H = H; p.sm_scale = float(sm);
    s3bdvws::dv_ws_kernel<<<dim3(S / s3bdvws::kBlockN, H), s3bdvws::kNThreads,
                            int(sizeof(s3bdvws::SharedStorageDv))>>>(p);
    CK(cudaGetLastError());
  };

  // ---------------- dvdk2 (fused dv+dk @kv128) launch: ws / non-ws ----------------
  auto fill_dvdk_params = [&](s3bdvdk2::ParamsDvdk& p) {
    auto layoutSFQ = s3bdvdk2::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvdk2::kBlockM), D, H));
    auto layoutSFQt = s3bdvdk2::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvdk2::kBlockN), D, S, H));
    cute::Tensor mQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(dQd)),
        cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
    cute::Tensor mD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(dDd)), mQ.layout());
    cute::Tensor mQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(dQtd)),
        cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
    cute::Tensor mDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::Element const*>(dDtd)), mQt.layout());
    cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(dsfQ)), layoutSFQ);
    cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(dsfD)), layoutSFQ);
    cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(dsfQt)), layoutSFQt);
    cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk2::ElementSF const*>(dsfDt)), layoutSFQt);
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
    cute::Tensor mLse = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(dLse)),
        cute::make_layout(make_shape(S * H * 4), make_stride(cute::_1{})));
    cute::Tensor mDlt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(dDlt)),
        cute::make_layout(make_shape(S * H * 4), make_stride(cute::_1{})));
    p.tma_lse = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mLse, cute::Layout<cute::Shape<cute::_256>>{},
                                    make_shape(Int<256>{}), cute::_1{});
    p.tma_dlt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDlt, cute::Layout<cute::Shape<cute::_256>>{},
                                    make_shape(Int<256>{}), cute::_1{});
    p.layout_sfq = layoutSFQ; p.layout_sfqt = layoutSFQt; p.layout_sfdt = layoutSFQt;
    p.K = dKd; p.V = dVd; p.sfK = dsfK; p.sfV = dsfV;
    p.lse_raw = dLse; p.dlt_raw = dDlt;
    p.dK = ddK2; p.dV = ddV2;
    p.S = S; p.H = H; p.sm_scale = float(sm);
  };
  auto launch_dvdk2 = [&](bool ws) {
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute((const void*)s3bdvdk2::dvdk2_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(s3bdvdk2::SharedStorageDvdk))));
      CK(cudaFuncSetAttribute((const void*)s3bdvdk2::dvdk2_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(s3bdvdk2::SharedStorageDvdk))));
      attr = true;
    }
    s3bdvdk2::ParamsDvdk p{};
    fill_dvdk_params(p);
    if (ws) {
      s3bdvdk2::dvdk2_kernel<true><<<dim3(S / s3bdvdk2::kBlockN, H), 384,
                              int(sizeof(s3bdvdk2::SharedStorageDvdk))>>>(p);
    } else {
      s3bdvdk2::dvdk2_kernel<false><<<dim3(S / s3bdvdk2::kBlockN, H), 256,
                              int(sizeof(s3bdvdk2::SharedStorageDvdk))>>>(p);
    }
    CK(cudaGetLastError());
  };
  auto launch_dvdk3 = [&] {
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute((const void*)s3bdvdk3::dvdk3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(s3bdvdk3::SharedStorageDvdk3))));
      attr = true;
    }
    s3bdvdk2::ParamsDvdk p{};
    fill_dvdk_params(p);
    s3bdvdk3::dvdk3_kernel<<<dim3(S / s3bdvdk2::kBlockN, H), 512,
                            int(sizeof(s3bdvdk3::SharedStorageDvdk3))>>>(p);
    CK(cudaGetLastError());
  };

  // ---------------- old dvdk (kv=64) launchers: atom (4,2,1) vs (2,2,1) ----------------
  auto launch_dvdk64 = [&](bool v221) {
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute(s3bdvdk::dvdk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(s3bdvdk::SharedStorageDvDK))));
      CK(cudaFuncSetAttribute(s3bdvdk221::dvdk_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(s3bdvdk221::SharedStorageDvDK))));
      attr = true;
    }
    auto fill421 = [&](s3bdvdk::ParamsDvDK& p) {
      auto layoutSFQ = s3bdvdk::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvdk::kBlockM), D, H));
      auto layoutSFQt = s3bdvdk::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvdk::kBlockN), D, S, H));
      cute::Tensor mQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(dQd)),
          cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
      cute::Tensor mD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(dDd)), mQ.layout());
      cute::Tensor mQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(dQtd)),
          cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
      cute::Tensor mDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::Element const*>(dDtd)), mQt.layout());
      cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(dsfQ)), layoutSFQ);
      cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(dsfD)), layoutSFQ);
      cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(dsfQt)), layoutSFQt);
      cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk::ElementSF const*>(dsfDt)), layoutSFQt);
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
      p.K = dKd; p.V = dVd; p.sfK = dsfK; p.sfV = dsfV;
      p.lse = dLse; p.delta = dDlt; p.dV = ddV2; p.dK = ddK2;
      p.S = S; p.H = H; p.sm_scale = float(sm);
    };
    auto fill221 = [&](s3bdvdk221::ParamsDvDK& p) {
      auto layoutSFQ = s3bdvdk221::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bdvdk221::kBlockM), D, H));
      auto layoutSFQt = s3bdvdk221::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bdvdk221::kBlockN), D, S, H));
      cute::Tensor mQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::Element const*>(dQd)),
          cute::make_layout(make_shape(S, D, H), make_stride(D, cute::_1{}, S * D)));
      cute::Tensor mD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::Element const*>(dDd)), mQ.layout());
      cute::Tensor mQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::Element const*>(dQtd)),
          cute::make_layout(make_shape(D, S, H), make_stride(S, cute::_1{}, D * S)));
      cute::Tensor mDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::Element const*>(dDtd)), mQt.layout());
      cute::Tensor mSFQ = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::ElementSF const*>(dsfQ)), layoutSFQ);
      cute::Tensor mSFD = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::ElementSF const*>(dsfD)), layoutSFQ);
      cute::Tensor mSFQt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::ElementSF const*>(dsfQt)), layoutSFQt);
      cute::Tensor mSFDt = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<s3bdvdk221::ElementSF const*>(dsfDt)), layoutSFQt);
      p.tma_q = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQ, s3bdvdk221::SmemLayoutQ{}(_, _, cute::_0{}),
                                    make_shape(Int<s3bdvdk221::kBlockM>{}, Int<D>{}), cute::_1{});
      p.tma_d = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mD, s3bdvdk221::SmemLayoutQ{}(_, _, cute::_0{}),
                                    make_shape(Int<s3bdvdk221::kBlockM>{}, Int<D>{}), cute::_1{});
      p.tma_qt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mQt, s3bdvdk221::SmemLayoutQt{}(_, _, cute::_0{}),
                                     make_shape(Int<D>{}, Int<s3bdvdk221::kBlockM>{}), cute::_1{});
      p.tma_dt = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, mDt, s3bdvdk221::SmemLayoutQt{}(_, _, cute::_0{}),
                                     make_shape(Int<D>{}, Int<s3bdvdk221::kBlockM>{}), cute::_1{});
      p.tma_sfq = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQ, s3bdvdk221::SmemLayoutSFT{},
                                                make_shape(Int<128>{}, Int<D>{}), cute::_1{});
      p.tma_sfd = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFD, s3bdvdk221::SmemLayoutSFT{},
                                                make_shape(Int<128>{}, Int<D>{}), cute::_1{});
      p.tma_sfqt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFQt, s3bdvdk221::SmemLayoutSFT{},
                                                 make_shape(Int<D>{}, Int<128>{}), cute::_1{});
      p.tma_sfdt = cute::make_tma_copy<uint16_t>(cute::SM90_TMA_LOAD{}, mSFDt, s3bdvdk221::SmemLayoutSFT{},
                                                 make_shape(Int<D>{}, Int<128>{}), cute::_1{});
      p.layout_sfq = layoutSFQ; p.layout_sfqt = layoutSFQt;
      p.K = dKd; p.V = dVd; p.sfK = dsfK; p.sfV = dsfV;
      p.lse = dLse; p.delta = dDlt; p.dV = ddV2; p.dK = ddK2;
      p.S = S; p.H = H; p.sm_scale = float(sm);
    };
    if (v221) {
      s3bdvdk221::ParamsDvDK p{};
      fill221(p);
      s3bdvdk221::dvdk_ws_kernel<<<dim3(S / s3bdvdk221::kBlockN, H), s3bdvdk221::kNThreads,
                              int(sizeof(s3bdvdk221::SharedStorageDvDK))>>>(p);
    } else {
      s3bdvdk::ParamsDvDK p{};
      fill421(p);
      s3bdvdk::dvdk_ws_kernel<<<dim3(S / s3bdvdk::kBlockN, H), s3bdvdk::kNThreads,
                              int(sizeof(s3bdvdk::SharedStorageDvDK))>>>(p);
    }
    CK(cudaGetLastError());
  };

  printf("smem dq: %zu dk: %zu dv: %zu\n", sizeof(s3bws::SharedStorageDq),
         sizeof(s3bdkws::SharedStorageDk), sizeof(s3bdvws::SharedStorageDv));
#ifndef S3B_BENCH
  launch_dq_ref();
#endif
  launch_dq(); launch_dk(); launch_dv();
  CK(cudaDeviceSynchronize());

#ifdef S3B_BENCH
  // Thermal/power noise on this box is +-20% at the big shape; use small
  // shapes + grouped timing: 5 groups x 20 back-to-back launches per group,
  // report per-iter min/median of the group means. Alternate
  // candidate/baseline binaries when comparing, never trust a single run.
  auto bench = [&](const char* nm, auto fn) {
    fn(); fn(); CK(cudaDeviceSynchronize());
    float ts[5];
    for (int g = 0; g < 5; ++g) {
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a);
      for (int i = 0; i < 20; ++i) fn();
      cudaEventRecord(b); CK(cudaEventSynchronize(b));
      float ms; cudaEventElapsedTime(&ms, a, b);
      ts[g] = ms / 20;
      cudaEventDestroy(a); cudaEventDestroy(b);
    }
    std::sort(ts, ts + 5);
    printf("%s: min %.3f  med %.3f ms\n", nm, ts[0], ts[2]);
    return ts[0];
  };
  float tq = bench("dq_ws", launch_dq);
  float tk = bench("dk_ws", launch_dk);
  float tv = bench("dv_ws", launch_dv);
  printf("total(min): %.3f ms\n", tq + tk + tv);
  float t2w = bench("dvdk2_ws", [&] { launch_dvdk2(true); });
  float t2n = bench("dvdk2_nows", [&] { launch_dvdk2(false); });
  printf("dk+dv: %.3f  vs fused ws: %.3f  nows: %.3f ms\n", tk + tv, t2w, t2n);
  if (getenv("S3B_DVDK3"))  // c21 rejected (+174%): kept for reference, opt-in only
    printf("dvdk3_split: %.3f ms\n", bench("dvdk3_split", [&] { launch_dvdk3(); }));
  float t421 = bench("dvdk64_421", [&] { launch_dvdk64(false); });
  float t221 = bench("dvdk64_221", [&] { launch_dvdk64(true); });
  printf("fused kv64: (4,2,1)@256t: %.3f  (2,2,1)@128t: %.3f ms\n", t421, t221);
#else
  std::vector<float> gQ(S * D), gK(S * D), gV(S * D), gQr(S * D);
  CK(cudaMemcpy(gQ.data(), ddQ, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gK.data(), ddK, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gV.data(), ddV, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gQr.data(), ddQref, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  { size_t bad = 0;
    for (size_t i = 0; i < gQ.size(); ++i)
      if (std::memcmp(&gQ[i], &gQr[i], 4) != 0) ++bad;
    printf("dQ bitwise vs ref kernel: %s (%zu/%zu differ)\n",
           bad ? "MISMATCH" : "MATCH", bad, gQ.size()); }
  auto rel = [&](const std::vector<float>& g, const std::vector<double>& r, const char* nm) {
    double num = 0, den = 0;
    for (int i = 0; i < S * D; ++i) { num += (g[i] - r[i]) * (g[i] - r[i]); den += r[i] * r[i]; }
    printf("%s rel-L2 vs fp64-dequant ref: %.4f\n", nm, std::sqrt(num / den));
  };
  rel(gQ, rQ, "dQ"); rel(gK, rK, "dK"); rel(gV, rV, "dV");
  for (int ws = 1; ws >= 0; --ws) {
    CK(cudaMemset(ddK2, 0xFF, (size_t)H * S * D * 4)); CK(cudaMemset(ddV2, 0xFF, (size_t)H * S * D * 4));
    launch_dvdk2(ws == 1); CK(cudaDeviceSynchronize());
    std::vector<float> g2k(S * D), g2v(S * D);
    CK(cudaMemcpy(g2k.data(), ddK2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g2v.data(), ddV2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    rel(g2k, rK, ws ? "dK dvdk2_ws" : "dK dvdk2_nows");
    rel(g2v, rV, ws ? "dV dvdk2_ws" : "dV dvdk2_nows");
  }
  for (int v = 1; v >= 0; --v) {
    CK(cudaMemset(ddK2, 0xFF, (size_t)H * S * D * 4)); CK(cudaMemset(ddV2, 0xFF, (size_t)H * S * D * 4));
    launch_dvdk64(v == 1); CK(cudaDeviceSynchronize());
    std::vector<float> g2k(S * D), g2v(S * D);
    CK(cudaMemcpy(g2k.data(), ddK2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g2v.data(), ddV2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    rel(g2k, rK, v ? "dK dvdk64_221" : "dK dvdk64_421");
    rel(g2v, rV, v ? "dV dvdk64_221" : "dV dvdk64_421");
  }
  // dvdk3 split prototype: rel-L2 vs fp64 ref + bitwise vs dvdk2_nows
  {
    CK(cudaMemset(ddK2, 0xFF, (size_t)H * S * D * 4)); CK(cudaMemset(ddV2, 0xFF, (size_t)H * S * D * 4));
    launch_dvdk2(false); CK(cudaDeviceSynchronize());
    std::vector<float> bk(S * D), bv(S * D);
    CK(cudaMemcpy(bk.data(), ddK2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bv.data(), ddV2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(ddK2, 0xFF, (size_t)H * S * D * 4)); CK(cudaMemset(ddV2, 0xFF, (size_t)H * S * D * 4));
    launch_dvdk3(); CK(cudaDeviceSynchronize());
    std::vector<float> g3k(S * D), g3v(S * D);
    CK(cudaMemcpy(g3k.data(), ddK2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g3v.data(), ddV2, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
    rel(g3k, rK, "dK dvdk3"); rel(g3v, rV, "dV dvdk3");
    size_t badk = 0, badv = 0;
    for (size_t i = 0; i < bk.size(); ++i) {
      if (std::memcmp(&bk[i], &g3k[i], 4) != 0) ++badk;
      if (std::memcmp(&bv[i], &g3v[i], 4) != 0) ++badv;
    }
    printf("dvdk3 bitwise vs dvdk2_nows: dK %s (%zu/%zu)  dV %s (%zu/%zu)\n",
           badk ? "MISMATCH" : "MATCH", badk, bk.size(),
           badv ? "MISMATCH" : "MATCH", badv, bv.size());
    if (getenv("S3B_DEBUG")) {
      for (int off : {0, 1, 2, 63, 64, 4096, 8192}) {
        printf("  [%5d] dV ref %9.4f  dvdk3 %9.4f   dK ref %9.4f  dvdk3 %9.4f\n",
               off, bv[off], g3v[off], bk[off], g3k[off]);
      }
    }
  }
#endif
  return 0;
}
