// S3B-WS dq e2e + bench. Single head S=256 correctness vs fp64-dequant ref;
// -DS3B_BENCH: H=32 S=16896 timing. Build (from /root/fa-blackwell):
//   nvcc -std=c++17 -O2 [-DS3B_BENCH] -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include kernels/attn_sm120/tests/s3b_dq_ws.cu \
//     -o kernels/attn_sm120/tests/s3b_dq_ws
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "s3b_dq_ws_kernel.cuh"

#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

using s3bws::Element;
#ifndef S3B_BENCH
static const int H = 1, S = 256, D = 128;
#else
static const int H = 32, S = 16896, D = 128;
#endif
static const int MT = S / 128, NT = S / 64;

static int e8m0_byte(double amax) {
  if (amax <= 0) return 1;
  double a = std::max(amax, std::ldexp(1.0, -126));
  return std::max(1, std::min(254, (int)std::ceil(std::log2(a)) - 8 + 127));
}

int main() {
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> Q((size_t)H * S * D), K((size_t)H * S * D), V((size_t)H * S * D), dO((size_t)H * S * D);
  for (auto& v : Q) v = nd(rng); for (auto& v : K) v = nd(rng);
  for (auto& v : V) v = nd(rng); for (auto& v : dO) v = nd(rng);

  // quantize (natural [S,D] + transposed [D,S]) per head
  std::vector<uint8_t> qQ(H * S * D), qK(H * S * D), qV(H * S * D), qD(H * S * D), qKt(H * S * D);
  std::vector<std::vector<int>> eQ(H), eK(H), eV(H), eD(H), eKt(H);   // sf bytes, pre-scatter
  auto quant_axis = [&](const float* x, int R, int C, uint8_t* q, std::vector<int>& e) {
    e.assign(R * (C / 32), 0);
    for (int r = 0; r < R; ++r)
      for (int kb = 0; kb < C / 32; ++kb) {
        double amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, (double)std::fabs(x[(size_t)r * C + kb * 32 + j]));
        int b = e8m0_byte(amax);
        double s = std::ldexp(1.0, b - 127);
        for (int j = 0; j < 32; ++j)
          q[(size_t)r * C + kb * 32 + j] = cutlass::float_e4m3_t(float(x[(size_t)r * C + kb * 32 + j] / s)).storage;
        e[r * (C / 32) + kb] = b;
      }
  };
  for (int h = 0; h < H; ++h) {
    const float* Qh = Q.data() + (size_t)h * S * D; const float* Kh = K.data() + (size_t)h * S * D;
    const float* Vh = V.data() + (size_t)h * S * D; const float* Dh = dO.data() + (size_t)h * S * D;
    quant_axis(Qh, S, D, qQ.data() + (size_t)h * S * D, eQ[h]);
    quant_axis(Kh, S, D, qK.data() + (size_t)h * S * D, eK[h]);
    quant_axis(Vh, S, D, qV.data() + (size_t)h * S * D, eV[h]);
    quant_axis(Dh, S, D, qD.data() + (size_t)h * S * D, eD[h]);
    // Kt [D, S]: scales along S
    std::vector<float> KtH(D * S);
    for (int d = 0; d < D; ++d) for (int s = 0; s < S; ++s) KtH[(size_t)d * S + s] = Kh[(size_t)s * D + d];
    quant_axis(KtH.data(), D, S, qKt.data() + (size_t)h * S * D, eKt[h]);
  }

  // SF gmem layouts (canonical cutlass tiled)
  auto layoutSFK = s3bws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bws::kBlockN), D, H));
  auto layoutSFKt = s3bws::BlkSF::tile_atom_to_shape_SFB(make_shape(int(s3bws::kBlockM), D, S, H));
  auto layoutSFQ = s3bws::BlkSF::tile_atom_to_shape_SFA(make_shape(S, int(s3bws::kBlockN), D, H));
  std::vector<uint8_t> sfK(cosize(layoutSFK), 0), sfV(cosize(layoutSFK), 0),
      sfKt(cosize(layoutSFKt), 0), sfQ(cosize(layoutSFQ), 0), sfD(cosize(layoutSFQ), 0);
  for (int h = 0; h < H; ++h) {
    for (int n = 0; n < S; ++n)
      for (int kb = 0; kb < D / 32; ++kb) {
        sfK[layoutSFK(make_coord(n, kb * 32, h))] = uint8_t(eK[h][n * (D / 32) + kb]);
        sfV[layoutSFK(make_coord(n, kb * 32, h))] = uint8_t(eV[h][n * (D / 32) + kb]);
        sfQ[layoutSFQ(make_coord(n, kb * 32, h))] = uint8_t(eQ[h][n * (D / 32) + kb]);
        sfD[layoutSFQ(make_coord(n, kb * 32, h))] = uint8_t(eD[h][n * (D / 32) + kb]);
      }
    for (int d = 0; d < D; ++d)
      for (int kb = 0; kb < S / 32; ++kb)
        sfKt[layoutSFKt(make_coord(d, kb * 32, h))] = uint8_t(eKt[h][d * (S / 32) + kb]);
  }

  auto up = [&](const std::vector<uint8_t>& v) {
    uint8_t* p; CK(cudaMalloc(&p, v.size())); CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice)); return p;
  };
  uint8_t *dQd = up(qQ), *dKd = up(qK), *dVd = up(qV), *dDd = up(qD), *dKtd = up(qKt);
  uint8_t *dsfK = up(sfK), *dsfV = up(sfV), *dsfKt = up(sfKt), *dsfQ = up(sfQ), *dsfD = up(sfD);

  // lse / delta
  std::vector<float> lsef(H * S), dltf(H * S);
  double sm = 1.0 / std::sqrt(128.0);
#ifndef S3B_BENCH
  std::vector<double> Ss(S * S), P(S * S), Od(S * D), lse(S), delta(S);
  auto deqN = [&](const std::vector<uint8_t>& q, const std::vector<int>& e, int r, int c) {
    return double(float(cutlass::float_e4m3_t::bitcast(q[r * D + c]))) * std::ldexp(1.0, e[r * (D / 32) + c / 32] - 127);
  };
  for (int i = 0; i < S; ++i) {
    double mx = -1e300;
    for (int j = 0; j < S; ++j) {
      double s = 0;
      for (int d = 0; d < D; ++d) s += deqN(qQ, eQ[0], i, d) * deqN(qK, eK[0], j, d);
      Ss[i * S + j] = s * sm; mx = std::max(mx, s * sm);
    }
    double sum = 0;
    for (int j = 0; j < S; ++j) { P[i * S + j] = std::exp(Ss[i * S + j] - mx); sum += P[i * S + j]; }
    lse[i] = mx + std::log(sum);
    for (int j = 0; j < S; ++j) P[i * S + j] /= sum;
  }
  for (int i = 0; i < S; ++i)
    for (int d = 0; d < D; ++d) {
      double o = 0;
      for (int j = 0; j < S; ++j) o += P[i * S + j] * deqN(qV, eV[0], j, d);
      Od[i * D + d] = o;
    }
  for (int i = 0; i < S; ++i) {
    double dl = 0;
    for (int d = 0; d < D; ++d) dl += deqN(qD, eD[0], i, d) * Od[i * D + d];
    delta[i] = dl;
  }
  for (int i = 0; i < S; ++i) { lsef[i] = float(lse[i]); dltf[i] = float(delta[i]); }
#else
  { std::mt19937 r2(1); std::normal_distribution<float> n2(3.f, 1.5f);
    for (auto& v : lsef) v = n2(r2); for (auto& v : dltf) v = n2(r2); }
#endif
  float *dLse, *dDlt; CK(cudaMalloc(&dLse, H * S * 4)); CK(cudaMalloc(&dDlt, H * S * 4));
  CK(cudaMemcpy(dLse, lsef.data(), H * S * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dDlt, dltf.data(), H * S * 4, cudaMemcpyHostToDevice));
  float* ddQ; CK(cudaMalloc(&ddQ, (size_t)H * S * D * 4));

  s3bws::ParamsDq p{};
  {
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(dKd)),
        make_layout(make_shape(S, D, H), make_stride(D, _1{}, S * D)));
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(dVd)), mK.layout());
    Tensor mKt = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(dKtd)),
        make_layout(make_shape(D, S, H), make_stride(S, _1{}, D * S)));
    Tensor mSFK = make_tensor(make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfK)), layoutSFK);
    Tensor mSFV = make_tensor(make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfV)), layoutSFK);
    Tensor mSFKt = make_tensor(make_gmem_ptr(reinterpret_cast<s3bws::ElementSF const*>(dsfKt)), layoutSFKt);
    p.tma_k = make_tma_copy(SM90_TMA_LOAD{}, mK, s3bws::SmemLayoutK{}(_, _, _0{}),
                            make_shape(Int<s3bws::kBlockN>{}, Int<D>{}), _1{});
    p.tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, s3bws::SmemLayoutK{}(_, _, _0{}),
                            make_shape(Int<s3bws::kBlockN>{}, Int<D>{}), _1{});
    p.tma_kt = make_tma_copy(SM90_TMA_LOAD{}, mKt, s3bws::SmemLayoutKt{}(_, _, _0{}),
                             make_shape(Int<D>{}, Int<s3bws::kBlockN>{}), _1{});
    p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, s3bws::SmemLayoutSFT{},
                                        make_shape(Int<128>{}, Int<D>{}), _1{});
    p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, s3bws::SmemLayoutSFT{},
                                        make_shape(Int<128>{}, Int<D>{}), _1{});
    p.tma_sfkt = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFKt, s3bws::SmemLayoutSFT{},
                                         make_shape(Int<D>{}, Int<128>{}), _1{});
  }
  p.layout_sfk = layoutSFK; p.layout_sfkt = layoutSFKt;
  p.Q = dQd; p.D = dDd; p.sfQ = dsfQ; p.sfD = dsfD;
  p.lse = dLse; p.delta = dDlt; p.dQ = ddQ;
  p.S = S; p.H = H; p.sm_scale = float(sm);
#ifndef S3B_BENCH
  float* dDbg; CK(cudaMalloc(&dDbg, 64 * 4)); CK(cudaMemset(dDbg, 0, 64 * 4));
  p.dbg = dDbg;
#endif

  // NOTE: resident sfQ/sfD loaded in-kernel as flat 512B tiles indexed ((h*MT+m))*512 —
  // verify that tile_atom_to_shape_SFA tile order matches (mt*TK+kt, in-tile formula).
  {
    bool ok = true;
    for (int m2 = 0; m2 < S && ok; ++m2)
      for (int kb = 0; kb < D / 32 && ok; ++kb) {
        size_t expect = ((size_t)(m2 / 128) * (D / 128) + (kb / 4)) * 512 + 16 * (m2 % 32) + 4 * ((m2 % 128) / 32) + (kb % 4);
        if (size_t(layoutSFQ(make_coord(m2, kb * 32, 0))) != expect) ok = false;
      }
    printf("SF flat-tile order match: %s\n", ok ? "YES" : "NO (kernel resident-SF indexing would be wrong)");
  }

  int smem = int(sizeof(s3bws::SharedStorageDq));
  printf("smem: %d bytes\n", smem);
  CK(cudaFuncSetAttribute(s3bws::dq_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  s3bws::dq_ws_kernel<<<dim3(MT, H), s3bws::kNThreads, smem>>>(p);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

#ifdef S3B_BENCH
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < 5; ++i) s3bws::dq_ws_kernel<<<dim3(MT, H), s3bws::kNThreads, smem>>>(p);
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  printf("dq_ws: %.1f ms\n", ms / 5);
#else
  std::vector<float> gQ(S * D);
  CK(cudaMemcpy(gQ.data(), ddQ, S * D * 4, cudaMemcpyDeviceToHost));
  std::vector<double> rQ(S * D, 0);
  std::vector<double> dS(S * S);
  for (int i = 0; i < S; ++i)
    for (int j = 0; j < S; ++j) {
      double dp = 0;
      for (int d = 0; d < D; ++d) dp += deqN(qD, eD[0], i, d) * deqN(qV, eV[0], j, d);
      dS[i * S + j] = P[i * S + j] * (dp - delta[i]);
    }
  for (int i = 0; i < S; ++i)
    for (int d = 0; d < D; ++d) {
      double dq = 0;
      for (int j = 0; j < S; ++j) dq += dS[i * S + j] * deqN(qK, eK[0], j, d);
      rQ[i * D + d] = dq;
    }
  { float hd[64]; CK(cudaMemcpy(hd, dDbg, 64 * 4, cudaMemcpyDeviceToHost));
    printf("dbg accS00=%.4f accS15=%.4f Q0=%.4f K0=%.4f SFQ0=%.0f SFK0=%.0f lse=%.3f dlt=%.3f\n",
           hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7]);
    { printf("tOrSFDS(0) right after gather = %.0f\n", hd[22]); printf("coord0=(%.0f,%.0f) szSFDS=%.0f szCoord=%.0f SFDS[0]=%.0f\n",
           hd[17], hd[18], hd[19], hd[20], hd[21]); }
  printf("dbg dS00=%.4f sDS00=%.4f sDSrc=%.4f SFDS0=%.0f | accQ0=%.4f tOrDS0=%.4f tOrKt0=%.4f SFDSf=%.0f SFKtf=%.0f\n",
           hd[8], hd[9], hd[10], hd[11], hd[12], hd[13], hd[14], hd[15], hd[16]); }
  for (int i = 0; i < 4; ++i)
    printf("row%d got: %.4f %.4f  ref: %.4f %.4f\n", i, gQ[i*D], gQ[i*D+1], rQ[i*D], rQ[i*D+1]);
  for (int i = 0; i < 4; ++i)
    printf("row%d got[64]: %.4f %.4f  ref: %.4f %.4f\n", 64+i, gQ[(64+i)*D], gQ[(64+i)*D+1], rQ[(64+i)*D], rQ[(64+i)*D+1]);
  for (int cb = 0; cb < 4; ++cb) {
    double n2 = 0, d2 = 0;
    for (int i = 0; i < S; ++i) for (int c = cb * 32; c < cb * 32 + 32; ++c) {
      n2 += (gQ[i * D + c] - rQ[i * D + c]) * (gQ[i * D + c] - rQ[i * D + c]);
      d2 += rQ[i * D + c] * rQ[i * D + c];
    }
    printf("cols %3d-%3d rel-L2: %.4f\n", cb * 32, cb * 32 + 31, std::sqrt(n2 / d2));
  }
  for (int blk = 0; blk < S / 32; ++blk) {
    double n2 = 0, d2 = 0;
    for (int i = blk * 32 * D; i < (blk + 1) * 32 * D; ++i) { n2 += (gQ[i] - rQ[i]) * (gQ[i] - rQ[i]); d2 += rQ[i] * rQ[i]; }
    printf("rows %3d-%3d rel-L2: %.4f\n", blk * 32, blk * 32 + 31, std::sqrt(n2 / d2));
  }
  double num = 0, den = 0;
  for (int i = 0; i < S * D; ++i) { num += (gQ[i] - rQ[i]) * (gQ[i] - rQ[i]); den += rQ[i] * rQ[i]; }
  printf("dQ rel-L2 vs fp64-dequant ref: %.4f\n", std::sqrt(num / den));
#endif
  return 0;
}
