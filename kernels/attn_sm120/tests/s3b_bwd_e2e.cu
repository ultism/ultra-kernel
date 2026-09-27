// S3B bwd e2e v2: single head, S = S3B_S (default 256), d=128, non-causal.
// Host: mxfp8-quantize Q/K/V/dO natural + transposed, lse/delta + fp64 reference
// grads on dequantized inputs; dumps Pt/DSt tile (block n=0, m=0) for bisect.
// Build (from /root/fa-blackwell):
//   nvcc -std=c++17 -O2 [-DS3B_S=128] [-DS3B_BENCH] -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include kernels/attn_sm120/tests/s3b_bwd_e2e.cu -o /tmp/s3b
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "s3b_bwd_kernel.cuh"
#include "s3b64_kernel.cuh"

#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

using s3b::Element;
#ifndef S3B_BENCH
#ifndef S3B_S
#define S3B_S 256
#endif
static const int H = 1, S = S3B_S, D = 128;
#else
static const int H = 32, S = 16896, D = 128;
#endif
static const int MT = S / 128;

static int e8m0_byte(double amax) {
  if (amax <= 0) return 1;
  double a = std::max(amax, std::ldexp(1.0, -126));
  return std::max(1, std::min(254, (int)std::ceil(std::log2(a)) - 8 + 127));
}
// natural quant: [S,128] -> data + sf tiles [MT][512] (scales along d)
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
// transposed quant: xt[d,s]=x[s,d]; scales along s. sf tiles [MT][512], (mn=d, kb=sblock)
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
  float *dLse, *dDlt; CK(cudaMalloc(&dLse, H * S * 4)); CK(cudaMalloc(&dDlt, H * S * 4));
  CK(cudaMemcpy(dLse, lsef.data(), H * S * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dDlt, dltf.data(), H * S * 4, cudaMemcpyHostToDevice));
  float *dDQ, *dDK, *dDV;
  CK(cudaMalloc(&dDQ, (size_t)H * S * D * 4)); CK(cudaMalloc(&dDK, (size_t)H * S * D * 4)); CK(cudaMalloc(&dDV, (size_t)H * S * D * 4));

  s3b::ParamsBwd p{};
  p.Q = up(qQ); p.K = up(qK); p.V = up(qV); p.D = up(qD);
  p.Qt = up(qQt); p.Kt = up(qKt); p.Dt = up(qDt);
  p.sfQ = up(sfQ); p.sfK = up(sfK); p.sfV = up(sfV); p.sfD = up(sfD);
  p.sfQt = up(sfQt); p.sfKt = up(sfKt); p.sfDt = up(sfDt);
  p.lse = dLse; p.delta = dDlt; p.dQ = dDQ; p.dK = dDK; p.dV = dDV;
  p.S = S; p.H = H; p.sm_scale = float(sm);
#ifndef S3B_BENCH
  float* dDbg; CK(cudaMalloc(&dDbg, 2 * 16384 * 4)); CK(cudaMemset(dDbg, 0, 2 * 16384 * 4));
  p.dbg = dDbg;
#endif

  int smem = int(sizeof(s3b::SharedStorageBwd));
  printf("smem: %d bytes\n", smem);
  CK(cudaFuncSetAttribute(s3b::dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(s3b::dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  s3b64::ParamsBwd64 p64{};
  p64.Q = p.Q; p64.K = p.K; p64.V = p.V; p64.D = p.D; p64.Qt = p.Qt; p64.Kt = p.Kt; p64.Dt = p.Dt;
  p64.sfQ = p.sfQ; p64.sfK = p.sfK; p64.sfV = p.sfV; p64.sfD = p.sfD;
  p64.sfQt = p.sfQt; p64.sfKt = p.sfKt; p64.sfDt = p.sfDt;
  p64.lse = p.lse; p64.delta = p.delta; p64.dQ = p.dQ; p64.dK = p.dK; p64.dV = p.dV;
  p64.S = p.S; p64.H = p.H; p64.sm_scale = p.sm_scale;
  int smem64v = int(sizeof(s3b64::SharedStorageDV));
  int smem64k = int(sizeof(s3b64::SharedStorageDK));
  printf("smem64 dv: %d dk: %d\n", smem64v, smem64k);
  CK(cudaFuncSetAttribute(s3b64::dv64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64v));
  CK(cudaFuncSetAttribute(s3b64::dk64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64k));
  s3b::dkdv_kernel<<<dim3(MT, H), s3b::kNThreads, smem>>>(p);   // legacy, output overwritten below
  CK(cudaGetLastError());
  s3b64::dv64_kernel<<<dim3(S / 64, H), s3b64::kNThreads, smem64v>>>(p64);
  CK(cudaGetLastError());
  s3b64::dk64_kernel<<<dim3(S / 64, H), s3b64::kNThreads, smem64k>>>(p64);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  s3b::dq_kernel<<<dim3(MT, H), s3b::kNThreads, smem>>>(p);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

#ifdef S3B_BENCH
  auto bench = [&](const char* nm, auto kern) {
    kern<<<dim3(MT, H), s3b::kNThreads, smem>>>(p); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) kern<<<dim3(MT, H), s3b::kNThreads, smem>>>(p);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("%s: %.1f ms\n", nm, ms / 5);
  };
  { s3b64::ParamsBwd64 p64{};
    p64.Q = p.Q; p64.K = p.K; p64.V = p.V; p64.D = p.D; p64.Qt = p.Qt; p64.Kt = p.Kt; p64.Dt = p.Dt;
    p64.sfQ = p.sfQ; p64.sfK = p.sfK; p64.sfV = p.sfV; p64.sfD = p.sfD;
    p64.sfQt = p.sfQt; p64.sfKt = p.sfKt; p64.sfDt = p.sfDt;
    p64.lse = p.lse; p64.delta = p.delta; p64.dQ = p.dQ; p64.dK = p.dK; p64.dV = p.dV;
    p64.S = p.S; p64.H = p.H; p64.sm_scale = p.sm_scale;
    int smem64v = int(sizeof(s3b64::SharedStorageDV));
    int smem64k = int(sizeof(s3b64::SharedStorageDK));
    CK(cudaFuncSetAttribute(s3b64::dv64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64v));
    CK(cudaFuncSetAttribute(s3b64::dk64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64k));
    auto bench64 = [&](const char* nm, auto kern, int sm) {
      kern<<<dim3(S / 64, H), s3b64::kNThreads, sm>>>(p64); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a);
      for (int i = 0; i < 5; ++i) kern<<<dim3(S / 64, H), s3b64::kNThreads, sm>>>(p64);
      cudaEventRecord(b); CK(cudaEventSynchronize(b));
      float ms; cudaEventElapsedTime(&ms, a, b);
      printf("%s: %.1f ms\n", nm, ms / 5);
    };
    bench64("dv64", s3b64::dv64_kernel, smem64v);
    bench64("dk64", s3b64::dk64_kernel, smem64k);
  }
  bench("dkdv", s3b::dkdv_kernel);
  bench("dq  ", s3b::dq_kernel);
  return 0;
#else
  // Pt/DSt tile dump check (block n=0, m=0)
  {
    std::vector<float> dump(2 * 16384);
    CK(cudaMemcpy(dump.data(), dDbg, 2 * 16384 * 4, cudaMemcpyDeviceToHost));
    double nP = 0, dP2 = 0, nS = 0, dS2 = 0; int shown = 0;
    for (int kv = 0; kv < 128; ++kv)
      for (int q = 0; q < 128; ++q) {
        double rp = P[q * S + kv], rs = dS[q * S + kv];
        double gp = dump[kv * 128 + q], gs = dump[16384 + kv * 128 + q];
        nP += (gp - rp) * (gp - rp); dP2 += rp * rp;
        nS += (gs - rs) * (gs - rs); dS2 += rs * rs;
        if (std::fabs(gp - rp) > 0.3 * std::fabs(rp) + 1e-3 && shown++ < 5)
          printf("Pt(kv=%d,q=%d) got=%.5f ref=%.5f\n", kv, q, gp, rp);
      }
    printf("Pt tile rel-L2: %.4f | DSt tile rel-L2: %.4f\n", std::sqrt(nP / dP2), std::sqrt(nS / dS2));
  }
  std::vector<float> gQ(S * D), gK(S * D), gV(S * D);
  CK(cudaMemcpy(gQ.data(), dDQ, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gK.data(), dDK, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gV.data(), dDV, (size_t)S * D * 4, cudaMemcpyDeviceToHost));
  auto rel = [&](const std::vector<float>& g, const std::vector<double>& r, const char* nm) {
    double num = 0, den = 0;
    for (int i = 0; i < S * D; ++i) { num += (g[i] - r[i]) * (g[i] - r[i]); den += r[i] * r[i]; }
    printf("%s rel-L2 vs fp64-dequant ref: %.4f\n", nm, std::sqrt(num / den));
  };
  rel(gQ, rQ, "dQ"); rel(gK, rK, "dK"); rel(gV, rV, "dV");
  return 0;
#endif
}
