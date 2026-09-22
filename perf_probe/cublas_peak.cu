// cuBLAS achievable-throughput probe for GB202 (RTX 5090, sm_120).
// Measures real GEMM TFLOP/s on three paths:
//   1) bf16 x bf16 -> fp32 compute (cublasGemmEx, CUBLAS_COMPUTE_32F)   == fa2's HMMA.F32 path
//   2) fp8 e4m3 x fp8 e4m3 -> fp32 compute (cublasLt, CUBLAS_COMPUTE_32F) == plain FP8 path
//   3) bf16 x bf16 -> fp16 compute... skipped: cublas fp16-acc path uses F16 acc (not the point).
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { printf("cublas err %d @%d\n", (int)(x), __LINE__); return 1; } } while (0)
#define CKC(x) do { cudaError_t e = (x); if (e) { printf("cuda err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static float bench(void (*fn)(void*), void* ctx, int iters) {
  fn(ctx);
  CKC(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) fn(ctx);
  cudaEventRecord(e1);
  CKC(cudaEventSynchronize(e1));
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms / iters;
}

struct BF16Ctx {
  cublasHandle_t h; __nv_bfloat16 *A, *B, *C; int N;
  float alpha = 1.f, beta = 0.f;
};
static void run_bf16(void* p) {
  BF16Ctx* c = (BF16Ctx*)p;
  cublasGemmEx(c->h, CUBLAS_OP_N, CUBLAS_OP_N, c->N, c->N, c->N, &c->alpha,
               c->A, CUDA_R_16BF, c->N, c->B, CUDA_R_16BF, c->N, &c->beta,
               c->C, CUDA_R_16BF, c->N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

struct FP8Ctx {
  cublasLtHandle_t h; cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t ad, bd, cd; void *A, *B, *C, *ws; size_t wsSize; int N;
  cublasLtMatmulAlgo_t algo;
  float alpha = 1.f, beta = 0.f;
};
static void run_fp8(void* p) {
  FP8Ctx* c = (FP8Ctx*)p;
  cublasLtMatmul(c->h, c->op, &c->alpha, c->A, c->ad, c->B, c->bd, &c->beta,
                 c->C, c->cd, c->C, c->cd, &c->algo, c->ws, c->wsSize, 0);
}

int main() {
  const int N = 8192;
  const int ITERS = 50;
  double tflop = 2.0 * N * N * N / 1e12;

  // ---- 1) bf16 -> fp32 compute ----
  cublasHandle_t h; CK(cublasCreate(&h));
  CK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
  __nv_bfloat16 *A, *B, *C;
  CKC(cudaMalloc(&A, sizeof(__nv_bfloat16) * N * N));
  CKC(cudaMalloc(&B, sizeof(__nv_bfloat16) * N * N));
  CKC(cudaMalloc(&C, sizeof(__nv_bfloat16) * N * N));
  CKC(cudaMemset(A, 0x3c, sizeof(__nv_bfloat16) * N * N));
  CKC(cudaMemset(B, 0x3c, sizeof(__nv_bfloat16) * N * N));
  BF16Ctx bc{h, A, B, C, N};
  float ms = bench(run_bf16, &bc, ITERS);
  printf("bf16 x bf16 -> f32 compute (cublasGemmEx): %8.1f TFLOP/s  (%.3f ms)\n", tflop / (ms / 1e3), ms);
  cublasDestroy(h); cudaFree(A); cudaFree(B); cudaFree(C);

  // ---- 2) fp8 e4m3 -> fp32 compute (cublasLt) ----
  FP8Ctx fc; fc.N = N;
  CK(cublasLtCreate(&fc.h));
  CK(cublasLtMatmulDescCreate(&fc.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opN = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(fc.op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  CK(cublasLtMatmulDescSetAttribute(fc.op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  CK(cublasLtMatrixLayoutCreate(&fc.ad, CUDA_R_8F_E4M3, N, N, N));
  CK(cublasLtMatrixLayoutCreate(&fc.bd, CUDA_R_8F_E4M3, N, N, N));
  CK(cublasLtMatrixLayoutCreate(&fc.cd, CUDA_R_16BF, N, N, N));
  fc.wsSize = 64 << 20;
  CKC(cudaMalloc(&fc.ws, fc.wsSize));
  CKC(cudaMalloc(&fc.A, (size_t)N * N));
  CKC(cudaMalloc(&fc.B, (size_t)N * N));
  CKC(cudaMalloc(&fc.C, sizeof(__nv_bfloat16) * N * N));
  CKC(cudaMemset(fc.A, 0x30, (size_t)N * N));
  CKC(cudaMemset(fc.B, 0x30, (size_t)N * N));
  // heuristic check: is there an FP8 algo at all on this arch?
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &fc.wsSize, sizeof(fc.wsSize)));
  cublasLtMatmulHeuristicResult_t heur;
  int nres = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(fc.h, fc.op, fc.ad, fc.bd, fc.cd, fc.cd, pref, 1, &heur, &nres));
  if (nres == 0) {
    printf("fp8 e4m3: no cublasLt algo on this arch\n");
  } else {
    fc.algo = heur.algo;
    ms = bench(run_fp8, &fc, ITERS);
    printf("fp8 e4m3 -> f32 compute (cublasLt):       %8.1f TFLOP/s  (%.3f ms)\n", tflop / (ms / 1e3), ms);
  }
  return 0;
}
