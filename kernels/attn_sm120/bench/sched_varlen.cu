// Does a REAL dynamic scheduler beat the HW block scheduler on VARIABLE-LENGTH
// (batched / long-tail) prefill load? sched_ab.cu already showed dynamic ≈ HW on
// uniform/single-causal (both grab work greedily in index order). The only lever a
// software scheduler has that HW lacks is REORDERING the work -- handing out the
// heaviest tiles first (LPT), which HW (fixed blockIdx dispatch order) cannot do.
// So this isolates that: per-tile cost is a host-built array (all schedulers do the
// SAME total work, differing only in ASSIGNMENT), and we add a 4th scheduler --
// dynamic with a host-sorted heaviest-first order -- on top of the three from
// sched_ab.cu. If dynamic-LPT beats HW on the realistic varlen/tail distributions,
// a real (host-planned) dynamic scheduler is worth building into the kernel; if it
// only ties, it isn't (for single-GPU dense; batch indexing aside).
//
// Build: nvcc -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a \
//        bench/sched_varlen.cu -o bench/sched_varlen
// Run:   ./bench/sched_varlen [waves] [seed]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); return 1; } } while(0)

static constexpr int kThreads = 256;

// Per-tile compute: burn d_iters[t] FMAs, block-reduce so nothing is DCE'd.
__device__ __forceinline__ void do_tile(int t, const int* d_iters, float* out) {
  int iters = d_iters[t];
  float a = 1.0f + 1e-3f * threadIdx.x, b = 1.0000003f;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) { a = a * b + 0.5f; b = b * 1.0000003f + a * 1e-9f; }
  __shared__ float s[kThreads];
  s[threadIdx.x] = a + b; __syncthreads();
  for (int o = kThreads >> 1; o > 0; o >>= 1) { if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o]; __syncthreads(); }
  if (threadIdx.x == 0) out[t] = s[0];
}

__global__ void k_nonpersistent(int n, const int* it, float* out) {            // grid=n, HW scheduler
  if (blockIdx.x < n) do_tile(blockIdx.x, it, out);
}
__global__ void k_static(int n, const int* it, float* out) {                   // grid=num_sm, grid-stride
  for (int t = blockIdx.x; t < n; t += gridDim.x) do_tile(t, it, out);
}
__global__ void k_dynamic(int n, const int* it, int* ctr, float* out) {        // atomic, index order
  __shared__ int s_t;
  while (true) { if (threadIdx.x == 0) s_t = atomicAdd(ctr, 1); __syncthreads(); int t = s_t; __syncthreads();
    if (t >= n) break; do_tile(t, it, out); }
}
__global__ void k_dynamic_lpt(int n, const int* it, const int* order, int* ctr, float* out) {  // atomic, heaviest-first
  __shared__ int s_c;
  while (true) { if (threadIdx.x == 0) s_c = atomicAdd(ctr, 1); __syncthreads(); int c = s_c; __syncthreads();
    if (c >= n) break; do_tile(order[c], it, out); }
}

template <typename L> static float time_min(L launch, int reps, int warmup) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < warmup; ++i) launch(); cudaDeviceSynchronize();
  float best = 1e30f;
  for (int i = 0; i < reps; ++i) { cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms); }
  cudaEventDestroy(a); cudaEventDestroy(b); return best;
}

// Build per-tile iters for a distribution. Returns the tile-cost list (in "unit" multiples).
// Model: a batch of requests; request with q-block-count Q contributes Q causal tiles of cost
// base*(i+1), i in [0,Q). Distributions differ in how the per-request Q is drawn.
static std::vector<int> gen(const char* dist, int target_tiles, int base, std::mt19937& rng) {
  std::vector<int> it;
  auto add_request = [&](int Q) { for (int i = 0; i < Q; ++i) it.push_back(base * (i + 1)); };
  if (!strcmp(dist, "uniform")) {                       // all tiles equal (sanity)
    for (int i = 0; i < target_tiles; ++i) it.push_back(base * 8);
  } else if (!strcmp(dist, "causal1")) {                // ONE long request: pure triangular
    add_request(target_tiles);
  } else if (!strcmp(dist, "varlen")) {                 // serving-like: mixed request lengths
    std::vector<int> menu = {1, 2, 4, 8, 16, 32};       // q-block counts
    std::vector<int> wt    = {30, 25, 20, 12, 8, 5};    // many short, few long
    std::discrete_distribution<int> pick(wt.begin(), wt.end());
    while ((int)it.size() < target_tiles) add_request(menu[pick(rng)]);
  } else if (!strcmp(dist, "tail")) {                   // adversarial: a few huge among many tiny
    std::uniform_int_distribution<int> coin(0, 99);
    while ((int)it.size() < target_tiles) add_request(coin(rng) < 90 ? 1 : 48);
  }
  return it;
}

int main(int argc, char** argv) {
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int num_sm = prop.multiProcessorCount;
  const int base = 1500;
  const float waves = (argc > 1) ? atof(argv[1]) : 8.0f;
  unsigned seed = (argc > 2) ? atoi(argv[2]) : 1234;
  const int reps = 50, warmup = 10;
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_static, kThreads, kThreads * sizeof(float)));
  int pgrid = num_sm * occ;
  printf("GPU %s SMs=%d occ=%d/SM persistent_grid=%d | target ~%.0f waves, base=%d, seed=%u\n",
         prop.name, num_sm, occ, pgrid, waves, base, seed);
  printf("%-9s %7s %8s | %9s %9s %9s %9s | %7s %7s %7s | %s\n", "dist", "tiles", "ideal",
         "HW(np)", "static", "dyn-idx", "dyn-LPT", "st/HW", "dy/HW", "LPT/HW", "HWeff");

  std::mt19937 rng(seed);
  int* d_iters = nullptr; int* d_order = nullptr; int* d_ctr = nullptr; float* d_out = nullptr;
  int cap = (int)(waves * num_sm) + 64;
  // requests overshoot target a bit; allocate generously
  CK(cudaMalloc(&d_iters, sizeof(int) * cap * 64)); CK(cudaMalloc(&d_order, sizeof(int) * cap * 64));
  CK(cudaMalloc(&d_ctr, sizeof(int))); CK(cudaMalloc(&d_out, sizeof(float) * cap * 64));

  const char* dists[] = {"uniform", "causal1", "varlen", "tail"};
  for (const char* dist : dists) {
    std::vector<int> it = gen(dist, (int)(waves * num_sm), base, rng);
    int n = (int)it.size();
    // host-sorted heaviest-first order for the LPT scheduler
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return it[a] > it[b]; });
    long long total = std::accumulate(it.begin(), it.end(), 0LL);
    double ideal = (double)total / (num_sm * occ);   // perfect-balance lower bound (in iter-units)

    CK(cudaMemcpy(d_iters, it.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_order, order.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    float tA = time_min([&] { k_nonpersistent<<<n, kThreads>>>(n, d_iters, d_out); }, reps, warmup);
    float tB = time_min([&] { k_static<<<pgrid, kThreads>>>(n, d_iters, d_out); }, reps, warmup);
    float tC = time_min([&] { cudaMemset(d_ctr, 0, sizeof(int)); k_dynamic<<<pgrid, kThreads>>>(n, d_iters, d_ctr, d_out); }, reps, warmup);
    float tD = time_min([&] { cudaMemset(d_ctr, 0, sizeof(int)); k_dynamic_lpt<<<pgrid, kThreads>>>(n, d_iters, d_order, d_ctr, d_out); }, reps, warmup);
    // HW efficiency = ideal-makespan / actual; needs a cycles/iter constant -> instead report
    // HWeff as (best-of-all / HW) so 1.00 means HW is already the best; <1 means something beat it.
    float best = std::min(std::min(tA, tB), std::min(tC, tD));
    printf("%-9s %7d %8.0f | %8.3fms %8.3fms %8.3fms %8.3fms | %6.2fx %6.2fx %6.2fx | %.3f\n",
           dist, n, ideal, tA, tB, tC, tD, tB / tA, tC / tA, tD / tA, best / tA);
  }
  printf("\nst/dy/LPT vs HW: >1 slower, <1 faster. HWeff<1.00 => some scheduler beat the HW block scheduler.\n");
  cudaFree(d_iters); cudaFree(d_order); cudaFree(d_ctr); cudaFree(d_out);
  return 0;
}
