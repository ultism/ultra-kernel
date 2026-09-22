// A/B test: persistent vs non-persistent tile scheduling on THIS GPU.
//
// The persistent-vs-not question for our MXFP8 prefill kernel is independent of
// the numeric format -- it only depends on (#tiles vs #SMs) and load imbalance
// (causal -> later query blocks attend to more keys -> more work). So instead of
// building SageAttention3's full nvfp4 stack, this isolates the scheduling effect
// with a tunable per-tile compute kernel and compares the three schedulers that
// SageAttention's tile_scheduler.h offers:
//
//   (A) non-persistent      grid = num_tiles,  1 tile/CTA, HW block scheduler
//   (B) static  persistent  grid = num_sm,     for(t=bid; t<N; t+=gridDim.x)
//   (C) dynamic persistent  grid = num_sm,     atomicAdd counter grabs next tile
//
// across tile-count multiples of the SM count and two work distributions:
//   uniform : every tile does the same work
//   causal  : tile's work ~ (qblock+1) within each (head*batch) group -- the
//             triangular imbalance a causal prefill actually has.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a \
//     bench/sched_ab.cu -o bench/sched_ab
//
// What to read off: under "causal", static-persistent (B) round-robins a fixed
// tile subsequence per CTA, so an unlucky CTA inherits the heavy tail and gates
// the launch; the HW scheduler (A) and the atomic scheduler (C) both grab work
// greedily and balance. Under "uniform" all three should match. That tells us
// whether persistent scheduling helps, hurts, or is neutral for our kernel.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(call)                                                               \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static constexpr int kThreads = 256;

enum Dist { UNIFORM = 0, CAUSAL = 1 };

// Per-tile compute: every thread burns `iters` FMAs, block-reduces so none is
// dead-code-eliminated, thread 0 writes the result. iters(t) sets the imbalance.
__device__ __forceinline__ int iters_for(int t, int base, int G, int dist) {
  if (dist == UNIFORM) return base;
  int qblock = t % G;          // position within a (head*batch) group
  return base * (qblock + 1);  // causal: later query blocks => more keys => more work
}

__device__ __forceinline__ void do_tile(int t, int base, int G, int dist,
                                         float* out) {
  int iters = iters_for(t, base, G, dist);
  float a = 1.0f + 1e-3f * threadIdx.x, b = 1.0000003f;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    a = a * b + 0.5f;
    b = b * 1.0000003f + a * 1e-9f;
  }
  __shared__ float s[kThreads];
  s[threadIdx.x] = a + b;
  __syncthreads();
  for (int o = kThreads >> 1; o > 0; o >>= 1) {
    if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[t] = s[0];
}

__global__ void k_nonpersistent(int num_tiles, int base, int G, int dist,
                                 float* out) {
  int t = blockIdx.x;  // grid == num_tiles
  if (t < num_tiles) do_tile(t, base, G, dist, out);
}

__global__ void k_static_persistent(int num_tiles, int base, int G, int dist,
                                     float* out) {
  for (int t = blockIdx.x; t < num_tiles; t += gridDim.x)
    do_tile(t, base, G, dist, out);
}

__global__ void k_dynamic_persistent(int num_tiles, int base, int G, int dist,
                                      int* counter, float* out) {
  __shared__ int s_t;
  while (true) {
    if (threadIdx.x == 0) s_t = atomicAdd(counter, 1);
    __syncthreads();
    int t = s_t;
    __syncthreads();
    if (t >= num_tiles) break;
    do_tile(t, base, G, dist, out);
  }
}

struct Timing { float ms; };

template <typename Launch>
static float time_min(Launch launch, int reps, int warmup) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < warmup; ++i) launch();
  cudaDeviceSynchronize();
  float best = 1e30f;
  for (int i = 0; i < reps; ++i) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    best = std::min(best, ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

int main(int argc, char** argv) {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  int num_sm = prop.multiProcessorCount;
  printf("GPU: %s  SMs=%d  (block=%d threads)\n", prop.name, num_sm, kThreads);

  const int base = 3000;  // FMA iters for the "unit" tile
  const int G = (argc > 1) ? atoi(argv[1]) : 8;  // q-blocks per group (causal period)
  const int reps = 40, warmup = 8;
  printf("causal period G = %d\n", G);

  // A persistent kernel must launch one CTA per RESIDENT slot, not one per SM:
  // grid = num_sm * max_active_blocks_per_sm. Otherwise it starves a GPU whose
  // occupancy for this kernel is >1 block/SM, and we'd be measuring occupancy
  // loss, not scheduling. (FA3/SageAttention use grid=num_sm because their heavy
  // warp-specialized CTAs already occupy 1 block/SM.)
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, k_static_persistent, kThreads, kThreads * sizeof(float)));
  int persistent_grid = num_sm * occ;
  printf("occupancy = %d block(s)/SM  ->  persistent grid = %d CTAs\n",
         occ, persistent_grid);

  // Tile counts as multiples of SM count: the regime axis that drives the answer.
  std::vector<float> mults = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};

  int max_tiles = (int)(mults.back() * num_sm) + 4;
  float* d_out = nullptr;
  int* d_counter = nullptr;
  CK(cudaMalloc(&d_out, sizeof(float) * max_tiles));
  CK(cudaMalloc(&d_counter, sizeof(int)));

  // Single-shot profile mode: launch each scheduler ONCE at a fixed heavy causal
  // config so `ncu` profiles exactly 3 clean kernels. Usage: ./sched_ab <G> profile
  if (argc > 2 && strcmp(argv[2], "profile") == 0) {
    int num_tiles = 32 * num_sm;  // 32 waves, causal
    printf("PROFILE: num_tiles=%d (causal), persistent_grid=%d\n",
           num_tiles, persistent_grid);
    k_nonpersistent<<<num_tiles, kThreads>>>(num_tiles, base, G, CAUSAL, d_out);
    k_static_persistent<<<persistent_grid, kThreads>>>(num_tiles, base, G, CAUSAL, d_out);
    CK(cudaMemset(d_counter, 0, sizeof(int)));
    k_dynamic_persistent<<<persistent_grid, kThreads>>>(num_tiles, base, G, CAUSAL,
                                                        d_counter, d_out);
    CK(cudaDeviceSynchronize());
    cudaFree(d_out);
    cudaFree(d_counter);
    return 0;
  }

  for (int dist = 0; dist <= 1; ++dist) {
    printf("\n=== distribution: %s ===\n", dist == UNIFORM ? "UNIFORM" : "CAUSAL");
    printf("%6s %8s | %10s %10s %10s | %9s %9s\n", "tiles", "waves",
           "nonpersist", "static-P", "dynamic-P", "B/A", "C/A");
    for (float m : mults) {
      int num_tiles = (int)(m * num_sm);
      float waves = (float)num_tiles / num_sm;

      float tA = time_min([&] {
        k_nonpersistent<<<num_tiles, kThreads>>>(num_tiles, base, G, dist, d_out);
      }, reps, warmup);

      float tB = time_min([&] {
        k_static_persistent<<<persistent_grid, kThreads>>>(num_tiles, base, G, dist, d_out);
      }, reps, warmup);

      float tC = time_min([&] {
        cudaMemset(d_counter, 0, sizeof(int));
        k_dynamic_persistent<<<persistent_grid, kThreads>>>(num_tiles, base, G, dist,
                                                            d_counter, d_out);
      }, reps, warmup);

      printf("%6d %8.1f | %9.3fms %9.3fms %9.3fms | %8.2fx %8.2fx\n",
             num_tiles, waves, tA, tB, tC, tB / tA, tC / tA);
    }
  }

  printf("\nB/A>1 => static-persistent SLOWER than non-persistent; "
         "C/A~1 => dynamic-persistent matches HW scheduler.\n");
  cudaFree(d_out);
  cudaFree(d_counter);
  return 0;
}
