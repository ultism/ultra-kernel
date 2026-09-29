// staging round-trip probe: mma64 A-frag -> partition_A STS -> LDS back
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "s3b_dvdk2_kernel.cuh"

using namespace s3bdvdk2;

template <int kMode>   // 0 = flat [128,64] elementwise, 1 = raw per-thread dump
__global__ void __launch_bounds__(256, 1) probe(Element* out) {
  extern __shared__ char smem_raw[];
  Element* buf = reinterpret_cast<Element*>(smem_raw);
  TiledMmaK64 mma64;
  int const tid = threadIdx.x;
  auto thr64 = mma64.get_thread_slice(tid);

  Tensor sDSshape = make_tensor(make_smem_ptr(static_cast<Element*>(nullptr)), SmemLayoutDS{});
  Tensor tOrP = thr64.partition_fragment_A(sDSshape);

  Tensor t_u8 = recast<uint8_t>(tOrP);
  for (int i = 0; i < size(t_u8); ++i) t_u8(i) = uint8_t((tid * 32 + i) % 251);

  Tensor tOrR = thr64.partition_fragment_A(sDSshape);
  Tensor r8 = recast<uint8_t>(tOrR);
  for (int i = 0; i < size(r8); ++i) r8(i) = 0xFF;
  __syncthreads();

  if constexpr (kMode == 0) {
    auto lay_flat = make_layout(make_shape(Int<kBlockN>{}, Int<64>{}), make_stride(Int<64>{}, _1{}));
    Tensor sT = make_tensor(make_smem_ptr(buf), lay_flat);
    Tensor tXs = thr64.partition_A(sT);
    copy(tOrP, tXs);
    __syncthreads();
    copy(tXs, tOrR);
  } else {
    // raw per-thread dump: thread t's k-atom frag (16B) at buf + (kb*256 + t)*16
    auto st = [&](int kb) {
      uint4 const v = *reinterpret_cast<uint4 const*>(&tOrP(_0{}, _0{}, kb));
      *reinterpret_cast<uint4*>(buf + (kb * 256 + tid) * 16) = v;
    };
    auto ld = [&](int kb) {
      *reinterpret_cast<uint4*>(&tOrR(_0{}, _0{}, kb)) =
          *reinterpret_cast<uint4 const*>(buf + (kb * 256 + tid) * 16);
    };
    st(0);
    __syncthreads();
    ld(0);
    st(1);
    __syncthreads();
    ld(1);
  }

  for (int i = 0; i < 32; ++i)
    reinterpret_cast<uint8_t*>(out)[tid * 32 + i] = r8(i);
}

template <int kMode>
void run(const char* nm) {
  cudaFuncSetAttribute((const void*)probe<kMode>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  uint8_t* d; cudaMalloc(&d, 256 * 32);
  probe<kMode><<<1, 256, 16384>>>(reinterpret_cast<Element*>(d));
  printf("%s sync: %s  ", nm, cudaGetErrorString(cudaDeviceSynchronize()));
  std::vector<uint8_t> h(256 * 32);
  cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost);
  int bad = 0, firstbad = -1;
  for (int t = 0; t < 256; ++t)
    for (int i = 0; i < 32; ++i)
      if (h[t * 32 + i] != uint8_t((t * 32 + i) % 251)) { ++bad; if (firstbad < 0) firstbad = t * 32 + i; }
  printf("%s (%d/8192 bad, first at e%d)\n", bad ? "FAIL" : "OK", bad, firstbad);
  if (bad) {
    printf("  thr0 got : "); for (int i = 0; i < 32; ++i) printf("%3d ", h[i]); printf("\n");
    printf("  thr0 want: "); for (int i = 0; i < 32; ++i) printf("%3d ", uint8_t((0 * 32 + i) % 251)); printf("\n");
    printf("  thr1 got : "); for (int i = 0; i < 32; ++i) printf("%3d ", h[32 + i]); printf("\n");
    printf("  thr1 want: "); for (int i = 0; i < 32; ++i) printf("%3d ", uint8_t((1 * 32 + i) % 251)); printf("\n");
  }
  cudaFree(d);
}

int main() {
  run<0>("flat  [128,64]      ");
  run<1>("raw dump per-thread");
  return 0;
}
