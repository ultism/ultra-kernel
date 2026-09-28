// probe: SF fragment shapes + per-lane coord maps for the bwd mma64 A/B SF gathers.
// nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr \
//   --expt-extended-lambda -I tmp/cutlass/include -I include kernels/attn_sm120/tests/sf_probe.cu -o /tmp/sf_probe
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;
using Element = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int SFVecSize = 32;
using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<Element, Element, float, ElementSF, SFVecSize>;
using TiledMmaK64 = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, _64>{}));

__global__ void probe() {
  int tid = threadIdx.x;   // one consumer warpgroup... need 256 threads worth; use 256
  if (tid >= 256) return;
  TiledMmaK64 mma64;
  auto thr64 = mma64.get_thread_slice(tid);
  Tensor sfpA = mxfp8::partition_SFA(make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);
  Tensor sfpB = mxfp8::partition_SFB(make_identity_tensor(make_shape(Int<128>{}, Int<64>{})), thr64);
  if (tid < 2 || tid == 32 || tid == 33) {
    printf("tid %d: sizeA=%d sizeA_k=%d sizeB=%d sizeB_k=%d\n", tid,
           int(size(sfpA)), int(size(sfpA(_, _, 0))), int(size(sfpB)), int(size(sfpB(_, _, 0))));
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < int(size(sfpA(_, _, k))); ++i) {
        auto c = sfpA(_, _, k)(i);
        if (i < 4 || i == int(size(sfpA(_, _, k))) - 1)
          printf("  A tid %d k%d i%2d -> (%d, %d)\n", tid, k, i, int(get<0>(c)), int(get<1>(c)));
      }
    for (int k = 0; k < 2; ++k)
      for (int i = 0; i < int(size(sfpB(_, _, k))); ++i) {
        auto c = sfpB(_, _, k)(i);
        if (i < 4 || i == int(size(sfpB(_, _, k))) - 1)
          printf("  B tid %d k%d i%2d -> (%d, %d)\n", tid, k, i, int(get<0>(c)), int(get<1>(c)));
      }
  }
}
int main() { probe<<<1, 256>>>(); cudaDeviceSynchronize(); return 0; }
