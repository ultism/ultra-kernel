// S5 probe (clean): for ONE lane at a time, print the QK-C key set (in C-fragment
// order) and the PV-A key set (in A-fragment order) for q-row 0, so we can derive the
// exact intra-quad shuffle that assembles tOrP from accS with no smem.
// Build: nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//        --expt-relaxed-constexpr --expt-extended-lambda -I tmp/cutlass/include -I include \
//        bench/s5_layout_probe.cu -o bench/s5_layout_probe
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>
using namespace cute;

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kBlockM = 128, kBlockN = 128, kHeadDim = 128, SFVecSize = 32;
using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<Element, Element, float, ElementSF, SFVecSize>;
using TiledMmaQK = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8,_1,_1>>{}, Tile<_128,_32,Int<kHeadDim>>{}));
using TiledMmaPV = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8,_1,_1>>{}, Tile<_128,_32,Int<kBlockN>>{}));

__global__ void probe(int T) {
  int tid = threadIdx.x;
  if (tid != T) return;
  TiledMmaQK mma_qk; TiledMmaPV mma_pv;
  auto cC = mma_qk.get_thread_slice(tid).partition_C(make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
  auto cA = mma_pv.get_thread_slice(tid).partition_A(make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
  // V is PV-B [head_dim, keys]; partition_B over (N=head_dim, K=keys) identity -> (hd,key).
  auto cB = mma_pv.get_thread_slice(tid).partition_B(make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
  // C: ((2,2),MMA_M=1,MMA_N=16). q-row mi=0 is atom-b=0. List keys for b=0 over all n.
  printf("lane %2d (q%%4=%d,row=%d)\n", tid, tid % 4, tid / 4);
  printf("  C keys (qk acc, mi=0): ");
  for (int n = 0; n < size<2>(cC); ++n)
    printf("%d ", int(get<1>(cC(make_coord(_0{}, _0{}), _0{}, n))));   // a=0; a=1 is +1
  printf("\n  A keys (pv-P, frag order): ");
  // A: ((4,2,2),MMA_M=1,MMA_K=4). q-row is atom dim1 (e1)=0. Print key for e1=0 slots.
  for (int k = 0; k < size<2>(cA); ++k)
    for (int e0 = 0; e0 < 4; ++e0)
      for (int e2 = 0; e2 < 2; ++e2)
        printf("%d ", int(get<1>(cA(make_coord(e0, _0{}, e2), _0{}, k))));
  printf("\n  B keys (pv-V, frag order, hd=0): ");
  // B: (atomB, MMA_N(head_dim), MMA_K(keys)). Take head_dim slice 0, list keys.
  for (int k = 0; k < size<2>(cB); ++k)
    for (int e = 0; e < size<0>(cB); ++e) {
      auto c = cB(e, _0{}, k);
      if (int(get<0>(c)) == 0) printf("%d ", int(get<1>(c)));   // only hd==0 row
    }
  printf("\n");
}

int main() {
  printf("=== QK-C vs PV-A key sets (q-row 0), per lane ===\n");
  for (int t = 0; t < 4; ++t) { probe<<<1, 32>>>(t); cudaDeviceSynchronize(); }
  return 0;
}
