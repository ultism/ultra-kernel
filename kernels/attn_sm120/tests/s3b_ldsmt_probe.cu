// c24 probe v3: full B-frag [N=128 d, K=64 q] via ldmatrix.m16n16.x2.trans.b8
// from the NAT tile sQ[64 q][128 d], vs golden LDSM_N on pre-transposed tile.
// Window mapping: k-atom kb (q 32kb..+32), n16-window nw (d 16nw..+16):
//   lane l -> row q = 32kb + l, 16B at d = 16nw.
// Post-fixup candidates tested host-side.
#include <cstdio>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include "s3b_dvdk2_kernel.cuh"

using namespace s3bdvdk2;

__global__ void __launch_bounds__(256, 1) probe(Element const* gnat, Element const* gtrn,
                                                Element* out_gold, Element* out_t) {
  extern __shared__ char smem_raw[];
  Element* sNat = reinterpret_cast<Element*>(smem_raw);
  Element* sTrn = reinterpret_cast<Element*>(smem_raw + 16384);
  int const tid = threadIdx.x;
  for (int i = tid; i < 64 * 128; i += 256) sNat[i] = gnat[i];
  for (int i = tid; i < 128 * 64; i += 256) sTrn[i] = gtrn[i];
  __syncthreads();

  TiledMmaK64 mma64;
  auto thr64 = mma64.get_thread_slice(tid);

  Tensor sDt = make_tensor(make_smem_ptr(sTrn),
                           make_layout(make_shape(Int<128>{}, Int<64>{}), make_stride(Int<64>{}, _1{})));
  Tensor tOrB = thr64.partition_fragment_B(sDt);
  auto ldN = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, mma64);
  auto tldN = ldN.get_thread_slice(tid);
  copy(ldN, tldN.partition_S(sDt), tldN.retile_D(tOrB));
  Tensor g8 = recast<uint8_t>(tOrB);
  for (int i = 0; i < size(g8); ++i) reinterpret_cast<uint8_t*>(out_gold)[tid * 32 + i] = g8(i);

  // test: all (kb, nw) windows; dump raw (no fixup) + cutlass-atom fixup
  int const lane = tid % 32;
  uint32_t raw[2 * 8 * 4];
  for (int kb = 0; kb < 2; ++kb)
    for (int nw = 0; nw < 8; ++nw) {
      Element const* src = sNat + (32 * kb + lane) * 128 + 16 * nw;
      uint32_t t0, t1, t2, t3;
      asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                   : "r"(cast_smem_ptr_to_uint(src)));
      int o = (kb * 8 + nw) * 4;
      raw[o + 0] = t0; raw[o + 1] = t1; raw[o + 2] = t2; raw[o + 3] = t3;
    }
  for (int i = 0; i < 2 * 8 * 4; ++i)
    reinterpret_cast<uint32_t*>(out_t)[tid * 64 + i] = raw[i];
}

int main() {
  Element *gn, *gt, *og, *ot;
  cudaMalloc(&gn, 64 * 128); cudaMalloc(&gt, 128 * 64);
  cudaMalloc(&og, 256 * 32); cudaMalloc(&ot, 256 * 64 * 4);
  cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  std::vector<uint8_t> hn(64 * 128), ht(128 * 64);
  std::vector<std::vector<uint8_t>> G(2);
  std::vector<uint8_t> T;
  // pattern A: v = q, pattern B: v = d  (both needed to decode (q,d) per byte)
  for (int pat = 0; pat < 2; ++pat) {
    for (int q = 0; q < 64; ++q)
      for (int d = 0; d < 128; ++d) {
        hn[q * 128 + d] = uint8_t(pat ? d : q);
        ht[d * 64 + q] = hn[q * 128 + d];
      }
    cudaMemcpy(gn, hn.data(), hn.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gt, ht.data(), ht.size(), cudaMemcpyHostToDevice);
    std::vector<uint8_t> hg(256 * 32); std::vector<uint8_t> ho(256 * 64 * 4);
    probe<<<1, 256, 65536>>>(gn, gt, og, ot);
    cudaMemcpy(hg.data(), og, hg.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(), ot, ho.size(), cudaMemcpyDeviceToHost);
    if (pat == 1) T = ho;   // keep pattern B (v=d) raw test dump for byte-ID
    G[pat] = hg;
    if (pat == 0) { T.clear(); }
  }
  printf("sync: %s\n", cudaGetErrorString(cudaGetLastError()));
  // We need test dumps for BOTH patterns; redo properly storing both.
  std::vector<std::vector<uint8_t>> TT(2);
  for (int pat = 0; pat < 2; ++pat) {
    for (int q = 0; q < 64; ++q)
      for (int d = 0; d < 128; ++d) {
        hn[q * 128 + d] = uint8_t(pat ? d : q);
        ht[d * 64 + q] = hn[q * 128 + d];
      }
    cudaMemcpy(gn, hn.data(), hn.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gt, ht.data(), ht.size(), cudaMemcpyHostToDevice);
    std::vector<uint8_t> ho(256 * 64 * 4);
    probe<<<1, 256, 65536>>>(gn, gt, og, ot);
    cudaMemcpy(ho.data(), ot, ho.size(), cudaMemcpyDeviceToHost);
    TT[pat] = ho;
  }

  // golden frag map: lane l byte b -> (q,d)
  auto gold = [&](int l, int b) { return std::make_pair(G[0][l * 32 + b], G[1][l * 32 + b]); };
  // test raw window (kb,nw) lane l word w -> (q,d) per byte
  auto test = [&](int l, int kb, int nw, int byte) {
    int idx = l * 256 + ((kb * 8 + nw) * 4) * 4 + byte;   // uint32 idx *4
    return std::make_pair(TT[0][idx], TT[1][idx]);
  };
  // print golden frag layout summary for lane 0..3 and the full byte map compactly
  for (int l : {0, 1, 4, 5}) {
    printf("GOLD l%02d: ", l);
    for (int b = 0; b < 32; ++b) { auto [q, d] = gold(l, b); printf("(%2d,%3d)", q, d); }
    printf("\n");
    printf("TEST l%02d (kb0nw0|nw1): ", l);
    for (int w = 0; w < 2; ++w) {
      for (int b = 0; b < 16; ++b) { auto [q, d] = test(l, 0, w, b); printf("(%2d,%3d)", q, d); }
      printf(" | ");
    }
    printf("\n");
  }
  return 0;
}
