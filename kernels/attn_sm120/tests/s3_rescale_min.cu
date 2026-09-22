// Minimal isolation of the S3 online-O-rescale bug: strip ALL surrounding machinery
// (TMA, pipelines, softmax, smem-P-transpose, NamedBarriers) and keep ONLY the
// pattern under suspicion:
//
//   accO = partition_fragment_C(mma_pv)   // MMA accumulator, persists across blocks
//   clear(accO)
//   for nb in 0..N:
//       accO *= scores_scale[nb]          // FP rescale of the accumulator
//       gemm(P, V, accO)                  // blockscaled PV accumulate into it
//
// P and V are dummy all-ones (e4m3 1.0), all block scales = 2^0, so each block's
// PV = 128 per output element (sum over 128 keys). With scores_scale = [0, tiny, 0, 0]
// (mimicking a run where the running max grows so early blocks underflow away), the
// math gives accO == 128 each element (only the last block survives, coeff 1).
//
//   ref[i] = 128.0   for all i
//
// If the device prints 0, the bug reproduces with NOTHING but rescale+gemm -> it's
// the ptxas codegen for "FP-scaled MMA accumulator fed back into a blockscaled gemm".
// If it prints 128, the bug lives in the s3 surroundings, not this pattern.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s3_rescale_min.cu -o tests/s3_rescale_min

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

#define CK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 128, SFVecSize = 32;
constexpr int NumMmaThreads = 256;
constexpr int N_BLOCKS = 4;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using TiledMmaPV = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kBlockN>>{}));
using TiledMmaQK = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kHeadDim>>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomKeys = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockN>>());
using SmemLayoutP  = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

// Canonical block-scaled SF smem (SFA for P, SFB for V), same construction as s3.
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaPV::AtomShape_MNK{}) / SFVecSize;
using Blk_MN = typename BlkSF::Blk_MN; using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>; using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>; using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{}, Int<kHeadDim>{} / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFP = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
using sSFB_shapeN = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFV = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFB_strideK{})));

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>>  sP;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutVt>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>>  sQ;   // (q,hd) all-ones, for QK
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>>  sK;   // (key,hd) all-ones
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFP>> sSFP;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> sSFV;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFP>> sSFQ;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> sSFK;
};

struct Params { float* out_O; float ss[N_BLOCKS]; int mode; int do_qk; };  // mode 0/1; do_qk adds a QK gemm into accS

__global__ void __launch_bounds__(NumMmaThreads, 1)
min_kernel(CUTE_GRID_CONSTANT Params const params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  int const tid = threadIdx.x;

  // fill data smem with e4m3(1.0) and SF smem with ue8m0 2^0 (byte 127). uniform => layout-agnostic.
  uint8_t const one_e4m3 = cutlass::float_e4m3_t(1.0f).storage;
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sP.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutP>); i += NumMmaThreads) p[i] = one_e4m3; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sV.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutVt>); i += NumMmaThreads) p[i] = one_e4m3; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sSFP.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutSFP>); i += NumMmaThreads) p[i] = 127; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sSFV.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutSFV>); i += NumMmaThreads) p[i] = 127; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sQ.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutP>); i += NumMmaThreads) p[i] = one_e4m3; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sK.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutP>); i += NumMmaThreads) p[i] = one_e4m3; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sSFQ.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutSFP>); i += NumMmaThreads) p[i] = 127; }
  { uint8_t* p = reinterpret_cast<uint8_t*>(ss.sSFK.begin());
    for (int i = tid; i < int(cute::cosize_v<SmemLayoutSFV>); i += NumMmaThreads) p[i] = 127; }
  __syncthreads();

  Tensor sP   = make_tensor(make_smem_ptr(ss.sP.begin()),  SmemLayoutP{});
  Tensor sV   = make_tensor(make_smem_ptr(ss.sV.begin()),  SmemLayoutVt{});
  Tensor sSFP = make_tensor(make_smem_ptr(ss.sSFP.begin()), SmemLayoutSFP{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFV{});

  TiledMmaPV mma_pv;
  auto thr_pv = mma_pv.get_thread_slice(tid);
  Tensor tOrP  = thr_pv.partition_fragment_A(sP);
  Tensor tOrV  = thr_pv.partition_fragment_B(sV);
  Tensor tOrSFP = mxfp8::partition_fragment_SFA(sSFP, thr_pv);
  Tensor tOrSFV = mxfp8::partition_fragment_SFB(sSFV, thr_pv);
  auto scP = make_tiled_copy_A(SmemCopyAtomData{}, mma_pv); auto tscP = scP.get_thread_slice(tid);
  auto scV = make_tiled_copy_B(SmemCopyAtomData{}, mma_pv); auto tscV = scV.get_thread_slice(tid);
  auto ts_pv = tile_shape(mma_pv);
  auto scSFP = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma_pv), make_shape(size<0>(ts_pv), size<2>(ts_pv)));
  auto scSFV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_pv), make_shape(size<1>(ts_pv), size<2>(ts_pv)));
  auto tscSFP = scSFP.get_thread_slice(tid); auto tscSFV = scSFV.get_thread_slice(tid);

  // mode<2: load operands ONCE here. mode>=2: reload inside the loop (mimics s3).
  if (params.mode < 2) {
    copy(scP, tscP.partition_S(as_position_independent_swizzle_tensor(sP)), tscP.retile_D(tOrP));
    copy(scV, tscV.partition_S(as_position_independent_swizzle_tensor(sV)), tscV.retile_D(tOrV));
    copy(scSFP, tscSFP.partition_S(as_position_independent_swizzle_tensor(sSFP)), tscSFP.retile_D(tOrSFP));
    copy(scSFV, tscSFV.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFV.retile_D(tOrSFV));
  }

  // QK operands (all-ones), to add a second live MMA accumulator accS like s3.
  Tensor sQ = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutP{});
  Tensor sK = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutP{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFP{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFV{});
  TiledMmaQK mma_qk; auto thr_qk = mma_qk.get_thread_slice(tid);
  Tensor tSrQ = thr_qk.partition_fragment_A(sQ);
  Tensor tSrK = thr_qk.partition_fragment_B(sK);
  Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_qk);
  Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK, thr_qk);
  { auto c = make_tiled_copy_A(SmemCopyAtomData{}, mma_qk); auto t = c.get_thread_slice(tid);
    copy(c, t.partition_S(as_position_independent_swizzle_tensor(sQ)), t.retile_D(tSrQ)); }
  { auto c = make_tiled_copy_B(SmemCopyAtomData{}, mma_qk); auto t = c.get_thread_slice(tid);
    copy(c, t.partition_S(as_position_independent_swizzle_tensor(sK)), t.retile_D(tSrK)); }
  auto ts_qk = tile_shape(mma_qk);
  { auto c = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma_qk), make_shape(size<0>(ts_qk), size<2>(ts_qk))); auto t = c.get_thread_slice(tid);
    copy(c, t.partition_S(as_position_independent_swizzle_tensor(sSFQ)), t.retile_D(tSrSFQ)); }
  { auto c = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_qk), make_shape(size<1>(ts_qk), size<2>(ts_qk))); auto t = c.get_thread_slice(tid);
    copy(c, t.partition_S(as_position_independent_swizzle_tensor(sSFK)), t.retile_D(tSrSFK)); }

  Tensor accO = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));
  clear(accO);

  for (int nb = 0; nb < N_BLOCKS; ++nb) {
    float s = params.ss[nb];
    if (params.do_qk) {
      // QK gemm into a fresh accS (consumed into s so it stays live, like s3's softmax).
      Tensor accS = partition_fragment_C(mma_qk, select<0, 1>(TileShape_MNK{}));
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrQ); ++k)
        cute::gemm(mma_qk, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                   make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accS);
      float acc = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accS); ++i) acc += accS(i);
      s += 0.f * acc;   // keep accS live without changing the math
    }
    if (params.mode >= 2) {  // reload operands each block (fixed data) + a NamedBarrier like s3
      copy(scP, tscP.partition_S(as_position_independent_swizzle_tensor(sP)), tscP.retile_D(tOrP));
      copy(scV, tscV.partition_S(as_position_independent_swizzle_tensor(sV)), tscV.retile_D(tOrV));
      copy(scSFP, tscSFP.partition_S(as_position_independent_swizzle_tensor(sSFP)), tscSFP.retile_D(tOrSFP));
      copy(scSFV, tscSFV.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFV.retile_D(tOrSFV));
    }
    if (params.mode == 0 || params.mode == 2) {
      // in-place: rescale the MMA accumulator, then gemm-accumulate into it.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accO); ++i) accO(i) *= s;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrP); ++k)
        cute::gemm(mma_pv, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                   make_zip_tensor(tOrV(_, _, k), tOrSFV(_, _, k)), accO);
    } else {
      // separate: gemm into a fresh accumulator, then accO = accO*s + blk.
      Tensor accB = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));
      clear(accB);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrP); ++k)
        cute::gemm(mma_pv, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                   make_zip_tensor(tOrV(_, _, k), tOrSFV(_, _, k)), accB);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accO); ++i) accO(i) = accO(i) * s + accB(i);
    }
  }

  Tensor mO = make_tensor(make_gmem_ptr(params.out_O),
                          make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, _1{})));
  copy(accO, thr_pv.partition_C(mO));
}

int run(const char* tag, int mode, const std::vector<float>& ss, int do_qk = 0) {
  Params params; params.mode = mode; params.do_qk = do_qk; params.out_O = nullptr;
  for (int i = 0; i < N_BLOCKS; ++i) params.ss[i] = ss[i];
  float* dO; CK(cudaMalloc(&dO, kBlockM * kHeadDim * sizeof(float))); params.out_O = dO;
  int smem = int(sizeof(SharedStorage));
  CK(cudaFuncSetAttribute(min_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  min_kernel<<<1, NumMmaThreads, smem>>>(params);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> hO(kBlockM * kHeadDim);
  CK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));
  // host ref: replay accO = accO*ss[nb] + 128 (each block PV = 128 for all-ones, scale 1).
  double ref = 0.0; for (int nb = 0; nb < N_BLOCKS; ++nb) ref = ref * (double)ss[nb] + 128.0;
  double maxabs = 0; int nz = 0;
  for (auto v : hO) { maxabs = std::max(maxabs, std::abs((double)v - ref)); if (v != 0.f) ++nz; }
  printf("  [%-18s] O[0]=%.4f O[mid]=%.4f ref=%.4f  max|abs|=%.3g  nonzero=%d/%zu  %s\n",
         tag, hO[0], hO[hO.size()/2], ref, maxabs, nz, hO.size(),
         maxabs < 1e-2 ? "OK" : "MISMATCH");
  cudaFree(dO);
  return maxabs < 1e-2 ? 0 : 1;
}

int main() {
  printf("S3 rescale isolation: accO *= ss[nb]; gemm(P=1,V=1 -> +128) ; N=%d blocks\n", N_BLOCKS);
  int rc = 0;
  // ss patterns. "tiny" mimics underflowed running-max rescale (only last block survives -> ref 128).
  rc |= run("inplace ss=1.0",    0, {1, 1, 1, 1});           // ref = 128*4 = 512
  rc |= run("inplace ss=0.5",    0, {0, 0.5f, 0.5f, 0.5f});  // ref = 128*(0.25+0.5+1)? trace
  rc |= run("inplace ss=tiny",   0, {0, 1e-26f, 0, 0});      // ref = 128 (only last survives)
  rc |= run("inplace ss=1e-3",   0, {0, 1e-3f, 1e-3f, 1e-3f});
  rc |= run("separate ss=tiny",  1, {0, 1e-26f, 0, 0});      // ref = 128
  rc |= run("separate ss=1e-3",  1, {0, 1e-3f, 1e-3f, 1e-3f});
  rc |= run("reload+inplace tiny", 2, {0, 1e-26f, 0, 0});    // ref = 128 (reload operands each block)
  rc |= run("reload+sep tiny",     3, {0, 1e-26f, 0, 0});    // ref = 128
  // + QK gemm (second live MMA accumulator accS), mimicking s3's register pressure.
  rc |= run("QK+inplace tiny",     0, {0, 1e-26f, 0, 0}, /*do_qk=*/1);
  rc |= run("QK+separate tiny",    1, {0, 1e-26f, 0, 0}, /*do_qk=*/1);
  rc |= run("QK+reload+sep tiny",  3, {0, 1e-26f, 0, 0}, /*do_qk=*/1);
  printf(rc == 0 ? "MIN PASS (bug NOT reproduced here -> it's the surroundings)\n"
                 : "MIN FAIL (bug reproduced in isolation -> rescale+gemm pattern)\n");
  return rc;
}
