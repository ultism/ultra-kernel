// S0.0 probe (v2): authoritative mxfp8 SF-via-TMA layout, replicated from the
// cutlass sm120 blockscaled builder (sm120_blockscaled_mma_builder.inl:182-209),
// which builds SmemLayoutAtomSF inline (TiledMma-agnostic) instead of via the
// deduce_smem_layoutSFA that needs UMMA's TiledMma::K (which our warp-atom
// TiledMMA lacks -- confirmed by probe v1).
//
// Confirms: (1) the SF smem atom instantiates + prints sane for SFVecSize=32,
// MMA_NSF=1; (2) the canonical gmem SF layout; (3) make_tma_copy<uint16_t> for SF
// instantiates. If all pass, mxfp8 SF-via-TMA is fully de-risked.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include tests/s0_sf_probe.cu -o tests/s0_sf_probe

#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t, 32>;

int main() {
  constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 128, SFVecSize = 32;
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(
      AtomMXF8{}, AtomLayoutMNK{}, Tile<_128, _32, Int<kHeadDim>>{}));

  constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVecSize;
  printf("MMA_NSF = %d  (expect 1 for mxf8)\n", MMA_NSF);

  using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
  using Blk_MN = typename BlkSF::Blk_MN;          // _128
  using Blk_SF = typename BlkSF::Blk_SF;          // _4
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using mnBasicBlockShape  = Shape<_32, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape   = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
  using kBasicBlockStride  = Stride<_0, _1>;

  // ---- SFA (Q, M side) ----
  using sSFA_shapeM  = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
  using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
  using sSF_shapeK   = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                                   Int<kHeadDim>{} / Int<SFVecSize>{} / Blk_SF{}),
                                        kBasicBlockShape{}));
  using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{},
                                                    Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}),
                                        kBasicBlockStride{}));
  using SmemLayoutAtomSFA = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}),
                                                 make_stride(sSF_strideMN{}, sSFA_strideK{})));

  // ---- SFB (K, N side; N tile padded to >=128) ----
  using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
  using sSFB_shapeN  = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
  using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{},
                                                    sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}),
                                        kBasicBlockStride{}));
  using SmemLayoutAtomSFB = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}),
                                                 make_stride(sSF_strideMN{}, sSFB_strideK{})));

  print("SmemLayoutAtomSFA = "); print(SmemLayoutAtomSFA{}); print("\n");
  printf("  cosize(SFA)=%d size(SFA)=%d\n", int(cosize(SmemLayoutAtomSFA{})), int(size(SmemLayoutAtomSFA{})));
  print("SmemLayoutAtomSFB = "); print(SmemLayoutAtomSFB{}); print("\n");
  printf("  cosize(SFB)=%d size(SFB)=%d\n", int(cosize(SmemLayoutAtomSFB{})), int(size(SmemLayoutAtomSFB{})));

  // ---- canonical gmem SF layout + SF TMA copy instantiation ----
  auto sfa_gmem = BlkSF::tile_atom_to_shape_SFA(make_shape(4096, kBlockN, kHeadDim));
  print("SFA gmem (seqlen=4096,hd=128) = "); print(sfa_gmem); print("\n");

  void* d_sf = nullptr;
  cudaMalloc(&d_sf, size_t(4096) * (kHeadDim / SFVecSize) * 4);  // generous, 16B-aligned
  auto mSFA = make_tensor(make_gmem_ptr(static_cast<cutlass::float_ue8m0_t const*>(
                              reinterpret_cast<cutlass::float_ue8m0_t*>(d_sf))), sfa_gmem);
  auto tma_sfa = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFA, SmemLayoutAtomSFA{},
                                         make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{});
  (void)tma_sfa;
  printf("SF TMA copy instantiated + descriptor encoded OK\n");

  printf("PROBE_OK\n");
  return 0;
}
