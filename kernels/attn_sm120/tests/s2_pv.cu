// S2: P-requant to mxfp8 (e4m3 + ue8m0) for the PV A-operand, validated by a
// real PV MMA -> O on a single n_block.
//
// The crux (measured in /tmp/pv_layout): the QK-C accumulator's N(key) micro-
// layout is "2 adjacent keys, stride 8" while the PV-A K(key) micro-layout is
// "4 adjacent keys". That 2-vs-4 granularity is intrinsic to FP32-accumulate n8
// vs FP8 k32, so NO permutation makes P thread-local across QK->PV (unlike
// SageAttention's nvfp4 n32k64 atom). => P must go through smem as a transpose
// buffer: write quantized e4m3 P by logical (q,key) via partition_C, ldmatrix it
// back as the PV-A operand by logical (q,key). Same for the per-32-key ue8m0 SF
// (written to a simple [q][keyblock] smem, gathered into the PV-SFA fragment via
// partition_SFA(identity) coords). V reuses the K data+SF path (head_dim==kBlockN
// here so the block-scaled layouts are identical).
//
// Validation isolates LAYOUT from QUANT scheme: the fp64 reference quantizes P
// with the SAME e4m3/ue8m0 rule (host cutlass::float_e4m3_t), so device O should
// match to fp32-MMA-accumulation precision, not just "within mxfp8 error".
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s2_pv.cu -o tests/s2_pv

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"
#include "flashinfer/attention/blackwell/prefill/flashinfer_tile_scheduler.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 128, SFVecSize = 32;
constexpr int NBLK = kHeadDim / SFVecSize;        // SF blocks along head_dim (QK)
constexpr int NKB  = kBlockN / SFVecSize;         // SF blocks along keys (PV)  = 4
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;     // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;                  // named barrier id for P-smem handoff

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using TiledMmaQK = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kHeadDim>>{}));
// PV: O[M=q, N=head_dim] = P[q, K=keys] * Vt[head_dim, K=keys]. TileK = kBlockN (4 k-atoms).
using TiledMmaPV = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kBlockN>>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<1, 2>(TileShape_MNK{})));
// P-smem transpose buffer [q, key] and Vt-smem [head_dim, keys]: inner extent = keys = kBlockN.
using SmemLayoutAtomKeys = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockN>>());
using SmemLayoutP  = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

// Canonical block-scaled SF smem (same construction as S0/S1). 128x128 operand ->
// reused verbatim for SFQ, SFK, and (head_dim==kBlockN) SFV. MMA_NSF = 1.
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVecSize;
using Blk_MN = typename BlkSF::Blk_MN; using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>; using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>; using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                               Int<kHeadDim>{} / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFQ = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
using sSFB_shapeN = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFK = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFB_strideK{})));
using SmemLayoutSFV = SmemLayoutSFK;

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

using LayoutSF = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim))));

using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kBlockM), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));
using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kBlockN), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutK{}, select<1, 2>(TileShape_MNK{}), _1{}));
using TMA_V = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kHeadDim), int(kBlockN)), make_stride(int(kBlockN), _1{})),
    SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{}));
using TMA_SF = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));

using PipeQ  = cutlass::PipelineTmaAsync<1>;   // Q
using PipeKV = cutlass::PipelineTmaAsync<1>;   // K and V (single n_block)
using StateQ  = cutlass::PipelineState<1>;
using StateKV = cutlass::PipelineState<1>;

struct Params {
  TMA_Q tma_q; TMA_K tma_k; TMA_V tma_v; TMA_SF tma_sfq, tma_sfk, tma_sfv;
  LayoutSF layout_sf;           // 128x128 (K, V)
  LayoutSF layout_sfq;          // seqlen_q x 128 (Q spans all m_blocks; same type, dynamic extents)
  int seqlen_q, seqlen_k;
  float sm_scale;
  float* out_O;   // [seqlen_q, head_dim]
  float* outPre;  // [seqlen_q, seqlen_k] device pre-quant float P (host re-quantizes the SAME P)
  float* outL;    // [seqlen_q] row_sum
};

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>>  sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutVt>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>>  sP;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> sSFQ;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> sSFK;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> sSFV;
  alignas(128)  ElementSF sSFP[kBlockM * NKB];   // simple [q][keyblock]
  struct {
    alignas(16) PipeQ::SharedStorage  pipeline_q;
    alignas(16) PipeKV::SharedStorage pipeline_k;
    alignas(16) PipeKV::SharedStorage pipeline_v;
  };
};

static constexpr uint32_t TmaBytesQ =
    cute::cosize_v<SmemLayoutQ> * sizeof(Element) + cute::cosize_v<SmemLayoutSFQ> * sizeof(ElementSF);
static constexpr uint32_t TmaBytesK =
    cute::cosize_v<SmemLayoutK> * sizeof(Element) + cute::cosize_v<SmemLayoutSFK> * sizeof(ElementSF);
static constexpr uint32_t TmaBytesV =
    cute::cosize_v<SmemLayoutVt> * sizeof(Element) + cute::cosize_v<SmemLayoutSFV> * sizeof(ElementSF);

// e4m3 + ue8m0 quant of one value given the block's ue8m0 scale-exponent.
__device__ __forceinline__ Element quant_e4m3(float v, int scale_exp) {
  return Element(v * exp2f(float(-scale_exp)));   // store = round_e4m3(v / 2^scale_exp)
}
// OCP-MX scale exponent for an e4m3 block: floor(log2(amax)) - 8  (e4m3 emax=8).
// Bit-extraction (deterministic; host mirrors it exactly via the same formula).
__host__ __device__ __forceinline__ int mx_scale_exp_bits(uint32_t b) {
  int e = int((b >> 23) & 0xFF) - 127;   // floor(log2) for a normal float
  int se = e - 8;
  return se < -127 ? -127 : (se > 127 ? 127 : se);
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (!(amax > 0.f)) return -127;
  return mx_scale_exp_bits(__float_as_uint(amax));
}

template <class Op>
__device__ __forceinline__ float quad_reduce(float v, Op op) {
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 2));
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 1));
  return v;
}

template <typename TileScheduler>
__global__ void __launch_bounds__(kNThreads, 1)
s2_kernel(CUTE_GRID_CONSTANT Params const params,
          CUTE_GRID_CONSTANT typename TileScheduler::Params const sched_params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  int const wg = cutlass::canonical_warp_group_idx();        // 0=producer
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  PipeQ::Params pq; pq.role = (wg == 0) ? PipeQ::ThreadCategory::Producer : PipeQ::ThreadCategory::Consumer;
  pq.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0); pq.num_consumers = NumMmaThreads;
  pq.transaction_bytes = TmaBytesQ; PipeQ pipeline_q(ss.pipeline_q, pq, Shape<_1, _1, _1>{});
  PipeKV::Params pk; pk.role = pq.role; pk.is_leader = pq.is_leader; pk.num_consumers = NumMmaThreads;
  pk.transaction_bytes = TmaBytesK; PipeKV pipeline_k(ss.pipeline_k, pk, Shape<_1, _1, _1>{});
  PipeKV::Params pv; pv.role = pq.role; pv.is_leader = pq.is_leader; pv.num_consumers = NumMmaThreads;
  pv.transaction_bytes = TmaBytesV; PipeKV pipeline_v(ss.pipeline_v, pv, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ   = make_tensor(make_smem_ptr(ss.sQ.begin()),  SmemLayoutQ{});
  Tensor sK   = make_tensor(make_smem_ptr(ss.sK.begin()),  SmemLayoutK{});
  Tensor sV   = make_tensor(make_smem_ptr(ss.sV.begin()),  SmemLayoutVt{});
  Tensor sP   = make_tensor(make_smem_ptr(ss.sP.begin()),  SmemLayoutP{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFQ{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFK{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFV{});

  float const sm_scale_log2 = params.sm_scale * kLog2e;
  TileScheduler scheduler;

  if (wg == 0) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mQ = params.tma_q.get_tma_tensor(make_shape(int(params.seqlen_q), int(kHeadDim)));
      Tensor mK = params.tma_k.get_tma_tensor(make_shape(int(params.seqlen_k), int(kHeadDim)));
      Tensor mV = params.tma_v.get_tma_tensor(make_shape(int(kHeadDim), int(params.seqlen_k)));
      Tensor mSFQ = params.tma_sfq.get_tma_tensor(shape(params.layout_sfq));
      Tensor mSFK = params.tma_sfk.get_tma_tensor(shape(params.layout_sf));
      Tensor mSFV = params.tma_sfv.get_tma_tensor(shape(params.layout_sf));
      auto bq = params.tma_q.get_slice(_0{}); auto bsq = params.tma_sfq.get_slice(_0{});
      auto bk = params.tma_k.get_slice(_0{}); auto bsk = params.tma_sfk.get_slice(_0{});
      auto bv = params.tma_v.get_slice(_0{}); auto bsv = params.tma_sfv.get_slice(_0{});
      StateQ  wq = cutlass::make_producer_start_state<PipeQ>();
      StateKV wk = cutlass::make_producer_start_state<PipeKV>();
      StateKV wv = cutlass::make_producer_start_state<PipeKV>();

      for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
           work = scheduler.get_next_work(sched_params, work)) {
        int const m_block = get<0>(work.get_block_coord(sched_params));
        Tensor gQ   = local_tile(mQ,   select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFQ = local_tile(mSFQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gK   = local_tile(mK,   select<1, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
        Tensor gSFK = local_tile(mSFK, select<1, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));
        Tensor gV   = local_tile(mV,   make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _0{}));
        Tensor gSFV = local_tile(mSFV, select<1, 2>(TileShape_MNK{}), make_coord(_0{}, _0{}));

        pipeline_q.producer_acquire(wq);
        copy(params.tma_q.with(*pipeline_q.producer_get_barrier(wq), 0), bq.partition_S(gQ), bq.partition_D(sQ));
        copy(params.tma_sfq.with(*pipeline_q.producer_get_barrier(wq), 0), bsq.partition_S(gSFQ), bsq.partition_D(sSFQ));
        ++wq;
        pipeline_k.producer_acquire(wk);
        copy(params.tma_k.with(*pipeline_k.producer_get_barrier(wk), 0), bk.partition_S(gK), bk.partition_D(sK));
        copy(params.tma_sfk.with(*pipeline_k.producer_get_barrier(wk), 0), bsk.partition_S(gSFK), bsk.partition_D(sSFK));
        ++wk;
        pipeline_v.producer_acquire(wv);
        copy(params.tma_v.with(*pipeline_v.producer_get_barrier(wv), 0), bv.partition_S(gV), bv.partition_D(sV));
        copy(params.tma_sfv.with(*pipeline_v.producer_get_barrier(wv), 0), bsv.partition_S(gSFV), bsv.partition_D(sSFV));
        ++wv;
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;     // 0..255
    int const warp = tid / 32, lane = tid % 32;
    TiledMmaQK mma_qk; TiledMmaPV mma_pv;
    auto thr_qk = mma_qk.get_thread_slice(tid);
    auto thr_pv = mma_pv.get_thread_slice(tid);

    // QK operands (smem->reg)
    Tensor tSrQ = thr_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_qk.partition_fragment_B(sK);
    Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_qk);
    Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK, thr_qk);
    auto scQ = make_tiled_copy_A(SmemCopyAtomData{}, mma_qk); auto tscQ = scQ.get_thread_slice(tid);
    auto scK = make_tiled_copy_B(SmemCopyAtomData{}, mma_qk); auto tscK = scK.get_thread_slice(tid);
    auto ts_qk = tile_shape(mma_qk);
    auto scSFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma_qk), make_shape(size<0>(ts_qk), size<2>(ts_qk)));
    auto scSFK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_qk), make_shape(size<1>(ts_qk), size<2>(ts_qk)));
    auto tscSFQ = scSFQ.get_thread_slice(tid); auto tscSFK = scSFK.get_thread_slice(tid);

    // PV operands
    Tensor tOrP  = thr_pv.partition_fragment_A(sP);
    Tensor tOrV  = thr_pv.partition_fragment_B(sV);
    // PV-SFP fragment: alloc from the CANONICAL SFA layout (head_dim==kBlockN so
    // SmemLayoutSFQ is the right (q,keys) SFA layout) -> correct cosize-1 structure.
    // Filled by gather; sfp_coord (below) has the same (_32,_1,_4) shape.
    Tensor tOrSFP = mxfp8::partition_fragment_SFA(sSFQ, thr_pv);
    Tensor tOrSFV = mxfp8::partition_fragment_SFB(sSFV, thr_pv);
    auto scP = make_tiled_copy_A(SmemCopyAtomData{}, mma_pv); auto tscP = scP.get_thread_slice(tid);
    auto scV = make_tiled_copy_B(SmemCopyAtomData{}, mma_pv); auto tscV = scV.get_thread_slice(tid);
    auto ts_pv = tile_shape(mma_pv);
    auto scSFV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_pv), make_shape(size<1>(ts_pv), size<2>(ts_pv)));
    auto tscSFV = scSFV.get_thread_slice(tid);
    // SF gather coords: per PV-SFA fragment slot -> (q, key)
    Tensor sfp_coord = mxfp8::partition_SFA(make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})), thr_pv);

    auto max_op = [](float a, float b) { return fmaxf(a, b); };
    auto add_op = [](float a, float b) { return a + b; };

    Tensor mO = make_tensor(make_gmem_ptr(params.out_O),
                            make_layout(make_shape(int(params.seqlen_q), int(kHeadDim)),
                                        make_stride(int(kHeadDim), _1{})));

    StateQ rq; StateKV rk, rv;
    for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
         work = scheduler.get_next_work(sched_params, work)) {
      int const m_block = get<0>(work.get_block_coord(sched_params));

      { auto t = pipeline_q.consumer_try_wait(rq); pipeline_q.consumer_wait(rq, t);
        copy(scQ, tscQ.partition_S(as_position_independent_swizzle_tensor(sQ)), tscQ.retile_D(tSrQ));
        copy(scSFQ, tscSFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ)), tscSFQ.retile_D(tSrSFQ));
        pipeline_q.consumer_release(rq); ++rq; }
      { auto t = pipeline_k.consumer_try_wait(rk); pipeline_k.consumer_wait(rk, t);
        copy(scK, tscK.partition_S(as_position_independent_swizzle_tensor(sK)), tscK.retile_D(tSrK));
        copy(scSFK, tscSFK.partition_S(as_position_independent_swizzle_tensor(sSFK)), tscSFK.retile_D(tSrSFK));
        pipeline_k.consumer_release(rk); ++rk; }

      // ---- QK ----
      Tensor accS = partition_fragment_C(mma_qk, select<0, 1>(TileShape_MNK{}));   // ((2,2),1,16)
      clear(accS);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tSrQ); ++k)
        cute::gemm(mma_qk, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                   make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accS);

      // ---- softmax (single block: row max/sum, exp2, no rescale) ----
      Tensor accS_rc = make_tensor(accS.data(), make_layout(
          make_layout(get<0, 1>(accS.layout()), get<1>(accS.layout())),
          make_layout(get<0, 0>(accS.layout()), get<2>(accS.layout()))));   // (2 rows, 32 cols)
      constexpr int kNRow = 2, kNCol = 32;
      float row_sum[2];
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        float m_cur = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < kNCol; ++ni) m_cur = fmaxf(m_cur, accS_rc(mi, ni));
        m_cur = quad_reduce(m_cur, max_op);
        float max_scaled = (m_cur == -INFINITY) ? 0.f : m_cur * sm_scale_log2;
        float s = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < kNCol; ++ni) { float p = exp2f(accS_rc(mi, ni) * sm_scale_log2 - max_scaled); accS_rc(mi, ni) = p; s += p; }
        row_sum[mi] = quad_reduce(s, add_op);
      }

      // ---- quantize P (e4m3 + ue8m0 per 32-key block), scatter to smem ----
      // per (row mi, keyblock sfi): amax over the 8 local cols, quad-reduce -> scale_exp.
      Tensor rP = make_fragment_like<Element>(accS);            // e4m3, QK-C layout
      Tensor rP_rc = make_tensor(rP.data(), accS_rc.layout());
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < kNRow; ++mi) {
        int q_local = warp * 16 + (lane / 4) + mi * 8;     // 0..127 within this m_block tile
        CUTLASS_PRAGMA_UNROLL
        for (int sfi = 0; sfi < NKB; ++sfi) {
          float amax = 0.f;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(accS_rc(mi, sfi * 8 + j)));
          amax = quad_reduce(amax, max_op);
          int se = mx_scale_exp(amax);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) rP_rc(mi, sfi * 8 + j) = quant_e4m3(accS_rc(mi, sfi * 8 + j), se);
          if ((lane % 4) == 0) ss.sSFP[q_local * NKB + sfi] = ElementSF::bitcast(uint8_t(se + 127));
        }
      }
      { // dump pre-quant float P + row_sum so the host quantizes the IDENTICAL P
        // (isolates the layout/MMA under test from fp32-vs-fp64 quant-scale noise).
        Tensor gPre = local_tile(make_tensor(make_gmem_ptr(params.outPre),
            make_layout(make_shape(int(params.seqlen_q), int(params.seqlen_k)), make_stride(int(params.seqlen_k), _1{}))),
            select<0, 1>(TileShape_MNK{}), make_coord(m_block, 0));
        copy(accS, thr_qk.partition_C(gPre));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) { int q = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
          if ((lane % 4) == 0) params.outL[q] = row_sum[mi]; }
      }
      // scatter e4m3 P into the swizzled transpose buffer by logical (q,key)
      copy(rP, thr_qk.partition_C(as_position_independent_swizzle_tensor(sP)));

      cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // P/SF visible to all consumers

      // ---- load PV operands ----
      copy(scP, tscP.partition_S(as_position_independent_swizzle_tensor(sP)), tscP.retile_D(tOrP));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tOrSFP); ++i) {
        auto c = sfp_coord(i);
        tOrSFP(i) = ss.sSFP[int(get<0>(c)) * NKB + int(get<1>(c)) / SFVecSize];
      }
      { auto t = pipeline_v.consumer_try_wait(rv); pipeline_v.consumer_wait(rv, t);
        copy(scV, tscV.partition_S(as_position_independent_swizzle_tensor(sV)), tscV.retile_D(tOrV));
        copy(scSFV, tscSFV.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFV.retile_D(tOrSFV));
        pipeline_v.consumer_release(rv); ++rv; }

      // ---- PV ----
      Tensor accO = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));   // (q, head_dim)
      clear(accO);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tOrP); ++k)
        cute::gemm(mma_pv, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                   make_zip_tensor(tOrV(_, _, k), tOrSFV(_, _, k)), accO);

      // ---- epilogue: normalize accO by row_sum in registers, then copy to gmem ----
      Tensor accO_rc = make_tensor(accO.data(), make_layout(
          make_layout(get<0, 1>(accO.layout()), get<1>(accO.layout())),
          make_layout(get<0, 0>(accO.layout()), get<2>(accO.layout()))));
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(accO_rc); ++mi) {
        float inv = (row_sum[mi] == 0.f) ? 0.f : 1.f / row_sum[mi];
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(accO_rc); ++ni) accO_rc(mi, ni) *= inv;
      }
      Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
      copy(accO, thr_pv.partition_C(gO));
    }
  }
}

// ---- host ----
static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }

int main() {
  constexpr int m_block_max = 2, n_block_total = 1;
  const int SQ = m_block_max * kBlockM, SK = n_block_total * kBlockN, HD = kHeadDim;
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  printf("S2: P-requant mxfp8 + PV -> O (single n_block) on WS+TMA+scheduler skeleton\n");
  printf("  seqlen_q=%d (%d m_blocks), seqlen_k=%d (%d n_block), sm_scale=%.4f\n", SQ, m_block_max, SK, n_block_total, sm_scale);

  std::vector<float> dataVals = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  std::vector<uint8_t> hQ(SQ * HD), hK(SK * HD), hV(HD * SK);   // hV is V^T [head_dim, keys]
  std::vector<int> qexp(SQ * NBLK), kexp(SK * NBLK), vexp(HD * NKB);
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) qexp[m * NBLK + b] = ((m + b) % 4) - 1;
  for (int n = 0; n < SK; ++n) for (int b = 0; b < NBLK; ++b) kexp[n * NBLK + b] = (n + 2 * b) % 3;
  for (int h = 0; h < HD; ++h) for (int b = 0; b < NKB; ++b) vexp[h * NKB + b] = ((h + b) % 3) - 1;
  for (int m = 0; m < SQ; ++m) for (int k = 0; k < HD; ++k) hQ[m * HD + k] = e4m3_byte(dataVals[(m * HD + k) % dataVals.size()]);
  for (int n = 0; n < SK; ++n) for (int k = 0; k < HD; ++k) hK[n * HD + k] = e4m3_byte(dataVals[(n * 7 + k * 3) % dataVals.size()]);
  for (int h = 0; h < HD; ++h) for (int n = 0; n < SK; ++n) hV[h * SK + n] = e4m3_byte(dataVals[(h * 5 + n * 2) % dataVals.size()]);

  auto layoutSF = BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim)));   // 128x128
  std::vector<uint8_t> hSFQ(cosize(layoutSF), 0), hSFK(cosize(layoutSF), 0), hSFV(cosize(layoutSF), 0);
  for (int m = 0; m < kBlockM; ++m) for (int b = 0; b < NBLK; ++b) hSFQ[layoutSF(make_coord(m, b * SFVecSize))] = ue8m0_byte_pow2(qexp[m * NBLK + b]);
  for (int n = 0; n < kBlockN; ++n) for (int b = 0; b < NBLK; ++b) hSFK[layoutSF(make_coord(n, b * SFVecSize))] = ue8m0_byte_pow2(kexp[n * NBLK + b]);
  for (int h = 0; h < kHeadDim; ++h) for (int b = 0; b < NKB; ++b) hSFV[layoutSF(make_coord(h, b * SFVecSize))] = ue8m0_byte_pow2(vexp[h * NKB + b]);
  // NOTE: SFQ is per m_block; here both m_blocks reuse the same 128-row SF pattern (qexp depends on m%...), fine for test.
  // Rebuild full SFQ for SQ rows:
  std::vector<uint8_t> hSFQfull(cosize(BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, int(kBlockN), int(kHeadDim)))), 0);
  auto layoutSFQfull = BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, int(kBlockN), int(kHeadDim)));
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) hSFQfull[layoutSFQfull(make_coord(m, b * SFVecSize))] = ue8m0_byte_pow2(qexp[m * NBLK + b]);

  auto deq = [&](uint8_t db, int e) { return double(float(reinterpret_cast<cutlass::float_e4m3_t&>(db))) * std::ldexp(1.0, e); };
  // host mirror of the device mxfp8 quant (bit-identical e4m3/ue8m0 rounding).
  auto host_se = [&](float amax) -> int {
    if (!(amax > 0.f)) return -127; uint32_t b; std::memcpy(&b, &amax, 4); return mx_scale_exp_bits(b);
  };
  // The reference O is computed AFTER the kernel runs, from the device's dumped
  // float P (hPre) quantized with the identical rule -> isolates layout/MMA from
  // fp32-vs-fp64 quant noise. (Filled in main() after copyback.)
  std::vector<double> hO_ref(SQ * HD, 0.0);

  Element *dQ, *dK, *dV; ElementSF *dSFQ, *dSFK, *dSFV; float* dO;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size())); CK(cudaMalloc(&dV, hV.size()));
  CK(cudaMalloc(&dSFQ, hSFQfull.size())); CK(cudaMalloc(&dSFK, hSFK.size())); CK(cudaMalloc(&dSFV, hSFV.size()));
  CK(cudaMalloc(&dO, SQ * HD * sizeof(float)));
  float *dPre, *dL;
  CK(cudaMalloc(&dPre, SQ * SK * sizeof(float))); CK(cudaMalloc(&dL, SQ * sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, hV.data(), hV.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQfull.data(), hSFQfull.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice));

  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(SQ, HD), make_stride(HD, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(SK, HD), make_stride(HD, _1{}));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, SK), make_stride(SK, _1{}));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQfull);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSF);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSFV), layoutSF);
  Params params;
  params.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  params.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}, select<1, 2>(TileShape_MNK{}), _1{});
  params.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  params.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{});
  params.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});
  params.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});
  params.layout_sf = layoutSF; params.layout_sfq = layoutSFQfull;
  params.seqlen_q = SQ; params.seqlen_k = SK; params.sm_scale = sm_scale; params.out_O = dO;
  params.outPre = dPre; params.outL = dL;

  using Scheduler = flashinfer::SingleTileScheduler;
  Scheduler::Arguments sa{m_block_max, 1, SQ, SK, cutlass::FastDivmod(1)};
  Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  dim3 grid = Scheduler::get_grid_dim(sa, 132);
  int smem_bytes = int(sizeof(SharedStorage));
  printf("  grid=(%u,%u,%u) smem=%d threads=%d\n", grid.x, grid.y, grid.z, smem_bytes, kNThreads);
  CK(cudaFuncSetAttribute(s2_kernel<Scheduler>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  s2_kernel<Scheduler><<<grid, kNThreads, smem_bytes>>>(params, sp);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<float> hO(SQ * HD), hPre(SQ * SK), hdL(SQ);
  CK(cudaMemcpy(hO.data(), dO, SQ * HD * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hPre.data(), dPre, SQ * SK * sizeof(float), cudaMemcpyDeviceToHost));   // device pre-quant float P
  CK(cudaMemcpy(hdL.data(), dL, SQ * sizeof(float), cudaMemcpyDeviceToHost));

  // Reference: quantize the device's OWN float P with the identical mxfp8 rule, then O=Pq@Vdq/rowsum.
  for (int m = 0; m < SQ; ++m) {
    float Pq[kBlockN];
    for (int kb = 0; kb < NKB; ++kb) {
      float amax = 0.f;
      for (int j = 0; j < SFVecSize; ++j) amax = std::max(amax, std::fabs(hPre[m * SK + kb * SFVecSize + j]));
      int se = host_se(amax);
      float scale = std::ldexp(1.0f, se), inv = std::ldexp(1.0f, -se);
      for (int j = 0; j < SFVecSize; ++j) {
        int n = kb * SFVecSize + j;
        Pq[n] = float(cutlass::float_e4m3_t(hPre[m * SK + n] * inv)) * scale;   // dequant
      }
    }
    for (int h = 0; h < HD; ++h) {
      double o = 0;
      for (int n = 0; n < SK; ++n) o += double(Pq[n]) * deq(hV[h * SK + n], vexp[h * NKB + n / SFVecSize]);
      hO_ref[m * HD + h] = o / double(hdL[m]);
    }
  }
  double maxrel = 0, maxabs = 0; int bad = 0, fm = -1, fh = -1;
  for (int m = 0; m < SQ; ++m) for (int h = 0; h < HD; ++h) {
    double r = hO_ref[m * HD + h], g = hO[m * HD + h];
    double rel = std::abs(g - r) / std::max(1e-4, std::abs(r));
    maxabs = std::max(maxabs, std::abs(g - r));
    if (rel > maxrel) maxrel = rel;
    if (rel > 5e-3 && std::abs(g - r) > 1e-4) { if (!bad) { fm = m; fh = h; } ++bad; }
  }
  printf("max|rel|=%.3g max|abs|=%.3g bad(>5e-3)=%d / %d\n", maxrel, maxabs, bad, SQ * HD);
  if (bad) printf("  first bad O[%d,%d]: got=%.5f exp=%.5f\n", fm, fh, hO[fm * HD + fh], hO_ref[fm * HD + fh]);
  printf(bad == 0 ? "S2 PASS\n" : "S2 FAIL\n");
  return bad == 0 ? 0 : 1;
}
