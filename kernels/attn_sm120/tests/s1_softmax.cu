// S1: online softmax layered onto the S0c skeleton (WS + PipelineTmaAsync ring +
// FlashInfer SingleTileScheduler). The consumer warpgroups run the QK MMA exactly
// as S0c (bit-exact, written out as a regression check), then fold each n_block
// into a running online softmax (row max / row sum, exp2, causal mask, running
// rescale of row_sum) -- the SageAttention online_softmax with the nvfp4 quant
// half dropped. Validates final per-query row_max (raw) and row_sum (== softmax
// denominator) vs an fp64 reference, for both non-causal and causal, over
// multiple n_blocks so the running rescale is genuinely exercised.
//
// accum layout (printed): ((_2,_2),_1,_16) -- standard cute m16n8.
//   reduction view = ((row=2,MMA_M=1),(col=2,MMA_N=16)) = 2 rows x 32 cols / thr.
//   m = m_block*kBlockM + warp*16 + lane/4 + mi*8
//   col = (ni/2)*8 + (lane%4)*2 + (ni%2)     [n_global = nb*kBlockN + col]
//   the 128 cols of a row live in a 4-thread quad (lane%4) -> Allreduce<4>.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s1_softmax.cu -o tests/s1_softmax

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
constexpr int kHeadDim = 128, kBlockM = 128, kBlockN = 128, SFVecSize = 32, kStages = 2;
constexpr int NBLK = kHeadDim / SFVecSize;
constexpr int kNWarps = 12;
constexpr int kNThreads = kNWarps * 32;          // 384
constexpr int NumMmaThreads = 256;               // 2 consumer warpgroups
constexpr int NumCopyThreads = 128;              // 1 producer warpgroup
constexpr float kLog2e = 1.4426950408889634f;

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using TiledMmaQK = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kHeadDim>>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomQ{}, make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

// SF smem atom (cutlass-canonical, from sm120 blockscaled builder) + stage dim for K.
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
using SmemLayoutAtomSFQ = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
using sSFB_shapeN = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFB_strideK{})));
using SmemLayoutSFQ = SmemLayoutAtomSFQ;
using SmemLayoutSFK = decltype(make_layout(
    append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))));

using SmemCopyAtomQ  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

using LayoutSF = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kHeadDim))));

using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kBlockM), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));
using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(8 * kBlockN), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{}));
using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{}));
using TMA_SFK = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{}));

using MainloopPipeline  = cutlass::PipelineTmaAsync<kStages>;
using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
using PipelineState  = cutlass::PipelineState<kStages>;
using PipelineStateQ = cutlass::PipelineState<1>;

struct Params {
  TMA_Q tma_q; TMA_K tma_k; TMA_SFQ tma_sfq; TMA_SFK tma_sfk;
  LayoutSF layout_sfq; LayoutSF layout_sfk;
  int seqlen_q; int seqlen_k; int n_block_total;
  float sm_scale;
  float* out_S;   // [seqlen_q, seqlen_k] raw QK (regression check)
  float* out_m;   // [seqlen_q] row_max (raw)
  float* out_l;   // [seqlen_q] row_sum (softmax denominator)
};

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element,   cute::cosize_v<SmemLayoutQ>> sQ;
  alignas(1024) cute::ArrayEngine<Element,   cute::cosize_v<SmemLayoutK>> sK;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> sSFQ;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> sSFK;
  struct {
    alignas(16) MainloopPipelineQ::SharedStorage pipeline_q;
    alignas(16) MainloopPipeline::SharedStorage  pipeline_k;
  };
};

static constexpr uint32_t TmaBytesQ =
    cute::cosize_v<SmemLayoutQ> * sizeof(Element) +
    cute::cosize_v<SmemLayoutSFQ> * sizeof(ElementSF);
static constexpr uint32_t TmaBytesK =
    cute::cosize_v<SmemLayoutK> / kStages * sizeof(Element) +
    cute::cosize_v<SmemLayoutSFK> / kStages * sizeof(ElementSF);

template <class Op>
__device__ __forceinline__ float quad_reduce(float v, Op op) {
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 2));
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 1));
  return v;
}

template <typename TileScheduler, bool Causal>
__global__ void __launch_bounds__(kNThreads, 1)
s1_kernel(CUTE_GRID_CONSTANT Params const params,
          CUTE_GRID_CONSTANT typename TileScheduler::Params const sched_params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  int const warp_group_idx = cutlass::canonical_warp_group_idx();   // 0=producer,1/2=consumer
  int const warp_idx_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const lane_predicate = cute::elect_one_sync();

  typename MainloopPipelineQ::Params pq;
  pq.role = (warp_group_idx == 0) ? MainloopPipelineQ::ThreadCategory::Producer
                                  : MainloopPipelineQ::ThreadCategory::Consumer;
  pq.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pq.num_consumers = NumMmaThreads;
  pq.transaction_bytes = TmaBytesQ;
  MainloopPipelineQ pipeline_q(ss.pipeline_q, pq, Shape<_1, _1, _1>{});

  typename MainloopPipeline::Params pk;
  pk.role = (warp_group_idx == 0) ? MainloopPipeline::ThreadCategory::Producer
                                  : MainloopPipeline::ThreadCategory::Consumer;
  pk.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pk.num_consumers = NumMmaThreads;
  pk.transaction_bytes = TmaBytesK;
  MainloopPipeline pipeline_k(ss.pipeline_k, pk, Shape<_1, _1, _1>{});

  __syncthreads();

  Tensor sQ   = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sK   = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutK{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFQ{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFK{});

  int const n_block_total = params.n_block_total;
  float const sm_scale_log2 = params.sm_scale * kLog2e;
  TileScheduler scheduler;

  if (warp_group_idx == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_idx_in_wg == 0 && lane_predicate) {
      Tensor mQ   = params.tma_q.get_tma_tensor(make_shape(int(params.seqlen_q), int(kHeadDim)));
      Tensor mK   = params.tma_k.get_tma_tensor(make_shape(int(params.seqlen_k), int(kHeadDim)));
      Tensor mSFQ = params.tma_sfq.get_tma_tensor(shape(params.layout_sfq));
      Tensor mSFK = params.tma_sfk.get_tma_tensor(shape(params.layout_sfk));
      auto bq = params.tma_q.get_slice(_0{}); auto bsq = params.tma_sfq.get_slice(_0{});
      auto bk = params.tma_k.get_slice(_0{}); auto bsk = params.tma_sfk.get_slice(_0{});

      PipelineStateQ write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
      PipelineState  write_k = cutlass::make_producer_start_state<MainloopPipeline>();

      for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
           work = scheduler.get_next_work(sched_params, work)) {
        int const m_block = get<0>(work.get_block_coord(sched_params));
        int const n_block_max = Causal
            ? cute::min(n_block_total, ((m_block + 1) * kBlockM + kBlockN - 1) / kBlockN)
            : n_block_total;

        Tensor gQ   = local_tile(mQ,   select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFQ = local_tile(mSFQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gK   = local_tile(mK,   select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));   // (N,K,nb)
        Tensor gSFK = local_tile(mSFK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor tQgQ = bq.partition_S(gQ); Tensor tQsQ = bq.partition_D(sQ);
        Tensor tQgSFQ = bsq.partition_S(gSFQ); Tensor tQsSFQ = bsq.partition_D(sSFQ);
        Tensor tKgK = group_modes<0, 3>(bk.partition_S(gK)); Tensor tKsK = group_modes<0, 3>(bk.partition_D(sK));
        Tensor tKgSFK = group_modes<0, 3>(bsk.partition_S(gSFK)); Tensor tKsSFK = group_modes<0, 3>(bsk.partition_D(sSFK));

        pipeline_q.producer_acquire(write_q);
        copy(params.tma_q.with(*pipeline_q.producer_get_barrier(write_q), 0), tQgQ, tQsQ);
        copy(params.tma_sfq.with(*pipeline_q.producer_get_barrier(write_q), 0), tQgSFQ, tQsSFQ);
        ++write_q;
        for (int nb = 0; nb < n_block_max; ++nb) {
          pipeline_k.producer_acquire(write_k);
          copy(params.tma_k.with(*pipeline_k.producer_get_barrier(write_k), 0),
               tKgK(_, nb), tKsK(_, write_k.index()));
          copy(params.tma_sfk.with(*pipeline_k.producer_get_barrier(write_k), 0),
               tKgSFK(_, nb), tKsSFK(_, write_k.index()));
          ++write_k;
        }
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;       // 0..255
    int const warp = tid / 32, lane = tid % 32;
    TiledMmaQK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
    Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_mma);
    Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK(_, _, _0{}), thr_mma);

    auto sc_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma);
    auto thr_sc_Q = sc_Q.get_thread_slice(tid);
    auto sc_K = make_tiled_copy_B(SmemCopyAtomQ{}, tiled_mma);
    auto thr_sc_K = sc_K.get_thread_slice(tid);
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto sc_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(tiled_mma),
                                       make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_sc_SFQ = sc_SFQ.get_thread_slice(tid);
    auto sc_SFK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(tiled_mma),
                                       make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_sc_SFK = sc_SFK.get_thread_slice(tid);

    Tensor mS = make_tensor(make_gmem_ptr(params.out_S),
                            make_layout(make_shape(int(params.seqlen_q), int(params.seqlen_k)),
                                        make_stride(int(params.seqlen_k), _1{})));

    auto max_op = [](float a, float b) { return fmaxf(a, b); };
    auto add_op = [](float a, float b) { return a + b; };

    PipelineStateQ read_q;
    PipelineState  read_k;
    for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
         work = scheduler.get_next_work(sched_params, work)) {
      int const m_block = get<0>(work.get_block_coord(sched_params));
      int const n_block_max = Causal
          ? cute::min(n_block_total, ((m_block + 1) * kBlockM + kBlockN - 1) / kBlockN)
          : n_block_total;

      // online-softmax row state: this thread owns 2 rows (mi = 0,1).
      float row_max[2] = {-INFINITY, -INFINITY};
      float row_sum[2] = {0.f, 0.f};

      {  // Q for this tile
        auto tok = pipeline_q.consumer_try_wait(read_q);
        pipeline_q.consumer_wait(read_q, tok);
        copy(sc_Q, thr_sc_Q.partition_S(as_position_independent_swizzle_tensor(sQ)), thr_sc_Q.retile_D(tSrQ));
        copy(sc_SFQ, thr_sc_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ)), thr_sc_SFQ.retile_D(tSrSFQ));
        pipeline_q.consumer_release(read_q);
        ++read_q;
      }

      for (int nb = 0; nb < n_block_max; ++nb) {
        auto tok = pipeline_k.consumer_try_wait(read_k);
        pipeline_k.consumer_wait(read_k, tok);
        int stage = read_k.index();
        copy(sc_K, thr_sc_K.partition_S(as_position_independent_swizzle_tensor(sK(_, _, stage))), thr_sc_K.retile_D(tSrK));
        copy(sc_SFK, thr_sc_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK(_, _, stage))), thr_sc_SFK.retile_D(tSrSFK));

        Tensor gS_blk = local_tile(mS, select<0, 1>(TileShape_MNK{}), make_coord(m_block, nb));
        Tensor tSgS = thr_mma.partition_C(gS_blk);
        Tensor accum = thr_mma.partition_fragment_C(gS_blk);   // ((_2,_2),_1,_16)
        clear(accum);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tSrQ); ++k)
          cute::gemm(tiled_mma, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                     make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accum);
        copy(accum, tSgS);   // raw QK regression check (S0c parity)
        pipeline_k.consumer_release(read_k);
        ++read_k;

        // ---- online softmax (no quant) ----
        // reduction view: ((row=2,MMA_M=1),(col=2,MMA_N=16)) -> acc_rc(mi, ni)
        Tensor acc_rc = make_tensor(accum.data(), make_layout(
            make_layout(get<0, 1>(accum.layout()), get<1>(accum.layout())),
            make_layout(get<0, 0>(accum.layout()), get<2>(accum.layout()))));
        constexpr int kNRow = 2, kNCol = 32;

        if constexpr (Causal) {
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < kNRow; ++mi) {
            int m_global = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < kNCol; ++ni) {
              int col = (ni / 2) * 8 + (lane % 4) * 2 + (ni % 2);
              int n_global = nb * kBlockN + col;
              if (n_global > m_global) acc_rc(mi, ni) = -INFINITY;
            }
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          float m_prev = row_max[mi];
          float m_cur = m_prev;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) m_cur = fmaxf(m_cur, acc_rc(mi, ni));
          m_cur = quad_reduce(m_cur, max_op);
          row_max[mi] = m_cur;

          float scores_scale = exp2f((m_prev - m_cur) * sm_scale_log2);  // first tile: ->0
          float max_scaled = (m_cur == -INFINITY) ? 0.f : (m_cur * sm_scale_log2);
          row_sum[mi] *= scores_scale;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) {
            float p = exp2f(acc_rc(mi, ni) * sm_scale_log2 - max_scaled);
            acc_rc(mi, ni) = p;
            row_sum[mi] += p;
          }
        }
      }

      // finalize: complete the row sum across the 4-thread quad.
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 2; ++mi) {
        row_sum[mi] = quad_reduce(row_sum[mi], add_op);
        int m_global = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
        if (m_global < params.seqlen_q && (lane % 4) == 0) {
          params.out_m[m_global] = row_max[mi];
          params.out_l[m_global] = row_sum[mi];
        }
      }
    }
  }
}

// ---- host ----
static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }

template <typename Scheduler, bool Causal>
static int run_case(const char* tag, Params params_base, typename Scheduler::Arguments sched_args,
                    int SQ, int SK, float sm_scale,
                    const std::vector<double>& refS) {
  Params params = params_base;
  typename Scheduler::Params sched_params = Scheduler::to_underlying_arguments(sched_args);
  dim3 grid = Scheduler::get_grid_dim(sched_args, /*num_sm=*/132);

  int smem_bytes = int(sizeof(SharedStorage));
  CK(cudaFuncSetAttribute(s1_kernel<Scheduler, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  s1_kernel<Scheduler, Causal><<<grid, kNThreads, smem_bytes>>>(params, sched_params);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  // fp64 reference for row_max (raw) and row_sum (exp2 denominator).
  const float sm_scale_log2 = sm_scale * kLog2e;
  std::vector<double> refM(SQ), refL(SQ);
  for (int m = 0; m < SQ; ++m) {
    double mx = -INFINITY;
    for (int n = 0; n < SK; ++n) {
      if (Causal && n > m) continue;
      mx = std::max(mx, refS[m * SK + n]);
    }
    double sum = 0;
    for (int n = 0; n < SK; ++n) {
      if (Causal && n > m) continue;
      sum += std::exp2((double)sm_scale_log2 * (refS[m * SK + n] - mx));
    }
    refM[m] = mx; refL[m] = sum;
  }

  std::vector<float> hM(SQ), hL(SQ);
  CK(cudaMemcpy(hM.data(), params.out_m, SQ * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hL.data(), params.out_l, SQ * sizeof(float), cudaMemcpyDeviceToHost));

  double maxrel_m = 0, maxrel_l = 0; int badm = 0, badl = 0, fm = -1, fl = -1;
  for (int m = 0; m < SQ; ++m) {
    double rm = std::abs(hM[m] - refM[m]) / std::max(1e-6, std::abs(refM[m]));
    double rl = std::abs(hL[m] - refL[m]) / std::max(1e-6, std::abs(refL[m]));
    if (rm > maxrel_m) maxrel_m = rm;
    if (rl > maxrel_l) maxrel_l = rl;
    if (rm > 2e-3) { if (!badm) fm = m; ++badm; }
    if (rl > 2e-3) { if (!badl) fl = m; ++badl; }
  }
  printf("  [%s] max|rel| m=%.3g (bad %d) l=%.3g (bad %d)\n", tag, maxrel_m, badm, maxrel_l, badl);
  if (badm) printf("    first bad m[%d]: got=%.5f exp=%.5f\n", fm, hM[fm], refM[fm]);
  if (badl) printf("    first bad l[%d]: got=%.5f exp=%.5f\n", fl, hL[fl], refL[fl]);
  return (badm == 0 && badl == 0) ? 0 : 1;
}

int main() {
  constexpr int m_block_max = 2, n_block_total = 4;
  const int SQ = m_block_max * kBlockM, SK = n_block_total * kBlockN, HD = kHeadDim;
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  printf("S1: online softmax on WS+TMA+FlashInfer-scheduler skeleton (QK -> m/l)\n");
  printf("  seqlen_q=%d (%d m_blocks), seqlen_k=%d (%d n_blocks), kStages=%d, sm_scale=%.4f\n",
         SQ, m_block_max, SK, n_block_total, kStages, sm_scale);

  std::vector<float> dataVals = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  std::vector<uint8_t> hQ(SQ * HD), hK(SK * HD);
  std::vector<int> qexp(SQ * NBLK), kexp(SK * NBLK);
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) qexp[m * NBLK + b] = ((m + b) % 4) - 1;
  // Later K-blocks are scaled up (+ n/kBlockN) so S grows across n_blocks and the
  // running max strictly increases -> the online-softmax rescale path is exercised.
  for (int n = 0; n < SK; ++n) for (int b = 0; b < NBLK; ++b) kexp[n * NBLK + b] = (n + 2 * b) % 3 + (n / kBlockN);
  for (int m = 0; m < SQ; ++m) for (int k = 0; k < HD; ++k) hQ[m * HD + k] = e4m3_byte(dataVals[(m * HD + k) % dataVals.size()]);
  for (int n = 0; n < SK; ++n) for (int k = 0; k < HD; ++k) hK[n * HD + k] = e4m3_byte(dataVals[(n * 7 + k * 3) % dataVals.size()]);

  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(SQ, kBlockN, HD));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(SK, kBlockN, HD));
  std::vector<uint8_t> hSFQ(cosize(layoutSFQ), 0), hSFK(cosize(layoutSFK), 0);
  for (int m = 0; m < SQ; ++m) for (int b = 0; b < NBLK; ++b) hSFQ[layoutSFQ(make_coord(m, b * SFVecSize))] = ue8m0_byte_pow2(qexp[m * NBLK + b]);
  for (int n = 0; n < SK; ++n) for (int b = 0; b < NBLK; ++b) hSFK[layoutSFK(make_coord(n, b * SFVecSize))] = ue8m0_byte_pow2(kexp[n * NBLK + b]);

  std::vector<double> refS(SQ * SK, 0.0);
  auto deq = [&](uint8_t db, int e) { return double(float(reinterpret_cast<cutlass::float_e4m3_t&>(db))) * std::ldexp(1.0, e); };
  for (int m = 0; m < SQ; ++m) for (int n = 0; n < SK; ++n) {
    double s = 0;
    for (int k = 0; k < HD; ++k) s += deq(hQ[m * HD + k], qexp[m * NBLK + k / SFVecSize]) * deq(hK[n * HD + k], kexp[n * NBLK + k / SFVecSize]);
    refS[m * SK + n] = s;
  }

  Element *dQ, *dK; ElementSF *dSFQ, *dSFK; float *dS, *dM, *dL;
  CK(cudaMalloc(&dQ, hQ.size())); CK(cudaMalloc(&dK, hK.size()));
  CK(cudaMalloc(&dSFQ, hSFQ.size())); CK(cudaMalloc(&dSFK, hSFK.size()));
  CK(cudaMalloc(&dS, SQ * SK * sizeof(float)));
  CK(cudaMalloc(&dM, SQ * sizeof(float))); CK(cudaMalloc(&dL, SQ * sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice));

  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(SQ, HD), make_stride(HD, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(SK, HD), make_stride(HD, _1{}));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Params params;
  params.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  params.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  params.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), _1{});
  params.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), _1{});
  params.layout_sfq = layoutSFQ; params.layout_sfk = layoutSFK;
  params.seqlen_q = SQ; params.seqlen_k = SK; params.n_block_total = n_block_total;
  params.sm_scale = sm_scale; params.out_S = dS; params.out_m = dM; params.out_l = dL;

  using Scheduler = flashinfer::SingleTileScheduler;
  Scheduler::Arguments sched_args{
      /*num_qo_tiles=*/m_block_max, /*num_qo_heads=*/1, /*qo_len=*/SQ, /*kv_len=*/SK,
      /*group_size_fastdiv=*/cutlass::FastDivmod(1)};
  dim3 grid = Scheduler::get_grid_dim(sched_args, 132);
  printf("  grid = (%u, %u, %u), smem = %zu bytes, threads = %d\n",
         grid.x, grid.y, grid.z, sizeof(SharedStorage), kNThreads);

  int rc = 0;
  rc |= run_case<Scheduler, false>("non-causal", params, sched_args, SQ, SK, sm_scale, refS);
  rc |= run_case<Scheduler, true >("causal    ", params, sched_args, SQ, SK, sm_scale, refS);

  printf(rc == 0 ? "S1 PASS\n" : "S1 FAIL\n");
  return rc;
}
