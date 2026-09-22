// Pure-CUDA launcher for the S9 ragged+GQA+per-tensor-fp8 prefill kernel (kUniformFp8).
// NO torch headers here -- this TU is compiled by nvcc (its EDG frontend rejects some
// torch 2.11 c10 headers); the pybind glue lives in mxfp8_ragged_ext.cpp (compiled by
// the host gcc). All device buffers are owned by the caller (torch tensors).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "s3_kernel.cuh"

using namespace flashinfer;

extern "C" void s3_ragged_fp8_launch(
    const void* Qd, const void* Kd, const void* Vt,   // e4m3 uint8: [Sq,Hq,D] [Sk,Hkv,D] [Hkv,D,Sk]
    int Sq_pad, int Sk_pad, int Hq, int Hkv, int group,
    float sm_scale, float o_scale, int causal,
    float* out_O, float* out_lse, float* out_l,
    int* work_indptr, int* head_indices, int* qo_tile_indices,
    int* qo_indptr, int* kv_indptr, int* qo_lens, int* kv_lens, int* batch_indices,
    int num_sm, uintptr_t stream_) {
  const int HD = kHeadDim;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

  Element* dQ = reinterpret_cast<Element*>(const_cast<void*>(Qd));
  Element* dK = reinterpret_cast<Element*>(const_cast<void*>(Kd));
  Element* dV = reinterpret_cast<Element*>(const_cast<void*>(Vt));
  // kUniformFp8: no SF tensors exist; the descriptors below point at out_O purely as a
  // valid aligned address (never dereferenced -- the kernel skips every SF TMA load).
  ElementSF* dSF = reinterpret_cast<ElementSF*>(out_O);

  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD, Hq));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD, Hkv));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad, Hkv));
  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(Sq_pad, HD, Hq), make_stride(Hq * HD, _1{}, HD));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Sk_pad, HD, Hkv), make_stride(Hkv * HD, _1{}, HD));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, Sk_pad, Hkv), make_stride(Sk_pad, _1{}, HD * Sk_pad));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSF), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSF), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSF), layoutSFV);

  Params p;
  p.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  p.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  p.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  p.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{});
  p.layout_sfq = layoutSFQ; p.layout_sfv = layoutSFV;
  p.seqlen_q = Sq_pad; p.seqlen_k = Sk_pad; p.n_block_total = Sk_pad / kBlockN;
  p.sm_scale = sm_scale; p.o_scale = o_scale;
  p.num_qo_heads = Hq; p.num_kv_heads = Hkv; p.tile_kv_len = nullptr;
  p.out_O = out_O; p.out_lse = out_lse; p.out_l = out_l;
  p.out_Ppre = nullptr; p.out_Mnb = nullptr; p.out_dbg = nullptr;

  using Sched = BatchPrefillPersistentTileScheduler<int>;
  Sched::Arguments sa;
  sa.work_indptr = work_indptr; sa.head_indices = head_indices; sa.qo_tile_indices = qo_tile_indices;
  sa.qo_indptr = qo_indptr; sa.kv_indptr = kv_indptr; sa.qo_lens = qo_lens; sa.kv_lens = kv_lens;
  sa.batch_indices = batch_indices; sa.group_size_fastdiv = cutlass::FastDivmod(group);
  sa.num_qo_heads = Hq;
  dim3 grid = Sched::get_grid_dim(sa, num_sm);
  typename Sched::Params sp = Sched::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorage));

  if (causal) {
    cudaFuncSetAttribute(s3_kernel<Sched, true, SFSource::kUniformFp8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    s3_kernel<Sched, true, SFSource::kUniformFp8><<<grid, kNThreads, smem, stream>>>(p, sp);
  } else {
    cudaFuncSetAttribute(s3_kernel<Sched, false, SFSource::kUniformFp8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    s3_kernel<Sched, false, SFSource::kUniformFp8><<<grid, kNThreads, smem, stream>>>(p, sp);
  }
}

extern "C" int s3_num_sm() {
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  if (const char* e = getenv("S3_NUM_SM")) n = std::atoi(e);
  return n;
}
