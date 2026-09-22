# SM120 orientation-flexible MXFP8 block-scaled GEMM (CuTe DSL)

Prototype for single-qdata MXFP8 training GEMMs on Blackwell GeForce (sm_120).

## Why

cuBLAS block-scaled FP8 GEMM is TN-only on compute capability 12.x (cuBLAS 13.4
docs: "A must be transposed and B non-transposed on Ada 8.9, Hopper 9.0, and
Blackwell GeForce 12.x"; 10.x excluded). Training backward needs NT/NN
orientations, so everyone (TE v2.19 2D-quant included, torchao) stores two
quantized copies per weight (~1.03x bf16). With a kernel that consumes MN-major
operands directly, the 32x32 square-block format (torchao #4777:
one qdata + dual swizzled scale frames) suffices: ~0.53x bf16.

Upstream tracking: pytorch/ao#4932, pytorch/pytorch#198126 (both closed by us —
TN-only is the long-standing FP8 restriction unchanged since sm_90; dual-copy
quantized weights are established industry practice, hence this kernel).

## Contents

- `dense_blockscaled_gemm_persistent_pingpong.py` — the kernel, forked from
  CUTLASS 4.8 `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`.
  Our changes vs upstream (see also `patch_v1.py`):
  - CLI majors relaxed (a_major k/m, b_major k/n, c_major n/m)
  - `b_manual_load` mode (fp8 + n-major B): warps 9-11 (idle in the stock
    example) copy B manually into the standard swizzled K-major SMEM layout
    (transpose-by-placement), guarded by an extra `PipelineAsync`
    (producer 96 threads, consumer one MMA warpgroup). MMA/ldmatrix/epilogue
    untouched; TMA path for A/SFA/SFB and the whole TN path byte-identical.
    v3 details: each thread transposes an 8(k)x16(n) half-block in registers
    via PRMT (2-level byte gather), stores u64 via inline-asm
    `st.shared.v2.u32`; loads are `.align(16)`-annotated LDG.E.128 with an
    n-fast lane mapping (coalescing beats bank spread: loads were the real
    bottleneck). Register budget rebalanced when manual: MMA 232->216,
    load wg 40->64 (setmaxregister values must be multiples of 8).
  - tx_count excludes B in manual mode; dummy TMA-B atom; divisibility=16
    marking for n-major B; N%128==0 && K%128==0 gate for the manual path.
- `blockscaled_gemm_dispatch.py` — shared dispatch helpers (unmodified).
- `patch_v1.py` — consolidated patch script (apply to a pristine CUTLASS 4.8
  copy to reproduce the kernel; note: a few follow-up fixes were applied
  afterwards by hand, see git history).

## Numbers (RTX 5060 Ti, 4096x3072x3072 MXFP8 e4m3/e8m0-vec32 -> bf16)

| path | time | note |
|---|---|---|
| cuBLAS TN (torch._scaled_mm) | 542us | reference |
| DSL k,k (TN) | 754-760us | stock example perf (unchanged) |
| DSL k,n (NT, v3 manual-B) | **1390-1486us** | beats bf16; v1 2409us, v0 5255us |
| bf16 cuBLAS | 1674us | beaten |

Correctness: all of k,k / k,n / m,k / m,n pass the built-in reference check.

## Run

```
python mxfp8_gemm/dense_blockscaled_gemm_persistent_pingpong.py \
  --mnkl 4096,3072,3072,1 --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
  --sf_dtype Float8E8M0FNU --sf_vec_size 32 --c_dtype BFloat16 \
  --b_major n --warmup_iterations 5 --iterations 20
```

Needs: `nvidia-cutlass-dsl` (tested 4.7.1), torch, sm_120 GPU.

## Known bottlenecks / TODO

- v3 store side still ~4-way bank-conflicted (n-stride 128B aliases banks);
  conflict-free stores need k-fast lanes, which breaks gmem coalescing —
  resolving both requires cross-thread exchange (shfl butterfly or an SMEM
  staging + ldmatrix/stmatrix u16-trick round trip, est. 800-1100us).
- Residual register spills in producer (LDL/STL ~18, STACK 96) at 64 regs.
- DSL pitfalls encoded in the code comments: `cute.recast_tensor` on a
  swizzled SMEM tensor silently yields a wrong layout (use a
  swizzle-in-layout view + `crd2idx` instead); `.load()` vectorization and
  `recast_ptr` drops alignment info unless `.align(16)` re-annotates;
  `nvvm.setmaxregister` values must be multiples of 8; verify widths in SASS
  (`nvdisasm`), never trust source-level intuition.
- a_major="m" (wgrad-style transposed mat_a) still uses the slow v0 universal
  fallback; same manual-load treatment applies.
- Scheduler: StaticPersistent grid-stride is fine for dense; if we add L2
  raster swizzle / tail pairing, keep the deterministic zero-coordination
  principle (cf. include/flashinfer/attention/blackwell/prefill/flashinfer_tile_scheduler.cuh).
- SF (scale factor) handling is orientation-complete already: for n-major B
  the kernel consumes the second scale frame (sm from torchao's
  `triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale`), layouts match
  cuBLAS SWIZZLE_32_4_4 / `to_blocked`.
