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

Upstream tracking: pytorch/ao#4932, pytorch/pytorch#198126.

## Contents

- `dense_blockscaled_gemm_persistent_pingpong.py` — the kernel, forked from
  CUTLASS 4.8 `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`.
  Our changes vs upstream (see also `patch_v1.py`):
  - CLI majors relaxed (a_major k/m, b_major k/n, c_major n/m)
  - `b_manual_load` mode (fp8 + n-major B): warps 9-11 (idle in the stock
    example) copy B manually: `ld.global.v4` 16B gmem N'-runs -> 16x scalar
    `st.shared` into the standard swizzled K-major SMEM layout
    (transpose-by-placement), guarded by an extra `PipelineAsync`
    (producer 96 threads, consumer one MMA warpgroup). MMA/ldmatrix/epilogue
    untouched; TMA path for A/SFA/SFB and the whole TN path byte-identical.
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
| DSL k,k (TN) | 754us | stock example perf |
| DSL k,n (NT, v1 manual-B) | 2409us | v0 scalar was 5255us |
| bf16 cuBLAS | 1674us | target to beat |

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

- Manual B store side is 16x scalar `st.shared` per 16B gmem load
  (byte-granularity transpose cannot vectorize both sides — K-major SMEM 16B
  units are 16 consecutive k', gmem runs are 16 consecutive n'). Options:
  (a) PRMT register transpose + `st.shared.u32` (~1.5x, est. 1400-1600us);
  (b) ldmatrix/stmatrix u16-trick SMEM transpose staging (est. 800-1100us);
  (c) pull warp 8 into the manual copy (marginal).
- a_major="m" (wgrad-style transposed mat_a) still uses the slow v0 universal
  fallback; same manual-load treatment applies.
- Scheduler: StaticPersistent grid-stride is fine for dense; if we add L2
  raster swizzle / tail pairing, keep the deterministic zero-coordination
  principle (cf. include/flashinfer/attention/blackwell/prefill/flashinfer_tile_scheduler.cuh).
- SF (scale factor) handling is orientation-complete already: for n-major B
  the kernel consumes the second scale frame (sm from torchao's
  `triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale`), layouts match
  cuBLAS SWIZZLE_32_4_4 / `to_blocked`.
