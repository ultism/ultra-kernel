"""
Proper benchmark of the *real* SageAttention3 Blackwell TMA + warp-specialized
attention kernel (nvfp4), as opposed to the synthetic FMA microbench in
bench/sched_ab.cu.

Purpose: close the open question from docs/gotcha.md -- the FMA microbench
*could not* measure persistent scheduling's real benefit (amortizing the
expensive TMA + mbarrier + producer/consumer prologue), so it only argued
against persistent for a cheap-prologue non-WS kernel. This runs the actual
WS+TMA kernel so we can A/B persistent vs non-persistent where the prologue
cost is real -- by toggling launch.h:41/42 (SingleTileScheduler vs
StaticPersistentTileScheduler) and rebuilding fp4attn_cuda between runs.

Each run: (1) validate output vs SDPA on the SAME inputs (FP4 -> loose cosine
tolerance, like SageAttention's own tests), then (2) time real workloads and
report TFLOPS. The --tag labels the build variant in the output.

Input layout for sageattn3_blackwell is (B, H, L, D), fp16/bf16.

Usage:
  PYTHONPATH=<sage3_dir> python bench/bench_sage3_real.py --tag persistent
"""
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

from sageattn3 import sageattn3_blackwell


def calc_diff(x, y):
    # SageAttention's own metric: 1 - cosine similarity (0 == identical).
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denom
    return (1 - sim).item()


def bench_ms(fn, warmup=10, iters=50):
    # L2 flush + CUDA-event timing, min over a few measured windows.
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        cache.zero_()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def make_qkv(b, h, l, d, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, h, l, d, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(b, h, l, d, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(b, h, l, d, dtype=dtype, device="cuda", generator=g)
    return q, k, v


def validate(b, h, l, d, dtype, is_causal):
    q, k, v = make_qkv(b, h, l, d, dtype)
    ref = sdpa(q.float(), k.float(), v.float(), is_causal=is_causal).to(dtype)
    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
    diff = calc_diff(out.float(), ref.float())
    # relative L2 as a second, scale-aware view
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    return diff, rel.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="?", help="build variant label")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    b, h, d = args.batch, args.heads, args.head_dim
    seqlens = [2048, 4096, 8192, 16384]

    print(f"=== SageAttention3 blackwell  [variant: {args.tag}] ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"B={b} H={h} D={d} dtype={args.dtype}")

    # --- correctness gate (small shape, both masks) -------------------------
    print("\n-- correctness vs SDPA (FP4: expect 1-cos a few e-3..e-2) --")
    for causal in (False, True):
        diff, rel = validate(2, 8, 2048, d, dtype, causal)
        flag = "OK" if diff < 5e-2 else "!!"
        print(f"  causal={causal!s:5s}  1-cos={diff:.3e}  relL2={rel:.3e}  [{flag}]")

    # --- timing sweep -------------------------------------------------------
    for causal in (False, True):
        print(f"\n-- timing  is_causal={causal} --")
        print(f"  {'seqlen':>7} {'ms':>9} {'TFLOPS':>9}")
        for l in seqlens:
            q, k, v = make_qkv(b, h, l, d, dtype)
            # 2 matmuls * 2 flops, halve for causal triangular
            flops = 4 * h * b * d * l * l / (2 if causal else 1)
            try:
                fn = lambda: sageattn3_blackwell(q, k, v, is_causal=causal)
                ms = bench_ms(fn)
                print(f"  {l:>7} {ms:>9.4f} {flops/ms*1e-9:>9.1f}")
            except Exception as ex:
                print(f"  {l:>7}  FAILED: {ex}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
