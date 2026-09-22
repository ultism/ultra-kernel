"""Validation of mxfp8_ragged_ext (ragged varlen + GQA + per-tensor fp8, kUniformFp8 path)
against an independent fp64 dequant oracle replaying the kernel's online softmax at the
kernel's 64-key block granularity. Same varlen batch as tests/bench_ragged.cu.

Run:  /root/vllm-omni/.venv/bin/python tests/test_ragged_ext.py
"""
import os, pathlib, sys, torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load

ROOT = pathlib.Path(__file__).resolve().parents[1]
_ext = None

def ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="mxfp8_ragged_ext",
            sources=[str(ROOT / "tests" / "csrc" / "mxfp8_ragged_ext.cpp"),
                     str(ROOT / "tests" / "csrc" / "mxfp8_ragged_kernel.cu")],
            extra_include_paths=[str(ROOT / "tmp" / "cutlass" / "include"),
                                 str(ROOT / "include"), str(ROOT / "tests")],
            extra_cflags=["-std=c++17", "-O2", "-fpermissive"],
            extra_cuda_cflags=["-std=c++17", "-O2", "-gencode", "arch=compute_120a,code=sm_120a",
                               "--expt-relaxed-constexpr", "--expt-extended-lambda"],
            verbose=False,
        )
    return _ext


def per_tensor_e4m3(X):
    s = X.abs().max().clamp_min(1e-12) / 448.0
    return (X / s).to(torch.float8_e4m3fn), s.double()


def dequant_per_tensor(d, s):
    return d.view(torch.uint8).view(torch.float8_e4m3fn).double() * s


def main():
    torch.manual_seed(0)
    dev = "cuda"
    D, Hq, Hkv = 128, 8, 2
    lens = [512, 1024, 768, 1536, 2048, 640, 1280, 896, 384, 1792, 512, 2560, 1024, 768, 2048, 1100]
    causal = True
    group = Hq // Hkv

    # --- one batch-wide scale per tensor (vllm-style per-layer scalars) ---
    Q = [torch.randn(L, Hq, D, device=dev) for L in lens]
    K = [torch.randn(L, Hkv, D, device=dev) for L in lens]
    V = [torch.randn(L, Hkv, D, device=dev) for L in lens]
    qd_all = torch.cat(Q); kd_all = torch.cat(K); vd_all = torch.cat(V)
    qd, qs = per_tensor_e4m3(qd_all)
    kd, ks = per_tensor_e4m3(kd_all)
    vd, vs = per_tensor_e4m3(vd_all)

    # --- pad each request to 128-multiples, pack, build indptr ---
    pad = lambda L: (L + 127) // 128 * 128
    qo_pad = [pad(L) for L in lens]
    kv_pad = qo_pad
    Sq_pad, Sk_pad = sum(qo_pad), sum(kv_pad)
    Qp = torch.zeros(Sq_pad, Hq, D, dtype=torch.uint8, device=dev)
    Kp = torch.zeros(Sk_pad, Hkv, D, dtype=torch.uint8, device=dev)
    Vp = torch.zeros(Hkv, D, Sk_pad, dtype=torch.uint8, device=dev)   # DIM-major
    qo = kv = 0
    qo_indptr, kv_indptr = [0], [0]
    for i, L in enumerate(lens):
        Qp[qo:qo + L] = qd.view(torch.uint8)[sum(lens[:i]):sum(lens[:i]) + L]
        Kp[kv:kv + L] = kd.view(torch.uint8)[sum(lens[:i]):sum(lens[:i]) + L]
        # V: [L, Hkv, D] uint8 -> per head [D, L] into the padded slab
        Vp[:, :, kv:kv + L] = vd.view(torch.uint8)[sum(lens[:i]):sum(lens[:i]) + L].permute(1, 2, 0)
        qo += qo_pad[i]; kv += kv_pad[i]
        qo_indptr.append(qo); kv_indptr.append(kv)
    to_i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    sm_scale = (1.0 / D ** 0.5) * float(qs) * float(ks)
    o_scale = float(vs)

    O, LSE, L = ext().s3_ragged_fp8_attn(
        Qp, Kp, Vp, to_i32(qo_indptr), to_i32(kv_indptr), to_i32(lens), to_i32(lens),
        Hq, Hkv, sm_scale, o_scale, causal, 0)

    # --- fp64 oracle per request: dequant + online softmax @64-key blocks + fixed P scale ---
    Qdq = dequant_per_tensor(qd, qs)   # [total, Hq, D]
    Kdq = dequant_per_tensor(kd, ks)
    Vdq = dequant_per_tensor(vd, vs)
    off = 0
    worst = 0.0
    for i, L in enumerate(lens):
        Kdq_r = Kdq[off:off + L].repeat_interleave(group, dim=1)   # GQA: expand kv heads -> q heads
        Vdq_r = Vdq[off:off + L].repeat_interleave(group, dim=1)
        s_all = torch.einsum("qhd,khd->hqk", Qdq[off:off + L], Kdq_r)  # raw scores
        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)
        s_all = s_all.masked_fill(mask, float("-inf"))
        o_req = torch.zeros(Hq, L, D, dtype=torch.float64, device=dev)
        row_sum = torch.zeros(Hq, L, dtype=torch.float64, device=dev)
        m_run = torch.full((Hq, L), float("-inf"), dtype=torch.float64, device=dev)
        for nb in range((L + 63) // 64):
            sblk = s_all[:, :, nb * 64:(nb + 1) * 64]
            m_prev = m_run
            m_cur = torch.maximum(m_prev, sblk.max(dim=2).values)
            ss = torch.where(m_prev == float("-inf"), torch.zeros_like(m_prev),
                             torch.exp((m_prev - m_cur) * (1.0 / D ** 0.5)))
            p = torch.exp((sblk - m_cur[:, :, None]) * (1.0 / D ** 0.5))
            p = torch.where(m_cur[:, :, None] == float("-inf"), torch.zeros_like(p),
                            torch.nan_to_num(p, nan=0.0))
            row_sum = row_sum * ss + p.sum(dim=2)
            Pdq = (p.float() * 256.0).to(torch.float8_e4m3fn).double() / 256.0   # fixed-256 P requant
            o_req = o_req * ss[:, :, None] + torch.einsum("hqk,khd->hqd", Pdq, Vdq_r[nb * 64:(nb + 1) * 64])
            m_run = m_cur
        o_req = o_req / row_sum[:, :, None]
        # NOTE: no extra *v_scale here -- the kernel accumulates PV on RAW e4m3 bytes and
        # applies o_scale=v_scale ONCE in the epilogue; Vdq already carries v_scale.
        got = O[qo_indptr[i]:qo_indptr[i] + L].permute(1, 0, 2).double()   # [Hq, L, D]
        d = (got - o_req).abs().max().item()
        # error distribution is ~1e-4; the max over 1.3M elements tails at ~2e-3 (fp32 vs
        # fp64 rounding + P e4m3 granularity). Gate at 5e-3 (~10 sigma): catches real
        # regressions, ignores the rounding tail (cf. the 2e-3 gate for 65K-element tests).
        assert d < 5e-3, f"req {i} MISMATCH {d}"
        print(f"  req {i:2d} L={L:5d}: max|abs|={d:.3g}")
        worst = max(worst, d)
        off += L
    print(f"ragged+GQA(8,2)+causal, per-tensor fp8: O max|abs| vs fp64 oracle = {worst:.3g}")
    print("TEST_RAGGED_EXT PASS")


if __name__ == "__main__":
    main()
