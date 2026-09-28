"""dvdk2 (fused dv+dk, mode 3) vs split (mode 1) validation + bench via torch ext.

Run:  /root/miniconda3/envs/ai-toolkit/bin/python test_mxfp8_bwd_dvdk2.py [--bench]
"""
import math
import os
import sys

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))

bwd = load(
    name="mxfp8_bwd_ext_dvdk2test",
    sources=[os.path.join(ROOT, "csrc", "mxfp8_bwd_ext.cpp"),
             os.path.join(ROOT, "csrc", "mxfp8_bwd_kernel.cu")],
    extra_include_paths=["/root/fa-blackwell/tmp/cutlass/include",
                         "/root/fa-blackwell/include",
                         os.path.join(ROOT)],
    extra_cuda_cflags=["-O2", "-std=c++17", "--expt-relaxed-constexpr",
                       "--expt-extended-lambda",
                       "-gencode", "arch=compute_120a,code=sm_120a"],
    extra_cflags=["-O2"],
    verbose=False,
)


def deq_nat(qd, rsf):
    # qd uint8 [H,S,D] e4m3, rsf uint8 [H,S,D//32] -> fp64
    H, S, D = qd.shape
    q = qd.view(torch.float8_e4m3fn).to(torch.float64).reshape(H, S, D // 32, 32)
    sc = torch.exp2(rsf.to(torch.float64) - 127).reshape(H, S, D // 32, 1)
    return (q * sc).reshape(H, S, D)


def deq_trn(qt, rsf):
    # qt uint8 [H,D,S] e4m3, rsf uint8 [H,D,S//32] -> fp64 [H,S,D]
    H, D, S = qt.shape
    q = qt.view(torch.float8_e4m3fn).to(torch.float64).reshape(H, D, S // 32, 32)
    sc = torch.exp2(rsf.to(torch.float64) - 127).reshape(H, D, S // 32, 1)
    return (q * sc).reshape(H, D, S).transpose(1, 2)


def run(H, S, bench=False, iters=20):
    torch.manual_seed(0)
    dev = "cuda"
    D = 128
    sm = 1.0 / math.sqrt(D)
    q = torch.randn(H, S, D, device=dev, dtype=torch.bfloat16) * 0.6
    k = torch.randn(H, S, D, device=dev, dtype=torch.bfloat16) * 0.6
    v = torch.randn(H, S, D, device=dev, dtype=torch.bfloat16) * 0.6
    do = torch.randn(H, S, D, device=dev, dtype=torch.bfloat16) * 0.6

    qd, rq, sfq = bwd.quant_nat(q)
    kd, rk, sfk = bwd.quant_nat(k)
    vd, rv, sfv = bwd.quant_nat(v)
    dd, rd, sfd = bwd.quant_nat(do)
    qt, _, sfqt = bwd.quant_trn(q)
    kt, _, sfkt = bwd.quant_trn(k)
    dt, _, sfdt = bwd.quant_trn(do)

    # fp64 reference from dequantized (what the kernels see); big shapes go
    # chunked-fp32 for lse/delta only and skip the full reference (OOM guard)
    Q = deq_nat(qd, rq); K = deq_nat(kd, rk); V = deq_nat(vd, rv); dO = deq_nat(dd, rd)
    small = H * S * S * 8 < (2 << 30)
    if small:
        Sm = (Q @ K.transpose(1, 2)) * sm
        lse = torch.logsumexp(Sm, dim=-1)                      # [H,S] fp64
        P = torch.exp(Sm - lse.unsqueeze(-1))
        O = P @ V
        delta = (dO * O).sum(-1)                               # [H,S]
        dP = dO @ V.transpose(1, 2)
        dS = P * (dP - delta.unsqueeze(-1))
        dK_ref = dS.transpose(1, 2) @ Q * sm
        dV_ref = P.transpose(1, 2) @ dO
        dQ_ref = dS @ K * sm
    else:
        lse = torch.zeros(H, S, dtype=torch.float32, device=dev)
        delta = torch.zeros(H, S, dtype=torch.float32, device=dev)
        Qf, Kf, Vf, dOf = Q.float(), K.float(), V.float(), dO.float()
        for c0 in range(0, S, 2048):
            c1 = min(c0 + 2048, S)
            Sc = (Qf[:, c0:c1] @ Kf.transpose(1, 2)) * sm
            lc = torch.logsumexp(Sc, dim=-1)
            Pc = torch.exp(Sc - lc.unsqueeze(-1))
            delta[:, c0:c1] = (dOf[:, c0:c1] * (Pc @ Vf)).sum(-1)
            lse[:, c0:c1] = lc
        del Qf, Kf, Vf, dOf

    lse32 = lse.float().contiguous()
    dlt32 = delta.float().contiguous()

    def go(mode):
        return bwd.mxfp8_bwd(qd, kd, vd, dd, qt, kt, dt,
                             sfq, sfk, sfv, sfd, sfqt, sfkt, sfdt,
                             lse32, dlt32, sm, mode)

    dq1, dk1, dv1 = go(1)
    dq3, dk3, dv3 = go(3)
    torch.cuda.synchronize()

    def rel(g, r):
        g = (g * sm if False else g).double()              # raw kernel out (no sm fold)
        num = (g - r).pow(2).sum().item(); den = r.pow(2).sum().item()
        return math.sqrt(num / den)

    # raw kernel outputs; references need the un-sm-scaled dq/dk
    if small:
        print(f"[H{H} S{S}] rel-L2 vs fp64 ref:")
        print(f"  mode1 split: dQ {rel(dq1, dQ_ref / sm):.4f}  dK {rel(dk1, dK_ref / sm):.4f}  dV {rel(dv1, dV_ref):.4f}")
        print(f"  mode3 fused: dQ {rel(dq3, dQ_ref / sm):.4f}  dK {rel(dk3, dK_ref / sm):.4f}  dV {rel(dv3, dV_ref):.4f}")
        for nm, a, b in [("dK", dk1, dk3), ("dV", dv1, dv3)]:
            d = (a.double() - b.double()).abs().max().item()
            r = (a.double() - b.double()).pow(2).sum().sqrt() / b.double().pow(2).sum().sqrt()
            print(f"  mode3-vs-mode1 {nm}: maxabs {d:.3e}  rel {r.item():.4f}")
    else:
        for nm, a, b in [("dQ", dq1, dq3), ("dK", dk1, dk3), ("dV", dv1, dv3)]:
            same = torch.equal(a, b)
            print(f"  mode3-vs-mode1 {nm}: {'BITWISE MATCH' if same else 'DIFF ' + str((a - b).abs().max().item())}")

    if bench:
        def bench_mode(mode):
            for _ in range(3): go(mode)
            torch.cuda.synchronize()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record()
            for _ in range(iters): go(mode)
            b.record(); torch.cuda.synchronize()
            return a.elapsed_time(b) / iters
        t1 = bench_mode(1); t3 = bench_mode(3)
        print(f"[H{H} S{S} bench {iters}it] mode1 split: {t1:.3f} ms   mode3 fused: {t3:.3f} ms   ({100*(t1-t3)/t1:+.1f}%)")


if __name__ == "__main__":
    do_bench = "--bench" in sys.argv
    run(4, 4096, bench=do_bench)
    if do_bench:
        run(32, 16896, bench=True, iters=10)
