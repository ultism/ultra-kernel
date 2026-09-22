"""Numerical validation of the SM120a MXFP8 block-scaled prefill core (step 1).

Oracle = torchao MX quantization (authoritative OCP MX) + an fp64 dequant
reference, independent of the kernel's CuTe layout. Grown stage by stage:
  1a  qk_tile        -- S = Q @ K^T over a full 128x128 tile vs dequant matmul

Run:
  /root/miniconda3/envs/torchao-dev/bin/python -m pytest tests/test_mxfp8_prefill.py -v -s
"""
import os
import pathlib

import pytest
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from torch.utils.cpp_extension import load
from torchao.prototype.mx_formats.mx_tensor import to_mx

ROOT = pathlib.Path(__file__).resolve().parents[1]

_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="mxfp8_prefill_ext",
            sources=[str(ROOT / "tests" / "csrc" / "mxfp8_prefill_ext.cu")],
            extra_include_paths=[
                str(ROOT / "tmp" / "cutlass" / "include"),
                str(ROOT / "include"),
            ],
            extra_cflags=["-std=c++20", "-O2"],
            extra_cuda_cflags=[
                "-std=c++20", "-O2",
                "-gencode", "arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
            ],
            verbose=True,
        )
    return _ext


def dequant(data_e4m3, scale_e8m0, block=32):
    """Exact fp64 dequant: e4m3 value * 2^(e8m0_exp-127), per 32-block along dim=1."""
    d = data_e4m3.double()
    s = scale_e8m0.double().repeat_interleave(block, dim=1)
    return d * s


# -------- Stage 1a: full-tile QK --------
@pytest.mark.parametrize("head_dim", [64, 128])
def test_qk_tile_matches_torchao(head_dim):
    torch.manual_seed(20 + head_dim)
    M, N = 128, 128
    Q = torch.randn(M, head_dim, device="cuda") * torch.randn(M, 1, device="cuda").abs() * 4
    K = torch.randn(N, head_dim, device="cuda") * torch.randn(N, 1, device="cuda").abs() * 4

    scale_A, data_A = to_mx(Q, torch.float8_e4m3fn, 32)
    scale_B, data_B = to_mx(K, torch.float8_e4m3fn, 32)

    S = ext().qk_tile(
        data_A.view(torch.uint8).contiguous(),
        data_B.view(torch.uint8).contiguous(),
        scale_A.view(torch.uint8).contiguous(),
        scale_B.view(torch.uint8).contiguous(),
    )

    Q_dq = dequant(data_A, scale_A)
    K_dq = dequant(data_B, scale_B)
    S_ref = (Q_dq @ K_dq.t()).float()

    abs_err = (S - S_ref).abs()
    rel = (abs_err / S_ref.abs().clamp_min(1e-6)).max().item()
    print(f"\nhead_dim={head_dim}: max|abs|={abs_err.max().item():.4g} "
          f"max|rel|={rel:.3g}  |S|range=[{S_ref.abs().min():.3g},{S_ref.abs().max():.3g}]")
    # Each product term is exact (e4m3*e4m3 scaled by pow2); only the fp32
    # accumulation order differs from the fp64 reference.
    torch.testing.assert_close(S, S_ref, rtol=2e-3, atol=2e-3)


_ext_attn = {}


def ext_attn(head_dim=128):
    """Separate extension exposing the S3 WS+TMA kernel (tests/s3_kernel.cuh).
    One module per head_dim (kHeadDim is a compile constant via -DS3_HEAD_DIM)."""
    if head_dim not in _ext_attn:
        _ext_attn[head_dim] = load(
            name=f"mxfp8_attn_ext_hd{head_dim}",
            sources=[str(ROOT / "tests" / "csrc" / "mxfp8_attn_ext.cu")],
            extra_include_paths=[
                str(ROOT / "tmp" / "cutlass" / "include"),
                str(ROOT / "include"),
            ],
            extra_cflags=["-std=c++20", "-O2", f"-DS3_HEAD_DIM={head_dim}"],
            extra_cuda_cflags=[
                "-std=c++20", "-O2", f"-DS3_HEAD_DIM={head_dim}",
                "-gencode", "arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
            ],
            verbose=True,
        )
    return _ext_attn[head_dim]


def _u8(t):
    return t.view(torch.uint8).contiguous()


def to_per_tensor_e4m3(X):
    """Mainstream per-tensor fp8 cache: ONE scalar per tensor. scale = amax/448 (e4m3 max), so the
    largest element maps to exactly 448 and nothing saturates. Returns (e4m3 tensor, fp64 scale)."""
    scale = X.abs().max().clamp_min(1e-12) / 448.0
    return (X / scale).to(torch.float8_e4m3fn), scale.double()


# -------- Stage 1e: full attention O/LSE end-to-end (S3) --------
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_attn_matches_torchao(causal, head_dim):
    torch.manual_seed(7 + int(causal))
    Sq = Sk = 512                       # 4 m_blocks x 4 n_blocks
    sm_scale = 1.0 / (head_dim ** 0.5)

    # dynamic range per row/col so block scales actually vary (like the QK test).
    Q = torch.randn(Sq, head_dim, device="cuda") * torch.randn(Sq, 1, device="cuda").abs() * 2
    K = torch.randn(Sk, head_dim, device="cuda") * torch.randn(Sk, 1, device="cuda").abs() * 2
    V = torch.randn(Sk, head_dim, device="cuda") * torch.randn(Sk, 1, device="cuda").abs() * 2

    sQ, dQ = to_mx(Q, torch.float8_e4m3fn, 32)                 # scale[Sq,d/32], data[Sq,d]
    sK, dK = to_mx(K, torch.float8_e4m3fn, 32)
    sV, dV = to_mx(V.t().contiguous(), torch.float8_e4m3fn, 32)  # V^T [d,Sk], block along keys

    O, LSE = ext_attn(head_dim).mxfp8_attn(
        _u8(dQ), _u8(dK), _u8(dV), _u8(sQ), _u8(sK), _u8(sV), sm_scale, causal)[:2]

    # ---- torchao reference: replay the kernel's ONLINE algorithm with torchao to_mx ----
    # The kernel requants each n_block's P relative to the RUNNING max M_nb (not the global
    # max), then telescopes onto accO. A full-softmax reference (global-max requant) disagrees
    # at the ~1% level because the per-block e4m3 binade differs; replaying online matches it.
    Qdq = dequant(dQ, sQ)                       # [Sq,d] fp64
    Kdq = dequant(dK, sK)                       # [Sk,d]
    Vdq = dequant(dV, sV)                       # [d,Sk]
    S = (Qdq @ Kdq.t())                         # [Sq,Sk]  (raw, no sm_scale)
    if causal:
        mask = torch.triu(torch.ones(Sq, Sk, device="cuda", dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    NINF = float("-inf")
    accO = torch.zeros(Sq, head_dim, dtype=torch.float64, device="cuda")      # fixed-scale (matches kernel)
    accO_dyn = torch.zeros_like(accO)                                         # dynamic per-block (measure cost)
    accO_true = torch.zeros_like(accO)                                        # no P requant (absolute accuracy)
    row_sum = torch.zeros(Sq, dtype=torch.float64, device="cuda")
    m_run = torch.full((Sq,), NINF, dtype=torch.float64, device="cuda")
    # real-64 (kBlockN=64): the online replay MUST step at the kernel's 64-key block
    # granularity -- a 128-key replay uses a different running-max/binade timing and
    # disagrees ~0.3-0.4% (P-quant granularity, NOT a kernel bug). See docs/gotcha.md.
    for nb in range(Sk // 64):
        sblk = S[:, nb * 64:(nb + 1) * 64]
        m_prev = m_run
        m_cur = torch.maximum(m_prev, sblk.max(dim=1).values)
        ss = torch.where(m_prev == NINF, torch.zeros_like(m_prev), torch.exp((m_prev - m_cur) * sm_scale))
        p = torch.exp((sblk - m_cur[:, None]) * sm_scale)
        p = torch.where(m_cur[:, None] == NINF, torch.zeros_like(p), torch.nan_to_num(p, nan=0.0))
        row_sum = row_sum * ss + p.sum(dim=1)
        Vblk = Vdq[:, nb * 64:(nb + 1) * 64].t()
        # kernel: FIXED scalar scale 256.0 (se=-8). P<=1.0 post-max-subtraction so it never
        # saturates (1.0*256=256<=448); e4m3 RNE rounding, dequant /256.
        Pdq = (p.float() * 256.0).to(torch.float8_e4m3fn).double() / 256.0
        accO = accO * ss[:, None] + Pdq @ Vblk
        # dynamic per-32-block to_mx -- only to quantify the precision we trade away
        sPd, dPd = to_mx(p.float(), torch.float8_e4m3fn, 32)
        accO_dyn = accO_dyn * ss[:, None] + dequant(dPd, sPd) @ Vblk
        accO_true = accO_true * ss[:, None] + p @ Vblk
        m_run = m_cur
    O_ref = accO / row_sum[:, None]
    O_dyn = accO_dyn / row_sum[:, None]
    O_true = accO_true / row_sum[:, None]
    LSE_ref = (m_run * sm_scale + torch.log(row_sum)).float()
    # The optimization's true cost: fixed-256 vs dynamic-per-block, each vs un-requantized P.
    # Report max-abs error normalized by the O magnitude SCALE (not per-element, which explodes
    # on the near-zero O entries that don't matter downstream).
    scale = O_true.abs().max()
    def err(A, B):
        d = (A - B).abs().max()
        return d.item(), (d / scale).item()
    fa, fr = err(O_ref, O_true); da, dr = err(O_dyn, O_true); xa, xr = err(O_ref, O_dyn)
    print(f"\n[P-quant cost | max|O|={scale:.3g}] "
          f"fixed-vs-trueP abs={fa:.3g}({fr:.2%}) | dynamic-vs-trueP abs={da:.3g}({dr:.2%}) "
          f"| fixed-vs-dynamic abs={xa:.3g}({xr:.2%})")

    o_abs = (O.double() - O_ref).abs()
    o_rel = (o_abs / O_ref.abs().clamp_min(1e-3)).max().item()
    lse_abs = (LSE.double() - LSE_ref.double()).abs().max().item()
    print(f"\ncausal={causal}: O max|abs|={o_abs.max().item():.3g} max|rel|={o_rel:.3g} "
          f"| LSE max|abs|={lse_abs:.3g} | |O_ref|range=[{O_ref.abs().min():.3g},{O_ref.abs().max():.3g}]")
    # P, Q, K, V are e4m3-requantized identically on both sides and the online algo is
    # replayed step-for-step, so only fp32 (kernel) vs fp64 (ref) accumulation order differs
    # -> ~1e-4 rel. Tolerance left at 2e-3 to catch real regressions without flakiness.
    torch.testing.assert_close(O.double(), O_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(LSE.double(), LSE_ref.double(), rtol=2e-3, atol=2e-3)


# -------- Stage 1e-partial: partial last KV block + V kFillZero, independent torchao oracle --------
# kv_len is NOT a 128-multiple, so the last n_block is partial. The kernel's TMA still loads a
# full 128-key tile; the padded tail [kv_len, Sk_pad) is poisoned with NaN bytes (0xFF) on BOTH
# V data and the fully-padded V-SF blocks -- a real recycled KV-cache tail. The QK mask zeroes P
# there, but PV reads V for ALL keys, so the kernel MUST kFillZero the pad or 0*NaN=NaN poisons O.
# This checks the partial-mask + kFillZero against the AUTHORITATIVE torchao oracle (the
# s6a_ragged.cu differential test only checks it against the self-consistent dense kernel).
# Non-causal only: causal with qo_len != kv_len needs the offset_q diagonal (slice-3, pending).
# Rebuild the kernel with -DS3_V_KFILLZERO=0 (edit the cflags) to see this FAIL with NaN.
def test_attn_partial_kvlen_matches_torchao(head_dim=128):
    torch.manual_seed(11)
    Sq, Sk_pad, kv_len = 256, 512, 460     # last block keys 384..511: valid 384..459, masked 460..511
    sm_scale = 1.0 / (head_dim ** 0.5)
    first_full_sf = -(-kv_len // 32)       # ceil(460/32)=15: block 15 fully masked; block 14 straddles

    Q = torch.randn(Sq, head_dim, device="cuda") * torch.randn(Sq, 1, device="cuda").abs() * 2
    K = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2
    V = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2

    sQ, dQ = to_mx(Q, torch.float8_e4m3fn, 32)
    sK, dK = to_mx(K, torch.float8_e4m3fn, 32)
    sV, dV = to_mx(V.t().contiguous(), torch.float8_e4m3fn, 32)   # [d, Sk_pad], block along keys

    # Poison the KERNEL's V padding gap with NaN (0xFF): all pad V DATA columns, and V-SF only
    # for the FULLY-padded 32-blocks (the straddling block keeps its real SF, shared w/ valid keys).
    dV_k, sV_k = _u8(dV).clone(), _u8(sV).clone()
    dV_k[:, kv_len:Sk_pad] = 0xFF
    sV_k[:, first_full_sf:] = 0xFF
    O, LSE = ext_attn(head_dim).mxfp8_attn(
        _u8(dQ), _u8(dK), dV_k, _u8(sQ), _u8(sK), sV_k, sm_scale, False, kv_len)[:2]
    assert torch.isfinite(O).all(), "kernel O has NaN/Inf -- V kFillZero did not sanitize the 0xFF pad"

    # reference: CLEAN (unpoisoned) dequant, mask keys >= kv_len, replay the kernel's online algo.
    Qdq, Kdq, Vdq = dequant(dQ, sQ), dequant(dK, sK), dequant(dV, sV)
    S = (Qdq @ Kdq.t())
    S[:, kv_len:] = float("-inf")          # partial-block + fully-padded key mask
    NINF = float("-inf")
    accO = torch.zeros(Sq, head_dim, dtype=torch.float64, device="cuda")
    row_sum = torch.zeros(Sq, dtype=torch.float64, device="cuda")
    m_run = torch.full((Sq,), NINF, dtype=torch.float64, device="cuda")
    for nb in range(Sk_pad // 64):   # 64-key replay: match kernel kBlockN (see stage-1e note)
        sblk = S[:, nb * 64:(nb + 1) * 64]
        m_cur = torch.maximum(m_run, sblk.max(dim=1).values)
        ss = torch.where(m_run == NINF, torch.zeros_like(m_run), torch.exp((m_run - m_cur) * sm_scale))
        p = torch.exp((sblk - m_cur[:, None]) * sm_scale)
        p = torch.where(m_cur[:, None] == NINF, torch.zeros_like(p), torch.nan_to_num(p, nan=0.0))
        row_sum = row_sum * ss + p.sum(dim=1)
        Vblk = Vdq[:, nb * 64:(nb + 1) * 64].t()
        Pdq = (p.float() * 256.0).to(torch.float8_e4m3fn).double() / 256.0   # kernel's fixed scale 256
        accO = accO * ss[:, None] + Pdq @ Vblk
        m_run = m_cur
    O_ref = accO / row_sum[:, None]
    LSE_ref = (m_run * sm_scale + torch.log(row_sum)).float()
    o_abs = (O.double() - O_ref).abs().max().item()
    lse_abs = (LSE.double() - LSE_ref.double()).abs().max().item()
    print(f"\npartial kv_len={kv_len} (Sk_pad={Sk_pad}, 0xFF pad): O max|abs|={o_abs:.3g} "
          f"LSE max|abs|={lse_abs:.3g} | O finite={bool(torch.isfinite(O).all())}")
    torch.testing.assert_close(O.double(), O_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(LSE.double(), LSE_ref.double(), rtol=2e-3, atol=2e-3)


# -------- Stage 1e-offset: slice-3 append/chunked causal (offset_q = kv_len - qo_len > 0) --------
# qo_len < kv_len: the qo_len queries are APPENDED at the end of kv_len keys (decode / chunked
# prefill), so request-local query m attends keys [0, m + offset_q], offset_q = kv_len - qo_len.
# This validates the kernel's SHIFTED causal diagonal (s3_kernel.cuh: n_local > m_local + offset_q)
# AND the matching n_block_max bound. The second config also has kv_len NOT a 128-multiple, so the
# last n_block is partial AND its padded keys (>= kv_len) are BOTH causally masked and 0xFF-poisoned
# -> exercises slice-3 together with the V kFillZero. Independent torchao oracle.
# (offset_q = 0 is already covered by test_attn_matches_torchao's causal case.)
@pytest.mark.parametrize("qo_len,kv_len,Sk_pad", [(128, 384, 384), (256, 460, 512)])
def test_attn_causal_offset_matches_torchao(qo_len, kv_len, Sk_pad, head_dim=128):
    torch.manual_seed(13 + qo_len)
    offset_q = kv_len - qo_len
    sm_scale = 1.0 / (head_dim ** 0.5)
    partial = (kv_len != Sk_pad)
    first_full_sf = -(-kv_len // 32)

    Q = torch.randn(qo_len, head_dim, device="cuda") * torch.randn(qo_len, 1, device="cuda").abs() * 2
    K = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2
    V = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2

    sQ, dQ = to_mx(Q, torch.float8_e4m3fn, 32)
    sK, dK = to_mx(K, torch.float8_e4m3fn, 32)
    sV, dV = to_mx(V.t().contiguous(), torch.float8_e4m3fn, 32)

    # poison the padded V tail (partial config only) -- a recycled KV-cache tail. Those keys are
    # causally masked (P==0) yet 0*NaN=NaN without kFillZero, so this proves the guard fires.
    dV_k, sV_k = _u8(dV).clone(), _u8(sV).clone()
    if partial:
        dV_k[:, kv_len:Sk_pad] = 0xFF
        sV_k[:, first_full_sf:] = 0xFF
    O, LSE = ext_attn(head_dim).mxfp8_attn(
        _u8(dQ), _u8(dK), dV_k, _u8(sQ), _u8(sK), sV_k, sm_scale, True, kv_len)[:2]
    assert torch.isfinite(O).all(), "kernel O has NaN/Inf -- slice-3 offset mask or kFillZero broke"

    # reference: clean dequant; shifted causal mask via triu(diagonal=offset_q+1), which masks
    # n - m >= offset_q + 1  <=>  n > m + offset_q, and at the last query (m=qo_len-1) cuts exactly
    # at n = kv_len -> it SUBSUMES the n >= kv_len pad mask. Replay the kernel's online algo.
    Qdq, Kdq, Vdq = dequant(dQ, sQ), dequant(dK, sK), dequant(dV, sV)
    S = (Qdq @ Kdq.t())                              # [qo_len, Sk_pad]
    mask = torch.triu(torch.ones(qo_len, Sk_pad, device="cuda", dtype=torch.bool), diagonal=offset_q + 1)
    S = S.masked_fill(mask, float("-inf"))
    NINF = float("-inf")
    accO = torch.zeros(qo_len, head_dim, dtype=torch.float64, device="cuda")
    row_sum = torch.zeros(qo_len, dtype=torch.float64, device="cuda")
    m_run = torch.full((qo_len,), NINF, dtype=torch.float64, device="cuda")
    for nb in range(Sk_pad // 64):   # 64-key replay: match kernel kBlockN (see stage-1e note)
        sblk = S[:, nb * 64:(nb + 1) * 64]
        m_cur = torch.maximum(m_run, sblk.max(dim=1).values)
        ss = torch.where(m_run == NINF, torch.zeros_like(m_run), torch.exp((m_run - m_cur) * sm_scale))
        p = torch.exp((sblk - m_cur[:, None]) * sm_scale)
        p = torch.where(m_cur[:, None] == NINF, torch.zeros_like(p), torch.nan_to_num(p, nan=0.0))
        row_sum = row_sum * ss + p.sum(dim=1)
        Vblk = Vdq[:, nb * 64:(nb + 1) * 64].t()
        Pdq = (p.float() * 256.0).to(torch.float8_e4m3fn).double() / 256.0   # kernel's fixed scale 256
        accO = accO * ss[:, None] + Pdq @ Vblk
        m_run = m_cur
    O_ref = accO / row_sum[:, None]
    LSE_ref = (m_run * sm_scale + torch.log(row_sum)).float()
    o_abs = (O.double() - O_ref).abs().max().item()
    lse_abs = (LSE.double() - LSE_ref.double()).abs().max().item()
    print(f"\ncausal-offset qo={qo_len} kv={kv_len} (offset_q={offset_q}, Sk_pad={Sk_pad}, "
          f"partial={partial}): O max|abs|={o_abs:.3g} LSE max|abs|={lse_abs:.3g} | "
          f"O finite={bool(torch.isfinite(O).all())}")
    torch.testing.assert_close(O.double(), O_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(LSE.double(), LSE_ref.double(), rtol=2e-3, atol=2e-3)


# -------- Stage S6b: per-tensor fp8 cache via the kUniformFp8 SFSource --------
# The mainstream KV-cache format (FA3 / FlashInfer-fp8 / SageAttention) is per-tensor fp8:
# torch.float8_e4m3fn data + ONE scalar q/k/v_scale. e4m3 bytes are bit-identical to ours, so a
# block-scaled mxfp8 kernel is a strict SUPERSET: run the block-scaled MMA with a UNIFORM SF (byte
# 127 = 2^0) and fold the scalars OUT -- q_scale*k_scale into sm_scale, v_scale into o_scale. The
# kernel (uniform_sf=True) SKIPS every SF TMA load, so the Qs/Ks/Vs we pass are POISONED with NaN
# (0xFF): a correct result proves the per-tensor path reads NO scale factors from the cache. The
# third config also has a partial last KV block with a 0xFF-poisoned V-data tail (a recycled cache
# tail) to exercise uniform-SF together with the V kFillZero. Independent torchao/torch oracle.
@pytest.mark.parametrize("causal,Sq,kv_len,Sk_pad", [
    (False, 512, 512, 512),
    (True,  512, 512, 512),
    (False, 256, 460, 512),
])
def test_attn_per_tensor_fp8_matches_torchao(causal, Sq, kv_len, Sk_pad, head_dim=128):
    torch.manual_seed(31 + Sq + int(causal))
    sm_scale = 1.0 / (head_dim ** 0.5)
    partial = (kv_len != Sk_pad)

    Q = torch.randn(Sq, head_dim, device="cuda") * torch.randn(Sq, 1, device="cuda").abs() * 2
    K = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2
    V = torch.randn(Sk_pad, head_dim, device="cuda") * torch.randn(Sk_pad, 1, device="cuda").abs() * 2

    qQ, q_scale = to_per_tensor_e4m3(Q)                       # [Sq, hd]
    qK, k_scale = to_per_tensor_e4m3(K)                       # [Sk_pad, hd]
    qV, v_scale = to_per_tensor_e4m3(V.t().contiguous())      # [hd, Sk_pad]

    dQ, dK, dV = _u8(qQ), _u8(qK), _u8(qV)
    if partial:
        dV = dV.clone(); dV[:, kv_len:Sk_pad] = 0xFF         # recycled cache tail (pad-masked keys)

    # POISON every SF with NaN -- uniform_sf=True makes the kernel ignore them entirely.
    pSF_q = torch.full((Sq, head_dim // 32), 0xFF, dtype=torch.uint8, device="cuda")
    pSF_k = torch.full((Sk_pad, head_dim // 32), 0xFF, dtype=torch.uint8, device="cuda")
    pSF_v = torch.full((head_dim, Sk_pad // 32), 0xFF, dtype=torch.uint8, device="cuda")

    sm_folded = sm_scale * q_scale.item() * k_scale.item()   # q_scale*k_scale -> sm_scale
    O, LSE = ext_attn(head_dim).mxfp8_attn(
        dQ, dK, dV, pSF_q, pSF_k, pSF_v, sm_folded, causal,
        kv_len if partial else -1, True, v_scale.item())[:2]  # uniform_sf=True, o_scale=v_scale
    assert torch.isfinite(O).all(), "kernel O has NaN/Inf -- uniform SF or V kFillZero broke"

    # reference: per-tensor dequant, online replay with the REAL sm_scale + the kernel's fixed-256 P.
    Qdq, Kdq, Vdq = qQ.double() * q_scale, qK.double() * k_scale, qV.double() * v_scale
    S = Qdq @ Kdq.t()                                        # [Sq, Sk_pad]
    if causal:
        S = S.masked_fill(torch.triu(torch.ones(Sq, Sk_pad, device="cuda", dtype=torch.bool), 1), float("-inf"))
    if partial:
        S[:, kv_len:] = float("-inf")
    NINF = float("-inf")
    accO = torch.zeros(Sq, head_dim, dtype=torch.float64, device="cuda")
    row_sum = torch.zeros(Sq, dtype=torch.float64, device="cuda")
    m_run = torch.full((Sq,), NINF, dtype=torch.float64, device="cuda")
    for nb in range(Sk_pad // 64):   # 64-key replay: match kernel kBlockN (see stage-1e note)
        sblk = S[:, nb * 64:(nb + 1) * 64]
        m_cur = torch.maximum(m_run, sblk.max(dim=1).values)
        ss = torch.where(m_run == NINF, torch.zeros_like(m_run), torch.exp((m_run - m_cur) * sm_scale))
        p = torch.exp((sblk - m_cur[:, None]) * sm_scale)
        p = torch.where(m_cur[:, None] == NINF, torch.zeros_like(p), torch.nan_to_num(p, nan=0.0))
        row_sum = row_sum * ss + p.sum(dim=1)
        Vblk = Vdq[:, nb * 64:(nb + 1) * 64].t()
        Pdq = (p.float() * 256.0).to(torch.float8_e4m3fn).double() / 256.0
        accO = accO * ss[:, None] + Pdq @ Vblk
        m_run = m_cur
    O_ref = accO / row_sum[:, None]
    LSE_ref = (m_run * sm_scale + torch.log(row_sum)).float()
    o_abs = (O.double() - O_ref).abs().max().item()
    lse_abs = (LSE.double() - LSE_ref.double()).abs().max().item()
    print(f"\nper-tensor fp8 causal={causal} kv_len={kv_len}/{Sk_pad} (q={q_scale.item():.3g} "
          f"k={k_scale.item():.3g} v={v_scale.item():.3g}): O max|abs|={o_abs:.3g} "
          f"LSE max|abs|={lse_abs:.3g} | O finite={bool(torch.isfinite(O).all())}")
    torch.testing.assert_close(O.double(), O_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(LSE.double(), LSE_ref.double(), rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    for hd in [64, 128]:
        test_qk_tile_matches_torchao(hd)
        print(f"head_dim={hd}: PASS")
    test_attn_partial_kvlen_matches_torchao(128)
    print("attn partial kv_len + V kFillZero: PASS")
    for cfg in [(128, 384, 384), (256, 460, 512)]:
        test_attn_causal_offset_matches_torchao(*cfg)
        print(f"attn causal-offset {cfg}: PASS")
    for cfg in [(False, 512, 512, 512), (True, 512, 512, 512), (False, 256, 460, 512)]:
        test_attn_per_tensor_fp8_matches_torchao(*cfg)
        print(f"attn per-tensor fp8 {cfg}: PASS")
    for hd in [64, 128]:
        for c in [False, True]:
            test_attn_matches_torchao(c, hd)
            print(f"attn head_dim={hd} causal={c}: PASS")
