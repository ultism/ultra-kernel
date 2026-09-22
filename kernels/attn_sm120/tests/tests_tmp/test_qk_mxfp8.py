"""Numerical validation of the SM120a MXFP8 block-scaled QK kernel.

Baseline/oracle = torchao MX quantization (authoritative OCP MX), independent of
the kernel's CuTe layout assumptions. Random Q/K -> torchao to_mx -> kernel ->
compare against the fp64 dequant-matmul reference. Unlike the C++ smoke tests
(Q=K=1.0), the data is non-uniform, so a wrong A/B thread-value layout is caught.

Run:
  /root/miniconda3/envs/torchao-dev/bin/python -m pytest tests/test_qk_mxfp8.py -v -s
"""
import os
import pathlib

import pytest
import torch

# Block-scaled mma.sync needs the arch-specific target (sm_120a), not sm_120.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from torch.utils.cpp_extension import load
from torchao.prototype.mx_formats.mx_tensor import to_mx

ROOT = pathlib.Path(__file__).resolve().parents[1]

_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="qk_mxfp8_ext",
            sources=[str(ROOT / "tests" / "csrc" / "qk_mxfp8_ext.cu")],
            extra_include_paths=[
                str(ROOT / "tmp" / "cutlass" / "include"),
                str(ROOT / "include"),
            ],
            # torch 2.12 headers need C++20 (matches how torchao itself builds).
            extra_cflags=["-std=c++20", "-O2"],
            extra_cuda_cflags=[
                "-std=c++20", "-O2",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
            ],
            verbose=True,
        )
    return _ext


def dequant(data_e4m3, scale_e8m0, block=32):
    """Exact fp64 dequant: e4m3 value * 2^(e8m0_exp-127), per 32-block."""
    d = data_e4m3.double()
    s = scale_e8m0.double().repeat_interleave(block, dim=1)  # .double() decodes 2^(b-127)
    return d * s


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_qk_mxfp8_matches_torchao_baseline(head_dim):
    torch.manual_seed(1234 + head_dim)
    M, N = 16, 8
    # Varied magnitude/sign so block scales differ across rows/cols/blocks.
    Q = (torch.randn(M, head_dim, device="cuda") * torch.randn(M, 1, device="cuda").abs() * 4)
    K = (torch.randn(N, head_dim, device="cuda") * torch.randn(N, 1, device="cuda").abs() * 4)

    scale_A, data_A = to_mx(Q, torch.float8_e4m3fn, 32)
    scale_B, data_B = to_mx(K, torch.float8_e4m3fn, 32)

    A_bytes = data_A.view(torch.uint8).contiguous()
    B_bytes = data_B.view(torch.uint8).contiguous()
    SFA_bytes = scale_A.view(torch.uint8).contiguous()
    SFB_bytes = scale_B.view(torch.uint8).contiguous()
    assert SFA_bytes.shape == (M, head_dim // 32)

    S = ext().qk_mxfp8(A_bytes, B_bytes, SFA_bytes, SFB_bytes)

    Q_dq = dequant(data_A, scale_A)
    K_dq = dequant(data_B, scale_B)
    S_ref = (Q_dq @ K_dq.t()).float()  # fp64 accumulate -> the "true" value

    abs_err = (S - S_ref).abs()
    denom = S_ref.abs().clamp_min(1e-6)
    rel = (abs_err / denom).max().item()
    print(f"\nhead_dim={head_dim}: max|abs|={abs_err.max().item():.4g} "
          f"max|rel|={rel:.3g}  |S|range=[{S_ref.abs().min():.3g},{S_ref.abs().max():.3g}]")

    # Each term is exact (e4m3*e4m3 scaled by pow2); only fp32 accumulation order
    # differs from the fp64 reference.
    torch.testing.assert_close(S, S_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    for hd in [32, 64, 128]:
        test_qk_mxfp8_matches_torchao_baseline(hd)
        print(f"head_dim={hd}: PASS")
