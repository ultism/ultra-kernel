# S9 e2e: vLLM FlashInferImpl monkeypatch — prefill goes to the s3 ragged+GQA+per-tensor-fp8
# kernel (kUniformFp8), decode stays on the stock flashinfer fa2 path.
# Install BEFORE constructing the LLM:  import bench.vllm_s3_patch; bench.vllm_s3_patch.install()
import os, pathlib, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_ext = None


def ext():
    global _ext
    if _ext is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        from torch.utils.cpp_extension import load
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


def _run_s3_prefill(self, layer, q, kv_cache, md, out):
    """q: [num_prefill_tokens, Hq, D] (RoPE'd, bf16). kv_cache: [nb, 2, ps, Hkv, D] fp8 bytes.
    out: [num_prefill_tokens, Hq*D] to fill."""
    w = md.prefill.wrapper
    np_ = md.num_prefills
    qo_indptr = w._qo_indptr_buf[:np_ + 1].cpu()
    kv_indptr = w._paged_kv_indptr_buf[:np_ + 1].cpu()
    kv_indices = w._paged_kv_indices_buf[:int(kv_indptr[-1])].cpu()
    last_len = w._paged_kv_last_page_len_buf[:np_].cpu()
    nb, two, ps, Hkv, D = kv_cache.shape
    Hq = q.shape[1]
    kvc_u8 = kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
    qo_lens = [int(qo_indptr[i + 1] - qo_indptr[i]) for i in range(np_)]
    kv_lens = [(int(kv_indptr[i + 1] - kv_indptr[i]) - 1) * ps + int(last_len[i]) for i in range(np_)]
    pad = lambda L: (L + 127) // 128 * 128
    # per-tensor fp8 quant of the whole prefill chunk's Q (one q_scale per launch)
    q_scale = (q.abs().max().clamp_min(1e-12) / 448.0).item()
    q8 = (q / q_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    Sq_pad, Sk_pad = sum(pad(l) for l in qo_lens), sum(pad(l) for l in kv_lens)
    dev = q.device
    Qp = torch.zeros(Sq_pad, Hq, D, dtype=torch.uint8, device=dev)
    Kp = torch.zeros(Sk_pad, Hkv, D, dtype=torch.uint8, device=dev)
    Vp = torch.zeros(Hkv, D, Sk_pad, dtype=torch.uint8, device=dev)
    qo = kv = 0
    qo_ip, kv_ip = [0], [0]
    for i in range(np_):
        ql, kl = qo_lens[i], kv_lens[i]
        Qp[qo:qo + ql] = q8[int(qo_indptr[i]):int(qo_indptr[i]) + ql]
        pages = kv_indices[int(kv_indptr[i]):int(kv_indptr[i + 1])].to(dev)
        blocks = kvc_u8[pages]                              # [npages, 2, ps, Hkv, D]
        Kp[kv:kv + kl] = blocks[:, 0].reshape(-1, Hkv, D)[:kl]
        Vp[:, :, kv:kv + kl] = blocks[:, 1].reshape(-1, Hkv, D)[:kl].permute(1, 2, 0)
        qo += pad(ql); kv += pad(kl); qo_ip.append(qo); kv_ip.append(kv)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    k_scale = float(layer._k_scale_float); v_scale = float(layer._v_scale_float)
    sm = float(self.scale) * q_scale * k_scale              # (1/sqrt D) * q_scale * k_scale
    O, _, _ = ext().s3_ragged_fp8_attn(Qp, Kp, Vp, i32(qo_ip), i32(kv_ip),
                                       i32(qo_lens), i32(kv_lens), Hq, Hkv,
                                       sm, v_scale, True, 0)
    qo = 0
    out2d = out.reshape(out.shape[0], -1)          # vllm output may be [tokens, Hq, D] or [tokens, Hq*D]
    for i in range(np_):
        ql = qo_lens[i]
        out2d[int(qo_indptr[i]):int(qo_indptr[i]) + ql] = O[qo:qo + ql].reshape(ql, -1).to(out.dtype)
        qo += pad(ql)


def _s3_forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
    from vllm.v1.attention.backends.flashinfer import FIDecode, FIPrefill, FlashInferBackend
    if attn_metadata is None:
        return output.fill_(0)
    if (attn_metadata.use_cascade or attn_metadata.num_prefill_tokens == 0
            or output_scale is not None or getattr(self, "logits_soft_cap", None)
            or getattr(self, "window_left", -1) not in (-1, 0)):
        return _orig_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                             output, output_scale, output_block_scale)
    num_dec = attn_metadata.num_decode_tokens
    query = query[:attn_metadata.num_actual_tokens]
    output = output[:attn_metadata.num_actual_tokens]
    if num_dec > 0:
        assert isinstance(attn_metadata.decode, FIDecode)
        w = attn_metadata.decode.wrapper
        kvc = kv_cache.view(torch.float8_e4m3fn) if kv_cache.dtype == torch.uint8 else kv_cache
        kvc = kvc.permute(*FlashInferBackend.get_kv_cache_stride_order())
        w.run(query[:num_dec], kvc, k_scale=layer._k_scale_float,
              v_scale=layer._v_scale_float, out=output[:num_dec])
    if attn_metadata.num_prefill_tokens > 0:
        assert isinstance(attn_metadata.prefill, FIPrefill)
        _run_s3_prefill(self, layer, query[num_dec:], kv_cache, attn_metadata,
                        output[num_dec:])
    return output


_orig_forward = None


def install():
    global _orig_forward
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    if _orig_forward is None:
        _orig_forward = FlashInferImpl.forward
    FlashInferImpl.forward = _s3_forward
    ext()   # pre-build the extension
    print("[s3_patch] installed: prefill -> s3 ragged fp8 kernel, decode -> stock fa2")


def uninstall():
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    if _orig_forward is not None:
        FlashInferImpl.forward = _orig_forward
        print("[s3_patch] uninstalled")
