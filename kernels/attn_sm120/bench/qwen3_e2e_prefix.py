import json
import os
import sys

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B-AWQ"

PREFIX = (
    "Attention mechanisms in deep learning have evolved through several distinct phases. "
    "Early sequence-to-sequence models for neural machine translation introduced additive "
    "attention, allowing decoders to focus on relevant encoder states at each generation step. "
    "This was followed by the Transformer architecture, which replaced recurrence entirely "
    "with scaled dot-product self-attention, enabling massive parallelism during training. "
    "Subsequent research addressed the quadratic complexity of self-attention through sparse "
    "patterns, low-rank approximations, kernel methods, and memory-efficient exact algorithms. "
    "The FlashAttention family demonstrated that IO-aware tiling could deliver large wall-clock "
    "speedups without any approximation, and modern serving stacks extend these ideas with "
    "paged key-value caches, continuous batching, and quantized cache storage. "
) * 6

P1 = PREFIX + "\nQuestion: name one IO-aware exact attention algorithm.\nAnswer:"
P2 = PREFIX + "\nQuestion: what storage technique do serving stacks use for KV?\nAnswer:"


def main():
    tag = sys.argv[1]
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.92,
        kv_cache_memory_bytes=1 << 30,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    r1 = llm.generate([P1], sp)[0].outputs[0]
    r2 = llm.generate([P2], sp)[0].outputs[0]
    out = {
        "r1": {"ids": list(r1.token_ids), "text": r1.text},
        "r2": {"ids": list(r2.token_ids), "text": r2.text},
    }
    with open(f"/tmp/opencode/e2e_prefix_{tag}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("PREFIX_E2E_OK", tag)
    print("R2:", r2.text[:120])


if __name__ == "__main__":
    main()
