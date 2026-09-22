# Qwen3 e2e: stock (fa2 prefill) vs s3-patched (ours) on vllm, fp8 KV cache, greedy.
# Usage:
#   /root/vllm-omni/.venv/bin/python bench/qwen3_e2e.py --stock --tag stock
#   /root/vllm-omni/.venv/bin/python bench/qwen3_e2e.py --patch --tag ours
import argparse, json, os, sys, time

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROMPTS = [
    "用一句话介绍注意力机制。",
    "请写一段 200 字关于旋转位置编码（RoPE）工作原理的说明。",
    "Read the following passage and summarize in one sentence: " + (
        "FlashAttention is an IO-aware exact attention algorithm that tiles the attention "
        "computation to avoid materializing the quadratic score matrix in HBM. " * 30),
    "Give me a detailed step-by-step guide, with at least eight numbered steps, on how to "
    "profile a CUDA attention kernel with NVIDIA Nsight Compute, including which metrics "
    "to inspect first and why.",
    "写一首关于显存墙的五言绝句，然后逐句解释含义。",
    "Translate to French: The tensor cores on consumer Blackwell GPUs only expose the "
    "synchronous mma.sync instruction, without any asynchronous matrix-multiply primitive.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    if args.patch:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
        import bench.vllm_s3_patch as p
        p.install()

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-8B-AWQ", enforce_eager=True,
              gpu_memory_utilization=0.92, max_model_len=4096,
              max_num_batched_tokens=1024, kv_cache_memory_bytes=1 << 30,
              kv_cache_dtype="fp8", disable_log_stats=True)
    print(f"MODEL_LOADED in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate(PROMPTS, sp)
    dt = time.time() - t0
    res = []
    for o in outs:
        res.append({
            "prompt": o.prompt[:60],
            "prompt_tokens": len(o.prompt_token_ids),
            "token_ids": list(o.outputs[0].token_ids),
            "text": o.outputs[0].text,
        })
        print(f"  [{len(o.prompt_token_ids):5d} tok prompt] -> {len(o.outputs[0].token_ids)} tok", flush=True)
    print(f"TOTAL_GEN_TIME {dt:.2f}s", flush=True)
    with open(f"/tmp/opencode/e2e_{args.tag}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"E2E_{args.tag.upper()}_OK", flush=True)


if __name__ == "__main__":
    main()
