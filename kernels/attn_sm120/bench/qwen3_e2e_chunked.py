import json, os, sys, time
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LONG = ("The history of attention mechanisms in deep learning began with sequence-to-sequence "
        "models for machine translation, where additive attention let decoders focus on relevant "
        "encoder states. " * 60 + "\nSummarize the above in two sentences.")


def main():
    tag = sys.argv[1]
    use_patch = len(sys.argv) > 2 and sys.argv[2] == "patch"
    if use_patch:
        sys.path.insert(0, "/root/fa-blackwell")
        import bench.vllm_s3_patch as p
        p.install()
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-8B-AWQ", enforce_eager=True,
              gpu_memory_utilization=0.92, max_model_len=4096,
              max_num_batched_tokens=1024, kv_cache_memory_bytes=1 << 30,
              kv_cache_dtype="fp8", disable_log_stats=True)
    sp = SamplingParams(temperature=0, max_tokens=32)
    out = llm.generate([LONG], sp)
    r = {"prompt_tokens": len(out[0].prompt_token_ids),
         "token_ids": list(out[0].outputs[0].token_ids),
         "text": out[0].outputs[0].text}
    print(f"prompt_tokens={r['prompt_tokens']}", flush=True)
    print("OUTPUT:", r["text"], flush=True)
    json.dump(r, open(f"/tmp/opencode/e2e_chunked_{tag}.json", "w"), ensure_ascii=False)
    print(f"CHUNKED_{tag.upper()}_OK", flush=True)


if __name__ == "__main__":
    main()
