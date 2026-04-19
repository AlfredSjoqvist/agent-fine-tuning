"""Smoke test: verl + transformers import and Qwen2.5-3B tokenizer load.

This is deliberately CPU-only and does NOT exercise vLLM / flash-attn / training.
Goal: confirm `import verl` works and HF hub is reachable from Modal.

Once this passes we'll layer on a GPU image that adds vLLM for rollouts.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_verl_smoke.py
"""

import modal

app = modal.App("cs224r-verl-smoke")

verl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ca-certificates")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.45",
        "tokenizers",
        "hf_transfer",
        "verl",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=verl_image, timeout=600)
def test_verl() -> dict:
    import importlib.metadata as im

    import verl
    from transformers import AutoTokenizer

    versions = {
        "verl": getattr(verl, "__version__", im.version("verl")),
        "torch": im.version("torch"),
        "transformers": im.version("transformers"),
    }
    print("installed versions:")
    for k, v in versions.items():
        print(f"  {k}: {v}")

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\nloading tokenizer for {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    msg = [
        {"role": "system", "content": "You are an ALFWorld agent."},
        {"role": "user", "content": "you see a cabinet 1, a fridge 1. task: put mug in coffeemachine 1."},
    ]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
    print(f"tokenized prompt length: {len(ids)} tokens")
    decoded = tok.decode(ids)
    print("decoded (first 200 chars):")
    print(decoded[:200])

    return {
        "versions": versions,
        "tokenizer_vocab_size": tok.vocab_size,
        "prompt_tokens": len(ids),
    }


@app.local_entrypoint()
def main():
    result = test_verl.remote()
    print("\nverl smoke test result:", result)
