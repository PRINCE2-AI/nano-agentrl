"""Environment check: torch/CUDA, GPU, optional libraries, one tiny generation, one W&B log.

Usage:
    python scripts/check_env.py                 # full check (downloads ~1GB model once)
    python scripts/check_env.py --skip_model    # only library / hardware checks

Every check is independent: a failure prints a clear message and the script moves on.
Exit code is 1 if any required check (torch, transformers, generation) failed, else 0.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check_torch() -> bool:
    """Print torch and CUDA versions plus GPU name and memory. Returns False if torch is missing."""
    try:
        import torch
    except ImportError as e:
        print(f'{FAIL} torch not importable ({e}). Install with: pip install -e ".[dev]"')
        return False

    print(f"{OK} torch {torch.__version__}")
    if not torch.cuda.is_available():
        print(f"{WARN} CUDA not available: running on CPU (fine for tests, too slow for training)")
        return True

    print(f"{OK} CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"{OK} GPU {i}: {props.name}, {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free, "
            f"bf16={'yes' if torch.cuda.is_bf16_supported() else 'no'}"
        )
    return True


def check_import(name: str, required: bool, hint: str) -> bool:
    """Try to import a library and print its version. Returns False only if required and missing."""
    try:
        module = importlib.import_module(name)
    except Exception as e:  # vllm can raise non-ImportError errors on unsupported platforms
        tag = FAIL if required else WARN
        print(f"{tag} {name} not importable ({type(e).__name__}: {e}). {hint}")
        return not required
    print(f"{OK} {name} {getattr(module, '__version__', '(unknown version)')}")
    return True


def check_generation(model_name: str, max_new_tokens: int) -> bool:
    """Load the model and greedily answer one question. Returns False on any error."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"{FAIL} generation skipped: {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {model_name} on {device} ({dtype}) ...")
    try:
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        model.eval()
        print(f"{OK} loaded in {time.time() - start:.1f}s")

        messages = [{"role": "user", "content": "What is 2+2?"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"{OK} generated in {time.time() - start:.1f}s: {answer.strip()!r}")
    except OSError as e:
        print(f"{FAIL} could not download/load {model_name}: {e}")
        print("       Check your internet connection or HF_HOME, or pass --model <local path>.")
        return False
    except Exception as e:
        print(f"{FAIL} generation failed ({type(e).__name__}): {e}")
        return False
    return True


def check_wandb(project: str) -> None:
    """Log one dummy metric to W&B, or skip if WANDB_API_KEY is not set. Never fails the run."""
    if not os.environ.get("WANDB_API_KEY"):
        print(f"{WARN} WANDB_API_KEY not set: skipping W&B logging check")
        return
    try:
        import wandb

        run = wandb.init(project=project, name="check_env", group="check_env", reinit=True)
        wandb.log({"check_env/dummy_metric": 1.0})
        url = run.get_url()
        run.finish()
        print(f"{OK} logged dummy metric to W&B: {url}")
    except Exception as e:
        print(f"{WARN} W&B logging failed ({type(e).__name__}): {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--skip_model", action="store_true", help="skip model download + generation"
    )
    parser.add_argument("--wandb_project", default="nano-agentrl")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("\n== Hardware ==")
    ok = check_torch()

    print("\n== Libraries ==")
    ok &= check_import("transformers", required=True, hint='pip install -e ".[dev]"')
    check_import(
        "vllm",
        required=False,
        hint='Optional; HF generate is used instead. pip install -e ".[vllm]"',
    )
    check_import("wandb", required=False, hint="pip install wandb")

    print("\n== Generation ==")
    if args.skip_model:
        print(f"{WARN} --skip_model given: skipping")
    else:
        ok &= check_generation(args.model, args.max_new_tokens)

    print("\n== W&B ==")
    check_wandb(args.wandb_project)

    print("\nAll required checks passed." if ok else "\nSome required checks FAILED (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
