r"""Entry point: python scripts/evaluate.py --ckpt <path or HF id> --split nq_dev_500

Examples (GPU box, retriever running on :8000):
    # baselines on Qwen2.5-3B-Instruct (repeat with --mode rag and --mode search)
    python scripts/evaluate.py --ckpt Qwen/Qwen2.5-3B-Instruct --mode direct \
        --split nq_dev_500 hotpotqa_dev_500
    # the released Search-R1 checkpoint, same harness
    python scripts/evaluate.py --ckpt PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
        --mode search --split nq_dev_500 hotpotqa_dev_500
    # our checkpoint
    python scripts/evaluate.py --ckpt runs/search_3b/checkpoints/step_001000 --mode search \
        --split nq_dev_500
    # tiny model, in-memory corpus, CPU
    python scripts/evaluate.py --dry_run --mode search --split tinyqa_test

Writes <out_dir>/<split>_<mode>.jsonl (one record per question x sample) and
<out_dir>/summary.json, and upserts the EM into docs/results.json / docs/results.md
(greedy single-sample runs only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_agentrl.data import load_split  # noqa: E402
from nano_agentrl.eval.evaluate import (  # noqa: E402
    MODES,
    evaluate_examples,
    save_records,
    summarize,
    update_results,
)
from nano_agentrl.generation import Generator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_model(ckpt: str, device: str, use_vllm: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if use_vllm:
        return Generator(None, tokenizer, backend="vllm", model_name=ckpt), tokenizer
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype).to(device).eval()
    return Generator(model, tokenizer, backend="hf"), tokenizer


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate a model on QA splits")
    parser.add_argument("--ckpt", default=None, help="checkpoint dir or HF model id")
    parser.add_argument("--split", nargs="+", default=["nq_dev_500"])
    parser.add_argument("--mode", choices=MODES, default="search")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--name", default=None, help="model label in the results table")
    parser.add_argument("--results_json", default=str(ROOT / "docs/results.json"))
    parser.add_argument("--results_md", default=str(ROOT / "docs/results.md"))
    parser.add_argument("--no_results_table", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="tiny model + in-memory corpus")
    args = parser.parse_args(argv)

    if args.dry_run:
        from nano_agentrl.retriever.client import TINY_CORPUS, InMemoryRetriever
        from nano_agentrl.tiny import make_char_tokenizer, make_tiny_model

        tokenizer = make_char_tokenizer()
        generator = Generator(make_tiny_model(len(tokenizer)).eval(), tokenizer)
        retriever = InMemoryRetriever(TINY_CORPUS)
        name = args.name or "tiny-random-gpt2"
        args.max_new_tokens = min(args.max_new_tokens, 16)
    else:
        if not args.ckpt:
            parser.error("--ckpt is required (or use --dry_run)")
        generator, tokenizer = load_model(args.ckpt, args.device, args.use_vllm)
        retriever = None
        if args.mode != "direct":
            from nano_agentrl.retriever.client import RetrieverClient

            retriever = RetrieverClient(args.retriever_url)
        name = args.name or args.ckpt.rstrip("/").split("/")[-1]

    out_dir = Path(args.out_dir or ROOT / "runs/eval" / name)
    summaries = {}
    for split in args.split:
        examples = load_split(split, args.max_examples)
        start = time.time()
        records = evaluate_examples(
            generator,
            tokenizer,
            examples,
            args.mode,
            retriever,
            n_samples=args.n_samples,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
        )
        summary = summarize(records) | {"seconds": time.time() - start}
        save_records(records, out_dir / f"{split}_{args.mode}.jsonl")
        summaries[split] = summary
        print(
            f"{name} | {args.mode} | {split}: EM {summary['em'] * 100:.1f} "
            f"(n={summary['num_questions']}), searches {summary['num_searches']:.2f}, "
            f"invalid format {summary['invalid_format_rate']:.2f}"
        )
        if args.n_samples == 1 and args.temperature == 0 and not args.no_results_table:
            update_results(
                args.results_json,
                args.results_md,
                {
                    "model": name,
                    "mode": args.mode,
                    "split": split,
                    "em": summary["em"],
                    "n": summary["num_questions"],
                    "date": date.today().isoformat(),
                },
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    main()
