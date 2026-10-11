r"""Evaluate every saved checkpoint of a run for accuracy AND reliability; write one CSV.

For each checkpoint and split: n samples per question at temperature 1.0 (default n = 8),
then pass@k and pass^k for k = 1..n (eval/pass_k.py), mean searches, invalid-format rate,
finish reasons, and the failure-taxonomy label fractions (eval/failure_taxonomy.py).

    python scripts/reliability_sweep.py --run_dir runs/search_3b --split nq_dev_500 \
        --max_examples 200 --n_samples 8 --use_vllm --judge rule
    python scripts/reliability_sweep.py --run_dir runs/dry_run --dry_run    # tiny, CPU

Output: <run_dir>/reliability.csv (one row per checkpoint x split, sorted by step) and the
raw records under <run_dir>/reliability/<step>/<split>.jsonl. Re-running skips checkpoint x
split pairs whose records already exist, so the sweep can be resumed or extended.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_agentrl.data import load_split  # noqa: E402
from nano_agentrl.eval.evaluate import (  # noqa: E402
    evaluate_examples,
    load_records,
    save_records,
    summarize,
)
from nano_agentrl.eval.failure_taxonomy import (  # noqa: E402
    LABELS,
    label_failures,
    label_fractions,
    make_judge,
)
from nano_agentrl.eval.pass_k import pass_metrics  # noqa: E402
from nano_agentrl.generation import Generator  # noqa: E402

STEP_RE = re.compile(r"step_(\d+)$")


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    ckpts = []
    for p in sorted((run_dir / "checkpoints").glob("step_*")):
        m = STEP_RE.search(p.name)
        if m and (p / "config.json").exists():
            ckpts.append((int(m.group(1)), p))
    return ckpts


def build_generator(ckpt: Path, dry_run: bool, use_vllm: bool, device: str):
    if dry_run:
        from transformers import GPT2LMHeadModel

        from nano_agentrl.tiny import make_char_tokenizer

        tok = make_char_tokenizer()
        return Generator(GPT2LMHeadModel.from_pretrained(ckpt).eval(), tok), tok
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    if use_vllm:
        return Generator(None, tok, backend="vllm", model_name=str(ckpt)), tok
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype).to(device).eval()
    return Generator(model, tok), tok


def row_for(step: int, split: str, records: list[dict], labelled: list[dict]) -> dict:
    s = summarize(records)
    row = {
        "step": step,
        "split": split,
        "num_questions": s["num_questions"],
        "num_samples": s["num_samples"],
        "num_searches": s["num_searches"],
        "invalid_format_rate": s["invalid_format_rate"],
    }
    for key, value in pass_metrics(records).items():
        row[key.replace("@", "_at_").replace("^", "_hat_")] = value  # pass_at_1, pass_hat_1
    num_failures = sum(not r["correct"] for r in records)
    row["failure_rate"] = num_failures / max(len(records), 1)
    row["num_failures"] = num_failures
    for label, frac in label_fractions(labelled).items():
        row[f"fail_{label}"] = frac if labelled else 0.0
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["split"], r["step"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Reliability sweep over checkpoints")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split", nargs="+", default=["nq_dev_500"])
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--judge", choices=["none", "rule", "claude"], default="rule")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true", help="tiny ckpts + in-memory corpus")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if args.dry_run:
        from nano_agentrl.retriever.client import TINY_CORPUS, InMemoryRetriever

        retriever = InMemoryRetriever(TINY_CORPUS)
        args.split, args.max_new_tokens = ["tinyqa_test"], min(args.max_new_tokens, 8)
        args.n_samples, args.max_turns = min(args.n_samples, 4), min(args.max_turns, 1)
    else:
        from nano_agentrl.retriever.client import RetrieverClient

        retriever = RetrieverClient(args.retriever_url)
    judge = None if args.judge == "none" else make_judge(args.judge, max_turns=args.max_turns)

    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir / 'checkpoints'}")
    rows = []
    for step, ckpt in ckpts:
        generator = None
        for split in args.split:
            rec_path = run_dir / "reliability" / f"step_{step:06d}" / f"{split}.jsonl"
            if rec_path.exists():
                records = load_records(rec_path)
            else:
                if generator is None:
                    generator, tokenizer = build_generator(
                        ckpt, args.dry_run, args.use_vllm, args.device
                    )
                examples = load_split(split, args.max_examples)
                records = evaluate_examples(
                    generator,
                    tokenizer,
                    examples,
                    "search",
                    retriever,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    topk=args.topk,
                    max_total_tokens=1000 if args.dry_run else 4096,
                )
                save_records(records, rec_path)
            labelled = (
                label_failures(records, judge, run_dir / "reliability" / "judge_cache.jsonl")
                if judge
                else []
            )
            row = row_for(step, split, records, labelled)
            rows.append(row)
            print(
                f"step {step:6d} {split}: pass^1 {row['pass_hat_1']:.3f}  "
                f"pass^{args.n_samples} {row[f'pass_hat_{args.n_samples}']:.3f}  "
                f"searches {row['num_searches']:.2f}"
            )
        del generator
    out = run_dir / "reliability.csv"
    write_csv(rows, out)
    print(f"wrote {out}  (labels: {', '.join(LABELS)})")
    return out


if __name__ == "__main__":
    main()
