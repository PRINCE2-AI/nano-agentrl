"""Entry point: python scripts/train.py --config configs/<name>.yaml

Examples:
    python scripts/train.py --config configs/gsm8k_0.5b.yaml
    python scripts/train.py --config configs/gsm8k_0.5b.yaml --dry_run  # 2 CPU steps
    python scripts/train.py --config configs/gsm8k_0.5b.yaml --set lr=2e-6 --set group_size=4

Re-running with the same run_dir resumes from its latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Less fragmentation when logits of different lengths come and go (must precede torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nano_agentrl.config import load_config  # noqa: E402
from nano_agentrl.trainer import build_trainer, find_latest_checkpoint  # noqa: E402

# Settings that make a run finish in seconds on a CPU with a tiny random model.
DRY_RUN_OVERRIDES = {
    "device": "cpu",
    "total_steps": 2,
    "prompts_per_step": 2,
    "group_size": 4,
    "micro_batch_size": 4,
    "max_new_tokens": 8,
    "eval_every": 2,
    "save_every": 2,
    "eval_num_examples": 4,
    "num_sample_trajectories": 2,
    "use_wandb": False,
    "use_vllm": False,
    "bf16": False,
    "gradient_checkpointing": False,
    "lr": 1e-3,
}
DRY_RUN_TASK_OVERRIDES = {
    "gsm8k": {"train_dataset": "arithmetic", "eval_splits": ["arithmetic_test"]},
    # tiny corpus + in-memory BM25 instead of the wiki-18 server; tiny GPT-2 has 1024 positions
    "search": {
        "train_dataset": "tinyqa",
        "eval_splits": ["tinyqa_test"],
        "max_turns": 2,
        "max_obs_tokens": 200,
        "max_total_tokens": 1000,
    },
}


def parse_overrides(pairs: list[str]) -> dict:
    """['lr=2e-6', 'eval_splits=[a, b]'] -> {'lr': 2e-06, 'eval_splits': ['a', 'b']}."""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GRPO training (nano-AgentRL)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true", help="2 steps, tiny model, CPU")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="no training: evaluate --ckpt on the config's eval_splits and write a json",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help='with --eval_only: checkpoint dir, or "base" for the untrained model '
        "(default: latest checkpoint in run_dir)",
    )
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args(argv)

    overrides = {}
    if args.dry_run:
        task = load_config(args.config).task
        overrides = {**DRY_RUN_OVERRIDES, **DRY_RUN_TASK_OVERRIDES[task], "run_dir": "runs/dry_run"}
    overrides.update(parse_overrides(args.set))
    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps

    if args.eval_only:
        evaluate_only(args, overrides)
        return
    cfg = load_config(args.config, overrides)
    trainer = build_trainer(cfg, dry_run=args.dry_run)
    trainer.train()
    print(f"done: {cfg.run_dir}")


def evaluate_only(args: argparse.Namespace, overrides: dict) -> dict:
    """Greedy eval of one checkpoint (or the base model). Writes <out>/eval.json and
    sample trajectories to <out>/samples/, where out = <run_dir>/eval_<label>."""
    ckpt = args.ckpt
    if ckpt is None:  # latest checkpoint of this run
        base_cfg = load_config(args.config, overrides)
        latest = find_latest_checkpoint(base_cfg.run_dir)
        if latest is None:
            raise SystemExit(f"no checkpoint in {base_cfg.run_dir}; pass --ckpt <dir> or base")
        ckpt = str(latest)
    label = "base" if ckpt == "base" else Path(ckpt).name
    overrides = {**overrides, "kl_coef": 0.0, "use_wandb": False, "num_sample_trajectories": 20}
    overrides["run_dir"] = str(Path(load_config(args.config, overrides).run_dir) / f"eval_{label}")
    cfg = load_config(args.config, overrides)

    trainer = build_trainer(cfg, dry_run=args.dry_run, policy_path=ckpt)
    n = {split: len(examples) for split, examples in trainer.eval_data.items()}
    print(f"[eval_only] {ckpt}: {n} questions")
    start = time.time()
    metrics = trainer.evaluate(chunk_size=args.eval_batch_size)
    result = {"ckpt": ckpt, "num_questions": n, "seconds": time.time() - start, **metrics}
    out = Path(cfg.run_dir) / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
