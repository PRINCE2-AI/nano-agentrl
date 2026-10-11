"""Sanity check: run TRL's GRPOTrainer on the same GSM8K setup to compare curves with our trainer.

Reads the same YAML config as scripts/train.py and translates it to a TRL GRPOConfig, using
the same prompt template, reward function and data. Logs to W&B with group "trl_sanity",
so its train/reward curve can sit next to ours. See docs/experiment_log.md for how to compare
the two fairly.

Usage:
    pip install -e ".[trl]"
    python scripts/sanity_trl.py --config configs/gsm8k_0.5b.yaml
    python scripts/sanity_trl.py --config configs/gsm8k_0.5b.yaml --dry_run   # tiny model, CPU
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_agentrl.config import Config, load_config  # noqa: E402
from nano_agentrl.data import load_train_dataset  # noqa: E402
from nano_agentrl.prompts import gsm8k_prompt, search_r1_prompt  # noqa: E402
from nano_agentrl.rewards import compute_reward  # noqa: E402

# Our loss_agg -> TRL loss_type (TRL >= 0.20 names).
LOSS_TYPE = {"token_mean": "dapo", "seq_mean": "grpo", "dr_grpo": "dr_grpo"}


def build_grpo_config(cfg: Config, output_dir: str, use_cpu: bool = False):
    """Translate our Config into a TRL GRPOConfig. Unknown fields are dropped with a notice,
    so the script survives TRL renames."""
    from trl import GRPOConfig

    completions_per_step = cfg.prompts_per_step * cfg.group_size
    wanted = {
        "output_dir": output_dir,
        "seed": cfg.seed,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "warmup_steps": cfg.warmup_steps,
        "lr_scheduler_type": "constant_with_warmup",
        "max_steps": cfg.total_steps,
        "per_device_train_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": max(1, completions_per_step // cfg.micro_batch_size),
        "num_generations": cfg.group_size,
        "max_completion_length": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "beta": cfg.kl_coef,
        "epsilon": cfg.clip_eps,
        "num_iterations": cfg.ppo_epochs,
        "loss_type": LOSS_TYPE[cfg.loss_agg],
        "scale_rewards": "group" if cfg.normalize_std else "none",
        "bf16": cfg.bf16 and not use_cpu,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "use_vllm": cfg.use_vllm,
        "logging_steps": cfg.log_every,
        "save_strategy": "no",
        "report_to": ["wandb"] if cfg.use_wandb else [],
        "run_name": f"trl_{cfg.wandb_run_name or Path(cfg.run_dir).name}",
        "use_cpu": use_cpu,
    }
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(set(wanted) - known)
    if dropped:
        print(f"[info] this TRL version has no {dropped}; using its defaults")
    return GRPOConfig(**{k: v for k, v in wanted.items() if k in known})


def build_dataset(cfg: Config, conversational: bool):
    """HF Dataset with a 'prompt' column (chat messages or plain text) and 'golds'."""
    from datasets import Dataset

    template = gsm8k_prompt if cfg.task == "gsm8k" else search_r1_prompt
    rows = load_train_dataset(cfg.train_dataset, cfg.max_train_examples, cfg.seed)
    records = []
    for r in rows:
        text = template(r["question"])
        prompt = [{"role": "user", "content": text}] if conversational else text
        records.append({"prompt": prompt, "golds": r["golds"]})
    return Dataset.from_list(records)


def make_reward_fn(cfg: Config):
    def reward_fn(prompts, completions, golds, **kwargs) -> list[float]:
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [
            compute_reward(t, g, cfg.task, cfg.use_format_reward, cfg.format_reward_weight)
            for t, g in zip(texts, golds, strict=True)
        ]

    reward_fn.__name__ = "outcome_reward"  # TRL logs rewards/<name>/mean
    return reward_fn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TRL GRPOTrainer sanity check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true", help="1 step, tiny model, CPU")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args(argv)

    import os

    from trl import GRPOTrainer

    try:  # TRL >= 1.x computes GRPO log-probs with Triton kernels, which exist on Linux only
        from trl.kernels import ChunkedLogProbFunction  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"TRL's GRPOTrainer needs Triton here ({e}). Run this script on the Linux GPU box."
        ) from e

    overrides = {"wandb_group": "trl_sanity"}
    if args.dry_run:
        overrides.update(
            total_steps=1,
            prompts_per_step=2,
            group_size=2,
            micro_batch_size=2,
            max_new_tokens=8,
            train_dataset="arithmetic",
            use_wandb=False,
            use_vllm=False,
            gradient_checkpointing=False,
            bf16=False,
        )
    cfg = load_config(args.config, overrides)
    if cfg.task == "search":
        raise SystemExit("sanity_trl.py covers the single-turn GSM8K setup only")
    os.environ.setdefault("WANDB_RUN_GROUP", cfg.wandb_group or "trl_sanity")
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    output_dir = args.output_dir or str(
        Path(cfg.run_dir).with_name(Path(cfg.run_dir).name + "_trl")
    )
    grpo_config = build_grpo_config(cfg, output_dir, use_cpu=args.dry_run)

    if args.dry_run:
        from nano_agentrl.tiny import make_char_tokenizer, make_tiny_model

        tokenizer = make_char_tokenizer()
        model = str(Path(output_dir) / "tiny_model")  # TRL reloads the model by path
        make_tiny_model(len(tokenizer), seed=cfg.seed).save_pretrained(model)
        tokenizer.save_pretrained(model)
        dataset = build_dataset(cfg, conversational=False)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = cfg.model_name
        dataset = build_dataset(cfg, conversational=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(cfg),
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
