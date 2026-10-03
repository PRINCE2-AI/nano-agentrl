"""Experiment config: one flat dataclass plus a YAML loader.

Defaults follow the "Key technical spec" in CLAUDE.md (Search-R1, Jin et al. 2025,
arXiv:2503.09516). A YAML file only needs to list the fields it changes; unknown
keys are an error so typos fail loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_TASKS = ("gsm8k", "search")
VALID_LOSS_AGG = ("token_mean", "seq_mean", "dr_grpo")


@dataclass
class Config:
    # --- run ---
    task: str = "search"  # "gsm8k" (single-turn math) or "search" (multi-turn Search-R1)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    seed: int = 42
    run_dir: str = "runs/default"

    # --- data ---
    train_dataset: str = "PeterJinGo/nq_hotpotqa_train"
    eval_splits: list[str] = field(default_factory=lambda: ["nq_dev_500", "hotpotqa_dev_500"])
    max_prompt_len: int = 512

    # --- rollout ---
    group_size: int = 8  # G responses sampled per question
    temperature: float = 1.0  # rollouts; eval is greedy unless measuring pass^k
    max_new_tokens: int = 512  # per generation turn
    max_turns: int = 4
    stop_strings: list[str] = field(default_factory=lambda: ["</search>", "</answer>"])
    use_vllm: bool = False  # HF generate fallback must always work

    # --- retrieval ---
    retriever_url: str = "http://127.0.0.1:8000"
    topk: int = 3

    # --- reward ---
    use_format_reward: bool = False  # outcome-only EM by default
    format_reward_weight: float = 0.1

    # --- GRPO ---
    clip_eps: float = 0.2
    kl_coef: float = 0.001  # k3 estimator against the frozen reference model
    normalize_std: bool = True  # False for Dr. GRPO
    adv_eps: float = 1e-6
    loss_agg: str = "token_mean"  # "token_mean" | "seq_mean" | "dr_grpo"
    ppo_epochs: int = 1

    # --- optimisation ---
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    prompts_per_step: int = 16  # questions per step; batch = prompts_per_step * group_size
    micro_batch_size: int = 4
    total_steps: int = 1000
    warmup_steps: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False

    # --- logging / checkpoints ---
    wandb_project: str = "nano-agentrl"
    wandb_run_name: str | None = None
    log_every: int = 1
    eval_every: int = 50
    save_every: int = 100
    num_sample_trajectories: int = 4

    def __post_init__(self) -> None:
        if self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {self.task!r}")
        if self.loss_agg not in VALID_LOSS_AGG:
            raise ValueError(f"loss_agg must be one of {VALID_LOSS_AGG}, got {self.loss_agg!r}")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 for group-relative advantages")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML file into a Config. `overrides` (e.g. from the CLI) win over the file."""
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    values.update(overrides or {})

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return Config(**values)


def save_config(config: Config, run_dir: str | Path) -> Path:
    """Write the fully resolved config to <run_dir>/config.yaml and return that path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
    return out
