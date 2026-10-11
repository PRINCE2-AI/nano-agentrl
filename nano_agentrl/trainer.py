"""GRPO training loop: rollouts, reference model, optimizer, checkpoints, weight sync, logging.

Wires together logprobs.py, grpo.py and rollout.py into the Search-R1 training recipe
(Search-R1 Sec. 3; GRPO from DeepSeekMath Sec. 4.1). One step:

    1. take `prompts_per_step` questions, sample `group_size` responses each   (rollout)
    2. score every response                                                    (rewards.py)
    3. group-relative advantages                                               (grpo.py)
    4. log-probs under policy and reference, clipped loss + KL, backward    (logprobs, grpo)
    5. clip grad norm, AdamW step, copy the new weights to the generator
    6. log metrics; every `eval_every` steps evaluate; every `save_every` steps checkpoint

Precision: master weights stay float32 and the forward runs under bf16 autocast (on CUDA).
Pure-bf16 weights would silently drop updates: bf16 keeps ~3 significant digits, so a
weight of 0.01 cannot change by lr * grad ~ 1e-6. The frozen reference model has no updates
and is stored directly in bf16.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nano_agentrl.config import Config, save_config
from nano_agentrl.generation import Generator
from nano_agentrl.grpo import compute_group_advantages, grpo_loss, zero_signal_group_frac
from nano_agentrl.logprobs import get_token_logprobs
from nano_agentrl.prompts import build_chat_prompt, gsm8k_prompt, search_r1_prompt
from nano_agentrl.rewards import compute_reward, has_valid_format
from nano_agentrl.rollout import Trajectory

CHECKPOINT_PREFIX = "step_"


# ---------------------------------------------------------------- small helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def disable_dropout(model: torch.nn.Module) -> None:
    """RL needs the same log-probs for the same tokens on every pass; dropout breaks that."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def eval_metric_name(split: str) -> str:
    """'nq_dev_500' -> 'eval/em_nq', 'gsm8k_test' -> 'eval/em_gsm8k'."""
    return f"eval/em_{split.split('_')[0]}"


def collate(trajectories: list[Trajectory], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-padded batch of prompt + response rows.

    Returns input_ids, attention_mask, loss_mask: each (B, T) int64. loss_mask is 0 over the
    prompt, padding and environment tokens, and 1 over policy-generated tokens.
    """
    rows = [t.prompt_ids + t.response_ids for t in trajectories]
    masks = [[0] * len(t.prompt_ids) + list(t.response_mask) for t in trajectories]
    T = max(len(r) for r in rows)
    B = len(rows)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    loss_mask = torch.zeros((B, T), dtype=torch.long)
    for i, (row, mask) in enumerate(zip(rows, masks, strict=True)):
        input_ids[i, : len(row)] = torch.tensor(row)
        attention_mask[i, : len(row)] = 1
        loss_mask[i, : len(mask)] = torch.tensor(mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpt_root = Path(run_dir) / "checkpoints"
    if not ckpt_root.exists():
        return None
    ckpts = sorted(p for p in ckpt_root.iterdir() if p.name.startswith(CHECKPOINT_PREFIX))
    complete = [p for p in ckpts if (p / "trainer_state.pt").exists()]
    return complete[-1] if complete else None


class MetricsLogger:
    """Appends every metrics dict to <run_dir>/metrics.jsonl, and to W&B when enabled."""

    def __init__(self, cfg: Config, run_dir: Path) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.wandb_run = None
        if cfg.use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    group=cfg.wandb_group,
                    config=asdict(cfg),
                    dir=str(run_dir),
                    resume="allow",
                )
            except Exception as e:  # logging must never kill a training run
                print(f"[warn] W&B disabled: {type(e).__name__}: {e}")
        elif cfg.use_wandb:
            print("[info] WANDB_API_KEY not set: logging to metrics.jsonl only")

    def log(self, metrics: dict[str, float], step: int) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


# ---------------------------------------------------------------- trainer


class Trainer:
    def __init__(
        self,
        cfg: Config,
        policy: torch.nn.Module,
        tokenizer: Any,
        train_data: list[dict],
        eval_data: dict[str, list[dict]],
        ref_model: torch.nn.Module | None = None,
        generator: Generator | None = None,
        retriever: Any = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.train_data = train_data
        self.eval_data = eval_data
        self.retriever = retriever

        self.policy = policy.to(self.device)
        disable_dropout(self.policy)
        if cfg.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()
        self.policy.train()

        self.ref_model = None
        if ref_model is not None and cfg.kl_coef > 0:
            self.ref_model = ref_model.to(self.device).eval()
            self.ref_model.requires_grad_(False)

        self.generator = generator or Generator(self.policy, tokenizer, backend="hf")
        self.optimizer = self._build_optimizer()
        warmup = max(cfg.warmup_steps, 0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0
        )
        self.use_autocast = cfg.bf16 and self.device.type == "cuda"

        self.step = 0
        self.epoch = 0
        self.cursor = 0
        self.order = self._epoch_order(0)
        self._prompt_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------ setup

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        kwargs = {"lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay, "betas": (0.9, 0.999)}
        if self.cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb

                return bnb.optim.AdamW8bit(params, **kwargs)
            except ImportError:
                print("[warn] bitsandbytes not installed: falling back to torch AdamW")
        return torch.optim.AdamW(params, **kwargs)

    def _autocast(self):
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------ data

    def _epoch_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.train_data)))
        random.Random(self.cfg.seed + epoch).shuffle(order)
        return order

    def next_batch(self) -> list[dict]:
        """`prompts_per_step` training examples; reshuffles (deterministically) each epoch."""
        n = self.cfg.prompts_per_step
        if self.cursor + n > len(self.order):
            self.epoch += 1
            self.order = self._epoch_order(self.epoch)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + n]
        self.cursor += n
        return [self.train_data[i] for i in idx]

    def encode_prompt(self, question: str) -> list[int]:
        if question not in self._prompt_cache:
            template = gsm8k_prompt if self.cfg.task == "gsm8k" else search_r1_prompt
            text = build_chat_prompt(self.tokenizer, template(question))
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            self._prompt_cache[question] = ids[-self.cfg.max_prompt_len :]
        return self._prompt_cache[question]

    # ------------------------------------------------------------ rollout

    def rollout(self, examples: list[dict], n: int, temperature: float) -> list[Trajectory]:
        """n trajectories per example (prompt-major), each with its reward filled in."""
        prompt_ids = [self.encode_prompt(ex["question"]) for ex in examples]
        if self.cfg.task == "search":
            trajectories = self._search_rollout(prompt_ids, examples, n, temperature)
        else:
            outputs = self.generator.generate(
                prompt_ids,
                n=n,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=temperature,
                stop_strings=self.cfg.stop_strings,
            )
            trajectories = [
                Trajectory(
                    prompt_index=o.prompt_index,
                    prompt_ids=prompt_ids[o.prompt_index],
                    response_ids=o.token_ids,
                    response_mask=[1] * len(o.token_ids),
                    text=o.text,
                    golds=examples[o.prompt_index]["golds"],
                    finish_reason=o.finish_reason,
                )
                for o in outputs
            ]
        for t in trajectories:
            t.reward = compute_reward(
                t.text,
                t.golds,
                self.cfg.task,
                self.cfg.use_format_reward,
                self.cfg.format_reward_weight,
            )
        return trajectories

    def _search_rollout(self, prompt_ids, examples, n, temperature) -> list[Trajectory]:
        raise NotImplementedError("multi-turn search rollouts are wired in milestone 2 (P12)")

    # ------------------------------------------------------------ policy update

    @torch.no_grad()
    def _frozen_logprobs(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        """Log-probs of the whole batch under `model`, micro-batched, no grad: (B, T)."""
        out = []
        mb = self.cfg.micro_batch_size
        for s in range(0, batch["input_ids"].shape[0], mb):
            with self._autocast():
                out.append(
                    get_token_logprobs(
                        model,
                        batch["input_ids"][s : s + mb],
                        batch["attention_mask"][s : s + mb],
                        batch["loss_mask"][s : s + mb],
                    )
                )
        return torch.cat(out)

    def accumulate_gradients(
        self,
        batch: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor | None,
        ref_logprobs: torch.Tensor | None,
    ) -> dict[str, float]:
        """Forward + backward over micro-batches; gradients equal one full-batch backward.

        Each micro-batch loss is re-weighted so the sum matches `loss_agg` on the full batch:
            token_mean: weight = micro-batch response tokens / batch response tokens
            seq_mean, dr_grpo: weight = micro-batch rows / batch rows
        (dr_grpo's B*T denominator stays exact because every micro-batch shares the same T.)
        If old_logprobs is None (ppo_epochs == 1) the current log-probs, detached, are used:
        the rollout policy IS the current policy, so the ratio is exactly 1.
        """
        cfg = self.cfg
        B = batch["input_ids"].shape[0]
        total_tokens = batch["loss_mask"].sum().clamp(min=1)
        sums = {"kl": 0.0, "clip_frac": 0.0, "pg_loss": 0.0, "neg_logprob": 0.0, "loss": 0.0}
        for s in range(0, B, cfg.micro_batch_size):
            sl = slice(s, s + cfg.micro_batch_size)
            mask = batch["loss_mask"][sl]
            with self._autocast():
                logprobs = get_token_logprobs(
                    self.policy, batch["input_ids"][sl], batch["attention_mask"][sl], mask
                )
            old = old_logprobs[sl] if old_logprobs is not None else logprobs.detach()
            ref = ref_logprobs[sl] if ref_logprobs is not None else None
            loss, stats = grpo_loss(
                logprobs,
                old,
                ref,
                advantages[sl],
                mask,
                clip_eps=cfg.clip_eps,
                kl_coef=cfg.kl_coef,
                loss_agg=cfg.loss_agg,
            )
            n_tok = mask.sum()
            if cfg.loss_agg == "token_mean":
                weight = n_tok / total_tokens
            else:
                weight = torch.tensor(mask.shape[0] / B)
            (loss * weight).backward()

            tok_frac = (n_tok / total_tokens).item()
            for key in ("kl", "clip_frac", "pg_loss"):
                sums[key] += stats[key] * tok_frac
            sums["neg_logprob"] += -(logprobs.detach() * mask).sum().item() / total_tokens.item()
            sums["loss"] += (loss.detach() * weight).item()
        return sums

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        t0 = time.time()
        examples = self.next_batch()
        trajectories = self.rollout(examples, cfg.group_size, cfg.temperature)
        rollout_sec = time.time() - t0

        t1 = time.time()
        rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
        advantages = compute_group_advantages(
            rewards, cfg.group_size, cfg.normalize_std, cfg.adv_eps
        ).to(self.device)
        batch = {
            k: v.to(self.device)
            for k, v in collate(trajectories, self.tokenizer.pad_token_id).items()
        }
        old_logprobs = self._frozen_logprobs(self.policy, batch) if cfg.ppo_epochs > 1 else None
        ref_logprobs = (
            self._frozen_logprobs(self.ref_model, batch) if self.ref_model is not None else None
        )

        grad_norms = []
        for epoch in range(cfg.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            sums = self.accumulate_gradients(batch, advantages, old_logprobs, ref_logprobs)
            if epoch == 0:
                first_epoch = sums
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.optimizer.step()
        self.scheduler.step()
        self.generator.sync_weights(self.policy)
        self.step += 1
        train_sec = time.time() - t1

        return {
            "train/reward_mean": rewards.mean().item(),
            "train/em": float(np.mean([t.reward == 1.0 for t in trajectories])),
            "train/response_len": float(np.mean([len(t.response_ids) for t in trajectories])),
            "train/num_searches": float(np.mean([t.num_searches for t in trajectories])),
            "train/invalid_format_rate": float(
                np.mean([not has_valid_format(t.text, cfg.task) for t in trajectories])
            ),
            "train/truncated_frac": float(
                np.mean([t.finish_reason in ("length", "max_turns") for t in trajectories])
            ),
            "train/kl": sums["kl"],
            # On-policy estimate: E[-log pi(token)] over sampled tokens = the policy's entropy.
            "train/entropy": first_epoch["neg_logprob"],
            "train/clip_frac": sums["clip_frac"],
            "train/pg_loss": sums["pg_loss"],
            "train/loss": sums["loss"],
            "train/grad_norm": float(np.mean(grad_norms)),
            "train/zero_signal_group_frac": zero_signal_group_frac(rewards, cfg.group_size),
            "train/lr": self.scheduler.get_last_lr()[0],
            "perf/rollout_sec": rollout_sec,
            "perf/train_sec": train_sec,
            "perf/gpu_mem_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            ),
        }

    # ------------------------------------------------------------ eval

    def evaluate(self, chunk_size: int = 32) -> dict[str, float]:
        """Greedy decoding on every eval split; saves a few sample trajectories per split."""
        metrics = {}
        for split, examples in self.eval_data.items():
            trajectories: list[Trajectory] = []
            for s in range(0, len(examples), chunk_size):
                trajectories += self.rollout(examples[s : s + chunk_size], n=1, temperature=0.0)
            metrics[eval_metric_name(split)] = float(
                np.mean([t.reward == 1.0 for t in trajectories])
            )
            metrics[f"eval/num_searches_{split.split('_')[0]}"] = float(
                np.mean([t.num_searches for t in trajectories])
            )
            self.save_samples(split, trajectories[: self.cfg.num_sample_trajectories])
        return metrics

    def save_samples(self, split: str, trajectories: list[Trajectory]) -> Path:
        """jsonl of trajectories, viewable with scripts/render_trajectory.py."""
        out_dir = self.run_dir / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{self.step:06d}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t in trajectories:
                row = asdict(t)
                row["prompt_text"] = self.tokenizer.decode(t.prompt_ids)
                row["response_tokens"] = [self.tokenizer.decode([i]) for i in t.response_ids]
                f.write(json.dumps(row) + "\n")
        return path

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self) -> Path:
        ckpt = self.run_dir / "checkpoints" / f"{CHECKPOINT_PREFIX}{self.step:06d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        torch.save(state, ckpt / "trainer_state.pt")  # written last: marks the ckpt complete
        self._prune_checkpoints()
        return ckpt

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.keep_last_checkpoints
        if keep <= 0:
            return
        ckpts = sorted((self.run_dir / "checkpoints").glob(f"{CHECKPOINT_PREFIX}*"))
        for old in ckpts[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    def load_trainer_state(self, ckpt: Path) -> None:
        """Restore optimizer, schedule, data position and RNGs. Policy weights are loaded
        from the checkpoint before the Trainer is built (see build_trainer)."""
        state = torch.load(ckpt / "trainer_state.pt", map_location=self.device, weights_only=False)
        self.step, self.epoch, self.cursor = state["step"], state["epoch"], state["cursor"]
        self.order = self._epoch_order(self.epoch)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        print(f"[resume] from {ckpt} at step {self.step}")

    # ------------------------------------------------------------ main loop

    def train(self, logger: MetricsLogger | None = None) -> None:
        cfg = self.cfg
        save_config(cfg, self.run_dir)
        logger = logger or MetricsLogger(cfg, self.run_dir)
        if self.step == 0 and cfg.eval_every > 0:
            logger.log(self.evaluate(), step=0)  # baseline before any update
        while self.step < cfg.total_steps:
            metrics = self.train_step()
            if self.step % cfg.log_every == 0:
                logger.log(metrics, step=self.step)
                print(
                    f"step {self.step:5d}  reward {metrics['train/reward_mean']:.3f}  "
                    f"kl {metrics['train/kl']:.4f}  clip {metrics['train/clip_frac']:.3f}  "
                    f"len {metrics['train/response_len']:.0f}  "
                    f"rollout {metrics['perf/rollout_sec']:.1f}s  "
                    f"train {metrics['perf/train_sec']:.1f}s"
                )
            if cfg.eval_every > 0 and self.step % cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                logger.log(eval_metrics, step=self.step)
                print(f"step {self.step:5d}  eval {eval_metrics}")
            if self.step % cfg.save_every == 0 or self.step == cfg.total_steps:
                self.save_checkpoint()
        logger.finish()


# ---------------------------------------------------------------- building from a config


def build_trainer(cfg: Config, dry_run: bool = False) -> Trainer:
    """Load models, tokenizer and data for `cfg`. Resumes from the latest checkpoint in
    cfg.run_dir if there is one. dry_run uses a tiny random GPT-2 and synthetic data."""
    from nano_agentrl.data import load_split, load_train_dataset

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    ckpt = find_latest_checkpoint(cfg.run_dir)

    if dry_run:
        from transformers import GPT2LMHeadModel

        from nano_agentrl.tiny import make_char_tokenizer, make_tiny_model

        tokenizer = make_char_tokenizer()
        policy = (
            GPT2LMHeadModel.from_pretrained(ckpt)
            if ckpt
            else make_tiny_model(len(tokenizer), seed=cfg.seed)
        )
        ref_model = make_tiny_model(len(tokenizer), seed=cfg.seed) if cfg.kl_coef > 0 else None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        policy = AutoModelForCausalLM.from_pretrained(ckpt or cfg.model_name, dtype=torch.float32)
        ref_model = None
        if cfg.kl_coef > 0:
            ref_dtype = torch.bfloat16 if cfg.bf16 and device.type == "cuda" else torch.float32
            ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=ref_dtype)

    train_data = load_train_dataset(cfg.train_dataset, cfg.max_train_examples, cfg.seed)
    eval_data = {s: load_split(s, cfg.eval_num_examples, seed=0) for s in cfg.eval_splits}

    generator = None
    if cfg.use_vllm and not dry_run:
        generator = Generator(policy, tokenizer, backend="vllm", model_name=cfg.model_name)

    retriever = None
    if cfg.task == "search":
        from nano_agentrl.retriever.client import RetrieverClient

        retriever = RetrieverClient(cfg.retriever_url)

    trainer = Trainer(
        cfg, policy, tokenizer, train_data, eval_data, ref_model, generator, retriever
    )
    if ckpt:
        trainer.load_trainer_state(ckpt)
        trainer.generator.sync_weights(trainer.policy)
    return trainer
