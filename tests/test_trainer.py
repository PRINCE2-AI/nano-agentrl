"""Tests for trainer.py and scripts/train.py --dry_run (tiny model, CPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nano_agentrl.config import Config
from nano_agentrl.rollout import Trajectory
from nano_agentrl.tiny import make_arithmetic_dataset, make_char_tokenizer, make_tiny_model
from nano_agentrl.trainer import Trainer, collate, eval_metric_name, find_latest_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts/train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)
CONFIG = str(ROOT / "configs/gsm8k_0.5b.yaml")


def read_metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_collate_masks():
    trajs = [
        Trajectory(0, [5, 6], [7, 8, 9], [1, 0, 1], "x"),
        Trajectory(0, [5], [7], [1], "y"),
    ]
    batch = collate(trajs, pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8, 9], [5, 7, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    assert batch["loss_mask"].tolist() == [[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]]


def test_eval_metric_name():
    assert eval_metric_name("nq_dev_500") == "eval/em_nq"
    assert eval_metric_name("hotpotqa_dev_500") == "eval/em_hotpotqa"


@pytest.mark.parametrize("loss_agg", ["token_mean", "seq_mean", "dr_grpo"])
def test_micro_batching_matches_full_batch(tmp_path, loss_agg):
    """Gradient accumulation over micro-batches must equal one full-batch backward."""
    tok = make_char_tokenizer()
    torch.manual_seed(0)
    trajs = [Trajectory(0, [10, 11, 12], [13 + i] * (2 + i), [1] * (2 + i), "") for i in range(4)]
    trajs[2].response_mask = [1, 0, 1, 1]  # an environment token in the middle
    batch = collate(trajs, tok.pad_token_id)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

    grads = []
    for mb in (4, 1):
        cfg = Config(
            task="gsm8k",
            device="cpu",
            micro_batch_size=mb,
            loss_agg=loss_agg,
            kl_coef=0.0,
            run_dir=str(tmp_path / f"mb{mb}"),
            gradient_checkpointing=False,
        )
        model = make_tiny_model(len(tok), seed=0)
        trainer = Trainer(cfg, model, tok, train_data=[], eval_data={})
        trainer.optimizer.zero_grad()
        trainer.accumulate_gradients(batch, adv, None, None)
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(grads[0], grads[1], atol=1e-6, rtol=1e-4)


def test_trainer_step_changes_weights_and_logs(tmp_path):
    tok = make_char_tokenizer()
    cfg = Config(
        task="gsm8k",
        device="cpu",
        group_size=4,
        prompts_per_step=2,
        max_new_tokens=6,
        micro_batch_size=4,
        lr=1e-2,
        kl_coef=0.1,
        run_dir=str(tmp_path),
        gradient_checkpointing=False,
        use_wandb=False,
    )
    model = make_tiny_model(len(tok), seed=0)
    ref = make_tiny_model(len(tok), seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    trainer = Trainer(cfg, model, tok, make_arithmetic_dataset(8), {}, ref_model=ref)
    # Force some reward variance so the advantage is non-zero.
    trainer.rollout = _rollout_with_rewards(trainer.rollout, [1.0, 0.0, 0.0, 1.0] * 2)
    metrics = trainer.train_step()
    after = list(model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    for key in (
        "train/reward_mean",
        "train/kl",
        "train/entropy",
        "train/clip_frac",
        "train/grad_norm",
        "train/zero_signal_group_frac",
        "perf/rollout_sec",
    ):
        assert key in metrics
    assert metrics["train/zero_signal_group_frac"] == 0.0
    assert metrics["train/kl"] == pytest.approx(0.0, abs=1e-6)  # policy == ref before the update


def _rollout_with_rewards(rollout_fn, rewards):
    def wrapped(*args, **kwargs):
        trajs = rollout_fn(*args, **kwargs)
        for t, r in zip(trajs, rewards, strict=False):
            t.reward = r
        return trajs

    return wrapped


def test_dry_run_end_to_end_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    train_script.main(["--config", CONFIG, "--dry_run", "--run_dir", str(run_dir)])

    metrics = read_metrics(run_dir)
    train_rows = [m for m in metrics if "train/reward_mean" in m]
    assert [m["step"] for m in train_rows] == [1, 2]
    assert any("eval/em_arithmetic" in m for m in metrics)
    assert (run_dir / "config.yaml").exists()
    assert find_latest_checkpoint(run_dir).name == "step_000002"
    assert list((run_dir / "samples").glob("*.jsonl"))

    # Resume: same run_dir, more steps. Must continue at step 3, not restart.
    train_script.main(
        ["--config", CONFIG, "--dry_run", "--run_dir", str(run_dir), "--total_steps", "3"]
    )
    steps = [m["step"] for m in read_metrics(run_dir) if "train/reward_mean" in m]
    assert steps == [1, 2, 3]
    assert find_latest_checkpoint(run_dir).name == "step_000003"


@pytest.mark.parametrize("config", ["search_0.5b.yaml", "search_3b.yaml"])
def test_search_dry_run_end_to_end(tmp_path, config):
    """Multi-turn search training with the in-memory retriever: loss masks, metrics, samples."""
    run_dir = tmp_path / "run"
    train_script.main(
        ["--config", str(ROOT / "configs" / config), "--dry_run", "--run_dir", str(run_dir)]
    )
    rows = [m for m in read_metrics(run_dir) if "train/reward_mean" in m]
    assert [m["step"] for m in rows] == [1, 2]
    for key in ("train/num_searches", "train/invalid_format_rate", "train/zero_signal_group_frac"):
        assert key in rows[0]
    assert any("eval/em_tinyqa" in m for m in read_metrics(run_dir))
    samples = list((run_dir / "samples").glob("*tinyqa_test.jsonl"))
    assert samples
    row = json.loads(samples[0].read_text().splitlines()[0])
    assert len(row["response_tokens"]) == len(row["response_mask"])


def test_nomask_ablation_trains_on_retrieved_tokens(tmp_path):
    run_dir = tmp_path / "run"
    cfg = str(ROOT / "configs/search_0.5b.yaml")
    train_script.main(
        [
            "--config",
            cfg,
            "--dry_run",
            "--run_dir",
            str(run_dir),
            "--set",
            "mask_retrieved_tokens=false",
            "--total_steps",
            "1",
        ]
    )
    sample = next((run_dir / "samples").glob("*.jsonl"))
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    assert all(all(m == 1 for m in r["response_mask"]) for r in rows)


def test_raw_prompt_without_chat_template(tmp_path):
    tok = make_char_tokenizer()
    tok.chat_template = "<|user|>{{ messages[0]['content'] }}<|assistant|>"
    for use_template, wrapped in ((True, True), (False, False)):
        cfg = Config(
            task="search",
            device="cpu",
            use_chat_template=use_template,
            run_dir=str(tmp_path),
            gradient_checkpointing=False,
            max_prompt_len=4096,
        )
        trainer = Trainer(cfg, make_tiny_model(len(tok)), tok, [], {})
        text = tok.decode(trainer.encode_prompt("Who wrote Hamlet?"))
        assert text.startswith("<|user|>") is wrapped
        assert text.rstrip().endswith("Question: Who wrote Hamlet?") or "<|assistant|>" in text


def test_ablation_script_flags_are_valid_config_fields():
    import dataclasses
    import re

    script = (ROOT / "scripts/run_ablations.sh").read_text()
    keys = set(re.findall(r"--set (\w+)=", script))
    assert keys >= {
        "mask_retrieved_tokens",
        "loss_agg",
        "normalize_std",
        "group_size",
        "model_name",
        "use_chat_template",
        "seed",
    }
    assert keys <= {f.name for f in dataclasses.fields(Config)}
