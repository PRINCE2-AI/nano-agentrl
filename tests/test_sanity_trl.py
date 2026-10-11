"""scripts/sanity_trl.py: our config must map onto TRL's GRPOConfig with the same settings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from nano_agentrl.config import load_config

pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sanity_trl", ROOT / "scripts/sanity_trl.py")
sanity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanity)


@pytest.mark.parametrize(
    "loss_agg, normalize_std, loss_type, scale",
    [("token_mean", True, "dapo", "group"), ("dr_grpo", False, "dr_grpo", "none")],
)
def test_config_mapping(tmp_path, loss_agg, normalize_std, loss_type, scale):
    cfg = load_config(
        ROOT / "configs/gsm8k_0.5b.yaml",
        {"loss_agg": loss_agg, "normalize_std": normalize_std, "use_wandb": False},
    )
    g = sanity.build_grpo_config(cfg, str(tmp_path), use_cpu=True)
    assert g.num_generations == cfg.group_size
    assert g.beta == cfg.kl_coef and g.epsilon == cfg.clip_eps
    assert g.loss_type == loss_type
    assert str(g.scale_rewards).lower() == scale
    completions = g.per_device_train_batch_size * g.gradient_accumulation_steps
    assert completions == cfg.prompts_per_step * cfg.group_size


def test_reward_fn_matches_ours():
    cfg = load_config(ROOT / "configs/gsm8k_0.5b.yaml")
    fn = sanity.make_reward_fn(cfg)
    completions = [[{"role": "assistant", "content": "<answer>4</answer>"}], "<answer>5</answer>"]
    assert fn(prompts=["a", "b"], completions=completions, golds=[["4"], ["4"]]) == [1.0, 0.0]
