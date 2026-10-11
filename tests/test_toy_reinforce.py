"""Tests for scripts/toy_reinforce.py (fast, CPU only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("toy_reinforce", ROOT / "scripts/toy_reinforce.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def test_discounted_returns_numeric_example():
    out = toy.discounted_returns([1.0, 1.0, 1.0], gamma=0.9)
    torch.testing.assert_close(out, torch.tensor([2.71, 1.9, 1.0]))


def test_llm_task_reward_improves():
    toy.set_seed(0)
    rewards = toy.run_llm(steps=25, lr=1e-2, seed=0, group_size=8, gen_len=4)
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.1


def test_cartpole_runs_and_plots(tmp_path):
    pytest.importorskip("gymnasium")
    toy.set_seed(0)
    rewards = toy.run_cartpole(episodes=5, lr=1e-2, gamma=0.99, seed=0)
    assert len(rewards) == 5 and all(r > 0 for r in rewards)
    out = toy.plot_rewards(rewards, tmp_path / "curve.png", "test")
    assert out.exists()
