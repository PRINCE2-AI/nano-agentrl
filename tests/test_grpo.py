"""Tests for nano_agentrl.grpo: advantages, clipping, masking, KL and loss aggregation."""

from __future__ import annotations

import math

import pytest
import torch

from nano_agentrl.grpo import (
    compute_group_advantages,
    grpo_loss,
    k3_kl,
    zero_signal_group_frac,
)

# ---------------------------------------------------------------- advantages


def test_advantages_numeric_example():
    adv = compute_group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]), group_size=4)
    torch.testing.assert_close(adv, torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-5, rtol=0)


def test_all_equal_group_gives_zero_advantage():
    adv = compute_group_advantages(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), group_size=3)
    assert (adv == 0).all()


def test_groups_are_independent():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    adv = compute_group_advantages(rewards, group_size=4)
    torch.testing.assert_close(adv[:4], compute_group_advantages(rewards[:4], 4))
    torch.testing.assert_close(adv[4:], compute_group_advantages(rewards[4:], 4))


def test_dr_grpo_advantages_skip_std():
    adv = compute_group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]), 4, normalize_std=False)
    torch.testing.assert_close(adv, torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_bad_group_size_raises():
    with pytest.raises(ValueError):
        compute_group_advantages(torch.zeros(5), group_size=4)


def test_zero_signal_group_frac():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])  # groups of 2: equal, equal, mixed
    assert zero_signal_group_frac(rewards, 2) == pytest.approx(2 / 3)


# ---------------------------------------------------------------- loss helpers


def make_inputs(log_ratio: float, adv: float, B: int = 2, T: int = 4):
    """logprobs (leaf, requires grad) with a constant log-ratio to old_logprobs."""
    old = torch.full((B, T), -1.0)
    logprobs = (old + log_ratio).clone().requires_grad_(True)
    advantages = torch.full((B,), adv)
    mask = torch.ones(B, T)
    return logprobs, old, advantages, mask


def test_ratio_is_one_on_first_step():
    logprobs, old, adv, mask = make_inputs(0.0, adv=1.0)
    loss, stats = grpo_loss(logprobs, old.clone(), None, adv, mask)
    assert stats["ratio_mean"] == pytest.approx(1.0)
    assert stats["clip_frac"] == 0.0
    assert loss.item() == pytest.approx(-1.0)  # -ratio * A = -1
    loss.backward()
    # d(-ratio*A)/d logprob = -ratio*A / n_tokens = -1/8 for every token
    torch.testing.assert_close(logprobs.grad, torch.full_like(logprobs, -1 / 8))


@pytest.mark.parametrize(
    "ratio, adv, should_clip",
    [
        (1.1, 1.0, False),  # inside the band
        (0.9, -1.0, False),  # inside the band
        (1.3, 1.0, True),  # above 1 + eps with A > 0: clipped
        (0.7, -1.0, True),  # below 1 - eps with A < 0: clipped
        (1.3, -1.0, False),  # above the band but A < 0: min keeps the unclipped term
        (0.7, 1.0, False),  # below the band but A > 0: min keeps the unclipped term
    ],
)
def test_clipping_only_beyond_band(ratio, adv, should_clip):
    logprobs, old, advantages, mask = make_inputs(math.log(ratio), adv)
    loss, stats = grpo_loss(logprobs, old, None, advantages, mask, clip_eps=0.2)
    loss.backward()
    if should_clip:
        assert stats["clip_frac"] == 1.0
        assert (logprobs.grad == 0).all()
    else:
        assert stats["clip_frac"] == 0.0
        assert (logprobs.grad != 0).all()


def test_masked_tokens_get_no_gradient():
    torch.manual_seed(0)
    logprobs = (-torch.rand(3, 6)).requires_grad_(True)
    old = logprobs.detach() + 0.05 * torch.randn(3, 6)
    ref = logprobs.detach() + 0.05 * torch.randn(3, 6)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1]]).float()
    adv = torch.tensor([1.0, -0.5, 2.0])
    for agg in ("token_mean", "seq_mean", "dr_grpo"):
        logprobs.grad = None
        loss, _ = grpo_loss(logprobs, old, ref, adv, mask, kl_coef=0.1, loss_agg=agg)
        loss.backward()
        assert (logprobs.grad[mask == 0] == 0).all(), agg
        assert (logprobs.grad[mask == 1] != 0).all(), agg


def test_masked_garbage_values_do_not_leak():
    """Huge values at masked positions (e.g. retrieved tokens) must not change the loss."""
    logprobs, old, adv, mask = make_inputs(0.0, adv=1.0)
    mask[:, -1] = 0
    clean, _ = grpo_loss(logprobs, old, old, adv, mask, kl_coef=0.1)
    old_bad = old.clone()
    old_bad[:, -1] = -1e4
    dirty, _ = grpo_loss(logprobs, old_bad, old_bad, adv, mask, kl_coef=0.1)
    assert torch.isfinite(dirty) and dirty.item() == pytest.approx(clean.item())


# ---------------------------------------------------------------- KL


def test_kl_zero_when_policy_equals_reference():
    logprobs, old, adv, mask = make_inputs(0.0, adv=1.0)
    _, stats = grpo_loss(logprobs, old, logprobs.detach().clone(), adv, mask, kl_coef=0.5)
    assert stats["kl"] == 0.0


def test_k3_numeric_example_and_nonnegative():
    x = torch.tensor([-0.1])  # ref - policy
    torch.testing.assert_close(k3_kl(torch.zeros(1), x), torch.tensor([math.exp(-0.1) + 0.1 - 1]))
    torch.manual_seed(0)
    assert (k3_kl(torch.randn(100), torch.randn(100)) >= 0).all()


def test_kl_term_added_to_loss():
    logprobs, old, adv, mask = make_inputs(0.0, adv=0.0)  # zero advantage: loss is KL only
    ref = old - 0.1
    loss, stats = grpo_loss(logprobs, old, ref, adv, mask, kl_coef=2.0)
    expected_kl = math.exp(-0.1) + 0.1 - 1
    assert stats["kl"] == pytest.approx(expected_kl, rel=1e-4)
    assert loss.item() == pytest.approx(2.0 * expected_kl, rel=1e-4)


# ---------------------------------------------------------------- aggregation


def test_loss_aggregation_modes():
    """Per-token loss = -A = -1 on every response token. Row 0 has 1 token, row 1 has 3."""
    logprobs = torch.zeros(2, 4, requires_grad=True)
    old = torch.zeros(2, 4)
    adv = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]]).float()
    token_mean, _ = grpo_loss(logprobs, old, None, adv, mask, loss_agg="token_mean")
    seq_mean, _ = grpo_loss(logprobs, old, None, adv, mask, loss_agg="seq_mean")
    dr_grpo, _ = grpo_loss(logprobs, old, None, adv, mask, loss_agg="dr_grpo")
    assert token_mean.item() == pytest.approx(-4 / 4)
    assert seq_mean.item() == pytest.approx(-(1 / 1 + 3 / 3) / 2)
    assert dr_grpo.item() == pytest.approx(-4 / (2 * 4))


def test_unknown_loss_agg_raises():
    logprobs, old, adv, mask = make_inputs(0.0, adv=1.0)
    with pytest.raises(ValueError):
        grpo_loss(logprobs, old, None, adv, mask, loss_agg="nope")


def test_zero_signal_stat_with_group_size():
    logprobs, old, _, mask = make_inputs(0.0, adv=0.0, B=4)
    adv = compute_group_advantages(torch.tensor([1.0, 1.0, 1.0, 0.0]), group_size=2)
    _, stats = grpo_loss(logprobs, old, None, adv, mask, group_size=2)
    assert stats["zero_signal_group_frac"] == pytest.approx(0.5)
