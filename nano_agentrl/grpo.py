"""Group-relative advantages and the GRPO / Dr. GRPO clipped policy loss with k3 KL.

Implements DeepSeekMath (Shao et al., 2024) Sec. 4.1: the outcome-supervised advantage
(Sec. 4.1.2), the clipped objective (Eq. 3) and the unbiased k3 KL estimator (Eq. 4).
loss_agg="dr_grpo" and normalize_std=False give Dr. GRPO ("Understanding R1-Zero-Like
Training: A Critical Perspective", Liu et al., 2025, Sec. 3.1). See docs/grpo_math.md.
"""

from __future__ import annotations

import torch

LOSS_AGGS = ("token_mean", "seq_mean", "dr_grpo")


def compute_group_advantages(
    rewards: torch.Tensor, group_size: int, normalize_std: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """A_i = (r_i - mean(group)) / (std(group) + eps), one scalar per response.

    Args:
        rewards: (N * G,) float. Responses of the same prompt are contiguous:
            rewards[0:G] belong to prompt 0, rewards[G:2G] to prompt 1, ...
        group_size: G, responses per prompt.
        normalize_std: divide by the group std (GRPO). False gives Dr. GRPO: A_i = r_i - mean.
        eps: keeps the division finite when every reward in a group is equal.

    Returns:
        advantages: (N * G,) float32. A group whose rewards are all equal gets exactly 0
        (no learning signal). The std is the population std (divide by G, not G - 1).

    Example, G = 4, rewards [1, 0, 0, 1]: mean 0.5, std 0.5, advantages [+1, -1, -1, +1].
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(f"{rewards.numel()} rewards is not a multiple of group_size {group_size}")
    grouped = rewards.float().view(-1, group_size)  # (N, G)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_std:
        centered = centered / (grouped.std(dim=1, unbiased=False, keepdim=True) + eps)
    return centered.view(-1)


def zero_signal_group_frac(rewards: torch.Tensor, group_size: int) -> float:
    """Fraction of prompts whose G rewards are all equal (all right or all wrong).

    Those groups have zero advantage everywhere, so they contribute no policy gradient.
    rewards: (N * G,). Returns a float in [0, 1].
    """
    grouped = rewards.view(-1, group_size)
    all_equal = (grouped == grouped[:, :1]).all(dim=1)
    return all_equal.float().mean().item()


def k3_kl(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimator of KL(pi_theta || pi_ref) (DeepSeekMath Eq. 4; Schulman, 2020).

        x  = log pi_ref(o_t) - log pi_theta(o_t)
        k3 = exp(x) - x - 1

    Always >= 0, exactly 0 when the two log-probs agree, and its expectation under
    pi_theta equals the true KL. Example: x = -0.1 -> e^-0.1 + 0.1 - 1 = 0.00484.
    Shapes: (B, T) in, (B, T) out.
    """
    x = ref_logprobs - logprobs
    return torch.exp(x) - x - 1


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    loss_agg: str = "token_mean",
    group_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped GRPO policy loss plus k3 KL penalty (DeepSeekMath Eq. 3, written as a loss).

    Args:
        logprobs: (B, T) float32, log pi_theta of each token under the current policy (with grad).
        old_logprobs: (B, T) float32, log pi_old: the policy that generated the rollouts (no grad).
        ref_logprobs: (B, T) float32 from the frozen reference model, or None if kl_coef == 0.
        advantages: (B,) one scalar per response (broadcast to all of its tokens),
            or (B, T) already broadcast.
        loss_mask: (B, T), 1 for policy-generated tokens, 0 for prompt / padding / retrieved.
        clip_eps: PPO clip range; the ratio is clipped to [1 - eps, 1 + eps].
        kl_coef: beta, weight of the k3 KL term.
        loss_agg: how per-token losses become one number:
            "token_mean": sum over all response tokens / number of response tokens
                          (every token weighs the same; DAPO-style).
            "seq_mean":   mean over sequences of (sum over its tokens / its length)
                          (original GRPO, Eq. 3's 1/|o_i|).
            "dr_grpo":    sum over all response tokens / (B * T), a constant denominator,
                          so long responses are not down-weighted (Dr. GRPO).
        group_size: if given, zero_signal_group_frac is computed per group of this size;
            otherwise it is the fraction of responses with advantage exactly 0.

    Per token, with ratio = exp(logprobs - old_logprobs):
        pg   = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)
        loss = pg + kl_coef * k3_kl(logprobs, ref_logprobs)

    Returns:
        loss: scalar tensor to call .backward() on.
        stats: floats "pg_loss", "kl", "clip_frac" (fraction of response tokens where the
            clipped term was the smaller one, i.e. the token got no gradient), "ratio_mean",
            "zero_signal_group_frac". All are means over response tokens.
    """
    if loss_agg not in LOSS_AGGS:
        raise ValueError(f"loss_agg must be one of {LOSS_AGGS}, got {loss_agg!r}")
    mask = loss_mask.to(logprobs.dtype)
    if advantages.dim() == 1:
        advantages = advantages[:, None].expand_as(logprobs)
    advantages = advantages.to(logprobs.dtype)

    # Clipped surrogate. Masked positions are set to log-ratio 0 so they stay finite.
    log_ratio = torch.where(mask.bool(), logprobs - old_logprobs, torch.zeros_like(logprobs))
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg = -torch.min(unclipped, clipped)  # (B, T)

    if ref_logprobs is not None and kl_coef != 0:
        kl = k3_kl(logprobs, torch.where(mask.bool(), ref_logprobs, logprobs))
    else:
        kl = torch.zeros_like(logprobs)
    per_token = (pg + kl_coef * kl) * mask  # masked tokens are exactly 0

    if loss_agg == "token_mean":
        loss = per_token.sum() / mask.sum().clamp(min=1)
    elif loss_agg == "seq_mean":
        seq_len = mask.sum(dim=1).clamp(min=1)
        loss = (per_token.sum(dim=1) / seq_len).mean()
    else:  # dr_grpo
        loss = per_token.sum() / (mask.shape[0] * mask.shape[1])

    with torch.no_grad():
        n_tokens = mask.sum().clamp(min=1)
        seq_adv = advantages[:, 0] if advantages.dim() == 2 else advantages
        if group_size is not None:
            zero_frac = (seq_adv.view(-1, group_size) == 0).all(dim=1).float().mean().item()
        else:
            zero_frac = (seq_adv == 0).float().mean().item()
        stats = {
            "pg_loss": ((pg * mask).sum() / n_tokens).item(),
            "kl": ((kl * mask).sum() / n_tokens).item(),
            "clip_frac": (((clipped < unclipped).to(mask.dtype) * mask).sum() / n_tokens).item(),
            "ratio_mean": ((ratio * mask).sum() / n_tokens).item(),
            "zero_signal_group_frac": zero_frac,
        }
    return loss, stats
