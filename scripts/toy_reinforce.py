"""Toy REINFORCE: CartPole-v1, then a tiny "LLM" task. The simplest policy gradient.

Implements REINFORCE (Williams, 1992):

    grad J(theta) = E[ sum_t  grad log pi_theta(a_t | s_t) * (G_t - b) ]

where G_t is the discounted return from step t onward and b is a baseline that lowers
variance without adding bias. GRPO is this same idea with three changes: the baseline is
the mean reward of a *group* of samples for the same prompt, the update is clipped
(PPO-style), and a KL term keeps the policy near a reference model.

Usage:
    python scripts/toy_reinforce.py --task cartpole            # needs: pip install gymnasium
    python scripts/toy_reinforce.py --task llm                 # tiny random GPT-2, CPU only
Both save a reward curve to runs/toy_reinforce_<task>.png.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...   Returns: (T,) float32.

    Example, rewards [1, 1, 1], gamma 0.9:
        G_2 = 1
        G_1 = 1 + 0.9 * 1    = 1.9
        G_0 = 1 + 0.9 * 1.9  = 2.71
    """
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------- CartPole


class MLPPolicy(nn.Module):
    """obs (obs_dim,) -> logits over actions (n_actions,)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def run_cartpole(episodes: int, lr: float, gamma: float, seed: int) -> list[float]:
    """Train REINFORCE on CartPole-v1. Returns the total reward of every episode."""
    try:
        import gymnasium as gym
    except ImportError as e:
        raise SystemExit("CartPole needs gymnasium: pip install gymnasium") from e

    env = gym.make("CartPole-v1")
    policy = MLPPolicy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(episodes):
        # 1. Roll out one full episode with the current policy, remembering log pi(a_t|s_t).
        obs, _ = env.reset(seed=seed + episode)
        log_probs, rewards = [], []
        done = False
        while not done:
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(float(reward))
            done = terminated or truncated

        # 2. Discounted return G_t for every step, then normalise (the baseline).
        returns = discounted_returns(rewards, gamma)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Policy-gradient loss: minimise -sum_t log pi(a_t|s_t) * A_t.
        loss = -(torch.stack(log_probs) * advantages).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))
        if episode % 50 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(f"episode {episode:4d}  reward(avg last 50) {recent:6.1f}")

    env.close()
    return episode_rewards


# ---------------------------------------------------------------- toy LLM task


def build_tiny_lm(vocab_size: int) -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=vocab_size, n_positions=32)
    return GPT2LMHeadModel(config)


def run_llm(
    steps: int, lr: float, seed: int, group_size: int = 16, gen_len: int = 6, target: int = 7
) -> list[float]:
    """REINFORCE on a random tiny GPT-2: reward = fraction of generated tokens equal to `target`.

    Each step samples `group_size` completions of the same prompt and uses the group mean
    reward as the baseline, which is exactly GRPO's advantage without the std or clipping.
    Returns the mean reward of every step.
    """
    vocab = 16
    model = build_tiny_lm(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    prompt = torch.tensor([[1, 2, 3]]).repeat(group_size, 1)  # (G, P)
    step_rewards = []

    for step in range(steps):
        # 1. Sample G completions token by token, keeping the log-prob of each choice.
        ids = prompt
        token_logprobs = []
        for _ in range(gen_len):
            logits = model(input_ids=ids).logits[:, -1, :]  # (G, V)
            dist = torch.distributions.Categorical(logits=logits)
            token = dist.sample()  # (G,)
            token_logprobs.append(dist.log_prob(token))
            ids = torch.cat([ids, token[:, None]], dim=1)
        seq_logprob = torch.stack(token_logprobs, dim=1).sum(dim=1)  # (G,)

        # 2. Reward per completion, and the group baseline.
        generated = ids[:, prompt.shape[1] :]
        rewards = (generated == target).float().mean(dim=1)  # (G,)
        advantages = rewards - rewards.mean()

        # 3. Loss = -mean_i A_i * log pi(o_i).
        loss = -(advantages * seq_logprob).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_rewards.append(rewards.mean().item())
        if step % 20 == 0:
            print(f"step {step:4d}  mean reward {rewards.mean().item():.3f}")
    return step_rewards


# ---------------------------------------------------------------- plotting + CLI


def plot_rewards(rewards: list[float], path: str | Path, title: str, window: int = 20) -> Path:
    """Save raw rewards plus a moving average to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(rewards, alpha=0.3, label="per step")
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smooth, label=f"moving avg ({window})")
    ax.set_xlabel("episode / step")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy REINFORCE on CartPole or a tiny LM")
    parser.add_argument("--task", choices=["cartpole", "llm"], default="cartpole")
    parser.add_argument("--episodes", type=int, default=500, help="CartPole episodes")
    parser.add_argument("--steps", type=int, default=200, help="LLM-task steps")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.task == "cartpole":
        rewards = run_cartpole(args.episodes, args.lr or 1e-2, args.gamma, args.seed)
    else:
        rewards = run_llm(args.steps, args.lr or 1e-3, args.seed)
    out = plot_rewards(rewards, Path(args.out_dir) / f"toy_reinforce_{args.task}.png", args.task)
    print(f"saved reward curve to {out}")


if __name__ == "__main__":
    main()
