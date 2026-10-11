# Experiment log

Notable results and decisions, newest last.

## 2026-10-03: Repo scaffolding (P0)

- Created package skeleton, flat `Config` dataclass + YAML loader (unknown keys raise),
  `configs/gsm8k_0.5b.yaml`, and a smoke test.
- Defaults: G=8, clip_eps=0.2, kl_coef=0.001 (k3), max_turns=4, topk=3, temperature=1.0,
  loss_agg=token_mean, format reward off.

## 2026-10-11: Milestone 1 code complete (P2–P7)

- P2 `get_token_logprobs`: position ids from the attention mask, so left padding gives the
  same log-probs as an unpadded sequence. Tests compare against a one-token-at-a-time loop.
- P4 GRPO: population std in the advantage (so [1,0,0,1] gives ±1). `clip_frac` counts
  tokens where the clipped branch won, i.e. tokens that got zero gradient.
- P6 trainer: float32 master weights with bf16 autocast. Pure-bf16 weights would drop lr≈1e-6
  updates. With `ppo_epochs: 1`, old log-probs are the detached current log-probs, so the
  ratio is exactly 1 and one forward pass is saved. `train/entropy` is the on-policy estimate
  E[−log π(token)], not the full-vocab entropy.
- No real training run yet: the dev laptop is CPU-only with ~4 GB free RAM. The first real
  run is planned on a GPU box.

### How to compare our trainer with TRL fairly (P7, `scripts/sanity_trl.py`)

1. **Same config file.** `sanity_trl.py` reads the same YAML and maps it field by field:
   G → `num_generations`, kl_coef → `beta`, clip_eps → `epsilon`, ppo_epochs → `num_iterations`,
   loss_agg token_mean/seq_mean/dr_grpo → `loss_type` dapo/grpo/dr_grpo,
   normalize_std → `scale_rewards` group/none. Batch: per_device × grad_accum = prompts × G.
2. **Same prompt, data and reward code** (`prompts.gsm8k_prompt`, `rewards.compute_reward`).
3. **Same seed, but expect different samples.** The two trainers draw random numbers in a
   different order, so compare the *mean of 3 seeds* per trainer, not single curves.
4. **Compare on the x-axis that matters.** Use optimizer steps, since each step sees
   prompts_per_step × G completions in both. Wall-clock differs (TRL can use vLLM and kernels).
5. **Metrics to overlay.** Our `train/reward_mean` against TRL's `train/reward`, and
   `train/kl` against TRL's `train/kl`. Response length is our `train/response_len` against
   TRL's `train/completions/mean_length`. Then the clip fraction.
6. **Known differences that are fine.** TRL's std may be the unbiased one (÷(G−1)). TRL
   computes log-probs with its own fused kernel. Bf16 rounding differs. Curves should agree
   within seed noise. A consistent gap (e.g. ours flat while TRL's reward rises) means a bug.
7. **Platform.** TRL ≥ 1.x needs Triton for GRPO, so run the comparison on the Linux GPU box.
   The tests only check the config mapping and the reward function.
