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

## 2026-10-11: Milestones 2 and 3 code complete (P8–P15)

- P8 retriever: Pyserini (wiki-18 Lucene index, needs Java) and rank_bm25 backends behind one
  FastAPI API. ~4.8k q/s on the 6-doc test corpus; the real number on wiki-18 is still TODO.
- P9/P10: the prompt, `<information>` format and invalid-action message are copied verbatim
  from the Search-R1 repo. All 4,000 checked dataset prompts match our template exactly.
  Eval splits are fixed-seed 500-question subsets of the Search-R1 test parquet (NQ test,
  HotpotQA dev).
- Bug found by tests: eval `qid` restarted per batch, which merged questions in pass^k.
- Bug found by tests: the rollout length guard ran after a turn, so the context could pass
  the model's limit. It now stops when another turn could overflow.
- Qwen2.5-0.5B-Instruct before RL (8 NQ questions, direct mode): EM 0, invalid format
  100%. The model ignores the `<answer>` tags.
- Search-R1 GRPO reference settings (train_grpo.sh): 512 prompts × G=5, lr 1e-6, warmup
  0.285, kl 0.001 low_var_kl, max_turns 2, 1005 steps. Ours on one GPU: 64 prompts × G=8
  for 3B, max_turns 4.
- P13: the failure judge uses claude-opus-5-5 at effort low, with JSON-schema output and
  server-side refusal fallback. Judgements are cached on disk.
- Next: the first real run (`search_0.5b`) on a GPU box.

## 2026-10-12: First real run, GSM8K GRPO, Qwen2.5-0.5B-Instruct, Kaggle 2×T4

Config `configs/gsm8k_0.5b_t4.yaml`: 200 steps, 8 prompts × G=8, fp32 + gradient
checkpointing, reference model on cuda:1, HF generate. 83 s/step, 4.8 h total, 10.6 GB on GPU 0.
Metrics: `runs/gsm8k_0.5b_t4/metrics.jsonl` (not in git). Figures: `docs/figures/gsm8k_0.5b_t4/`.

**Held-out EM (greedy, 100 GSM8K test questions, ±5% standard error):**
0.36 at step 0, then 0.47, 0.45, 0.49, 0.51, 0.46, 0.48, 0.52, 0.47 at steps 25–200.
That is +12 points, and almost all of it arrives by step 25. After that it is flat within noise.

| steps | train reward | invalid format | truncated | entropy | len | KL |
|---|---|---|---|---|---|---|
| 1–25 | 0.26 | 0.93 | 0.31 | 1.04 | 223 | 0.06 |
| 26–50 | 0.40 | 0.93 | 0.03 | 0.41 | 141 | 0.22 |
| 51–75 | 0.49 | 0.34 | 0.15 | 0.28 | 206 | 0.20 |
| 76–100 | 0.51 | 0.33 | 0.19 | 0.21 | 218 | 0.36 |
| 101–125 | 0.54 | 0.23 | 0.10 | 0.23 | 192 | 0.42 |
| 126–150 | 0.55 | 0.27 | 0.07 | 0.25 | 190 | 0.31 |
| 151–175 | 0.49 | 0.83 | 0.04 | 0.28 | 159 | 0.37 |
| 176–200 | 0.60 | 0.84 | 0.05 | 0.24 | 163 | 0.27 |

Findings:
1. **The early gain is mostly "stop rambling".** Truncation at the 300-token cap peaks at
   50–65% around steps 5–10 and falls to ~3% by step 30. Over the same window held-out EM
   jumps 0.36 → 0.47. Truncated answers have no final number, so they score 0.
2. **The gain is NOT format learning (my earlier guess was wrong).** The invalid-format rate
   is ~93% for steps 1–50, drops to 20–35% for steps 51–150, then drifts back to ~85% after
   step 150, while reward keeps rising. The cause: format reward is off, and GSM8K scoring
   falls back to "last number in the text", so the `<answer>` tag is never rewarded and the
   format drifts freely. For the search task this cannot happen, because EM requires an
   `<answer>` block. For GSM8K, `use_format_reward: true` or strict extraction would pin it.
3. **Entropy collapses fast**: 1.5 → 0.25 nats by step 60. kl_coef 0.001 barely constrains
   the policy: median KL is 0.26, with spikes of 2.1 (step 102) and 2.0 (step 174). Train
   reward still climbs 0.49 → 0.60 after step 50 while held-out EM is flat. That is probably
   noise plus overconfidence, not new skill.
4. **27–40% of groups carry zero signal** (all 8 samples equally right or wrong). That
   is a third of each batch wasted, and an argument for larger G or harder-prompt filtering.

Next:
- Re-evaluate step 0 and step 200 on all 1,319 test questions (±1.4%) before claiming
  the +12.
- Read samples to see which output format the model drifted to.
- Second run options: `use_format_reward: true`, `kl_coef: 0.01`. Then move to the search task.
