# CLAUDE.md — nano-AgentRL

## What this project is

nano-AgentRL is a minimal, readable, from-scratch PyTorch re-implementation of
**Search-R1** (Jin et al., 2025, arXiv:2503.09516): training an LLM with
reinforcement learning (GRPO) to interleave reasoning with search-engine calls.

Three goals, in order:
1. **From-scratch trainer**: multi-turn GRPO with retrieved-token loss masking,
   in ~1,500 lines of core code that a reader can follow end to end.
2. **Reproduction**: NQ + HotpotQA Exact Match results close to the paper,
   on a single 80GB GPU (Qwen2.5-0.5B for development, then 1.5B, then 3B).
3. **Original research angle**: measure how agent *reliability* changes during
   RL training (pass^k from τ-bench, search-call counts, format errors, a
   failure taxonomy), alongside accuracy.

Reference implementation for *reading only*: https://github.com/PeterGriffinJin/Search-R1
(built on verl). We do not import verl. TRL's GRPOTrainer may be used only in
`scripts/sanity_trl.py` as a sanity check against our own trainer.

## IMPORTANT: learning rules (read before every task)

The owner (Prince) is learning RL through this project. These files are
**HAND-WRITTEN BY PRINCE**:

- `nano_agentrl/logprobs.py`
- `nano_agentrl/grpo.py`
- `nano_agentrl/rollout.py`
- `scripts/toy_reinforce.py`

For these files, Claude must NOT write the implementation, even if asked casually.
Claude MAY:
- create the file with function signatures, type hints, and detailed docstrings
  (what the function must do, shapes of every tensor, which paper equation it
  implements), with the body as `raise NotImplementedError`;
- write unit tests for them (tests are encouraged — write them first);
- explain concepts, equations, and PyTorch APIs;
- review Prince's code and point out bugs as **hints** (say where and why it is
  wrong, not the corrected code), unless Prince explicitly writes
  "show me the fix" in that message.

Everything else (retriever server, eval harness, rewards, configs, trainer
plumbing, logging, plotting, scripts, README, docs) Claude may write fully.

When an explanation is requested, explain in simple English, step by step,
with a tiny numeric example where possible.

## Tech stack

- Python 3.11, PyTorch 2.x (bf16), Hugging Face `transformers`, `datasets`
- `vllm` for rollouts (optional at first; HF `generate` fallback must work)
- FastAPI + uvicorn for the retrieval server; `pyserini` or `rank_bm25` for BM25;
  `faiss` for dense (E5) retrieval later
- `wandb` for experiment tracking; `matplotlib` for plots
- `pytest`, `ruff`; dependencies in `pyproject.toml`
- Configs in YAML under `configs/`, loaded into a dataclass

## Repo structure

```
nano-agentrl/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/                  # gsm8k_0.5b.yaml, search_0.5b.yaml, search_1.5b.yaml, search_3b.yaml
├── nano_agentrl/
│   ├── config.py             # dataclass config + YAML loader
│   ├── logprobs.py           # HAND-WRITTEN: per-token log-probs with masks
│   ├── grpo.py               # HAND-WRITTEN: group advantages + GRPO / Dr.GRPO loss
│   ├── rollout.py            # HAND-WRITTEN: multi-turn generate → search → continue, loss_mask
│   ├── rewards.py            # exact match + format reward
│   ├── generation.py         # thin wrapper: vLLM or HF generate with stop strings
│   ├── trainer.py            # main loop, reference model, optimizer, checkpoints, weight sync
│   ├── retriever/
│   │   ├── server.py         # FastAPI: POST /search {queries: [...], topk: 3}
│   │   └── client.py
│   └── eval/
│       ├── em.py             # SQuAD-style normalization + EM
│       ├── pass_k.py         # pass@k and pass^k
│       ├── evaluate.py       # run a model/checkpoint over a dataset split
│       └── failure_taxonomy.py
├── scripts/                  # train.py, evaluate.py, plot.py, render_trajectory.py,
│                             # toy_reinforce.py (HAND-WRITTEN), sanity_trl.py
├── tests/
└── docs/                     # grpo_math.md, results.md, reliability.md, experiment_log.md
```

## Key technical spec

- **Trajectory format** (follow Search-R1): the model reasons in
  `<think>...</think>`, searches with `<search>query</search>`, the environment
  appends `<information>top-k docs</information>`, and the model finishes with
  `<answer>...</answer>`. Copy the exact instruction prompt template from the
  Search-R1 paper/repo into `nano_agentrl/prompts.py` and cite the source.
- **Generation stops** at `</search>` or `</answer>` (stop strings on decoded
  text, not token ids). `max_turns` default 4, `topk` default 3.
- **Reward**: outcome-only. 1.0 if the normalized `<answer>` exactly matches any
  gold answer (SQuAD normalization), else 0.0. Optional small format reward,
  off by default, behind a config flag.
- **loss_mask**: 1 for tokens the policy generated, 0 for prompt tokens and for
  every token inside retrieved `<information>` blocks.
- **GRPO**: for each question sample G responses (default G=8). Advantage
  A_i = (r_i − mean(r)) / (std(r) + eps), broadcast to all of response i's
  tokens. PPO-style clipped ratio (eps_clip default 0.2), KL to a frozen
  reference model (k3 estimator, coefficient configurable). Config switch for
  Dr. GRPO (no std normalization, no per-sequence length normalization).
  Log the fraction of groups where all rewards are equal (zero signal).
- **Rollouts** at temperature 1.0; evaluation greedy unless measuring pass^k.
- **Retrieval corpus**: wiki-18 from the Search-R1 HF collection
  (`PeterJinGo/wiki-18-corpus`, BM25 index `PeterJinGo/wiki-18-bm25-index`).
  Training data: `PeterJinGo/nq_hotpotqa_train`.

## Metrics to log (W&B, keep these names stable)

`train/reward_mean`, `train/em`, `train/response_len`, `train/num_searches`,
`train/invalid_format_rate`, `train/kl`, `train/entropy`, `train/clip_frac`,
`train/grad_norm`, `train/zero_signal_group_frac`, `eval/em_nq`, `eval/em_hotpotqa`,
`eval/pass_hat_k` (k = 1..8), `perf/rollout_sec`, `perf/train_sec`, `perf/gpu_mem_gb`.

## Coding conventions

- Small, readable functions. Type hints everywhere. Docstrings state tensor
  shapes, e.g. `logprobs: (B, T) float32`.
- At the top of each core file, a comment naming the paper section/equation it implements.
- No clever abstractions; prefer explicit code a student can follow.
- Every new module gets tests in `tests/`. Tests must run on **CPU in under 60s**
  using a tiny randomly initialised model built in the test from
  `GPT2Config(n_layer=2, n_head=2, n_embd=64)` — no downloads in tests.
- Seeds set via config; every run writes its resolved config to the run dir.
- Never commit data, indexes, checkpoints, or `.env` (keep `.gitignore` updated).

## Commands

- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q`
- Lint: `ruff check . && ruff format --check .`
- Retriever: `python -m nano_agentrl.retriever.server --index <path> --corpus <path>`
- Train: `python scripts/train.py --config configs/search_0.5b.yaml`
- Evaluate: `python scripts/evaluate.py --ckpt <path> --split nq_dev_500`

## Workflow

- Work milestone by milestone (see PROMPTS.md). Do not start the next milestone
  until tests for the current one pass.
- After each task: run `pytest -q` and `ruff check .`, summarise what changed,
  and list anything Prince must do by hand.
- Append notable experiment results and decisions to `docs/experiment_log.md`.
- Small commits with clear messages.
