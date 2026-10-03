# PROMPTS.md — Claude Code ke liye step-by-step prompts

**Kaise use karna hai**

1. Ek khaali folder `nano-agentrl` banao, `git init` karo, aur `CLAUDE.md` ko repo ke root mein rakho. Claude Code har session mein ise automatically padhta hai.
2. Neeche ke prompts **order mein** ek-ek karke Claude Code ko do. Har prompt ke baad tests pass hone chahiye, tabhi agla prompt dena.
3. **[CLAUDE]** wale steps Claude Code poora banayega. **[TUM]** wale steps mein Claude sirf skeleton + tests banayega, aur implementation tum likhoge. Wahi tumhari RL ki asli seekh hai.
4. Har step ke baad `git commit` karo.

Neeche ke prompts English mein hain taaki Claude Code exact samjhe. Tum unhe Hinglish mein bhi de sakte ho.

---

## Milestone 0: Repo setup

### P0 [CLAUDE]: Scaffolding
```
Read CLAUDE.md fully. Set up the repo skeleton exactly as described in "Repo structure":
pyproject.toml (with a [dev] extra: pytest, ruff), .gitignore (data/, indexes/, checkpoints/,
runs/, wandb/, .env), nano_agentrl/config.py (dataclass + YAML loader with sensible defaults
from the "Key technical spec" section), empty modules with module docstrings, configs/gsm8k_0.5b.yaml,
docs/experiment_log.md, and a README.md placeholder. For the HAND-WRITTEN files only create
a module docstring for now. Add one smoke test that imports every module and loads every config.
Run pytest and ruff, then summarise.
```

### P1 [CLAUDE]: Environment check script
```
Create scripts/check_env.py that prints: torch + CUDA version, GPU name and memory,
whether vllm/transformers/wandb import, then loads Qwen/Qwen2.5-0.5B-Instruct, generates
one answer to "What is 2+2?" and logs a dummy metric to wandb (skip wandb if WANDB_API_KEY
is not set). Keep it robust: clear error messages, no crashes on CPU-only machines.
```

---

## Milestone 1: GRPO on math (single-turn)

### P2 [TUM]: logprobs.py (skeleton + tests)
```
For nano_agentrl/logprobs.py (HAND-WRITTEN file — follow the learning rules in CLAUDE.md):
create the function signature
  get_token_logprobs(model, input_ids, attention_mask, response_mask) -> Tensor
with a detailed docstring: inputs/outputs with shapes, the off-by-one shift between logits
and labels, how padding is handled, and why we compute log_softmax then gather.
Body: raise NotImplementedError. Then write thorough tests in tests/test_logprobs.py using a
tiny random GPT2 (built from GPT2Config, no downloads): compare against a slow per-token
reference loop, left- and right-padding cases, masked positions must be exactly 0.
Also add a short explanation in docs/grpo_math.md section "Token log-probs".
Do NOT implement the function.
```
Ab **tum** function likho. Atak jao to:
```
I implemented get_token_logprobs; test X fails. Review my code and give me a HINT only
(where and why it is wrong), not the corrected code.
```

### P3 [TUM]: toy REINFORCE
```
I will hand-write scripts/toy_reinforce.py (REINFORCE on CartPole-v1 with gymnasium, then
a toy LLM task). Create only: a file skeleton with TODO comments describing each step of the
algorithm in order, argparse flags, and a plotting helper that saves the reward curve to
runs/toy_reinforce.png. Explain the REINFORCE update in 10 lines with a numeric example.
Do NOT write the policy-gradient update itself.
```

### P4 [TUM]: grpo.py (skeleton + tests)
```
For nano_agentrl/grpo.py (HAND-WRITTEN): create signatures and detailed docstrings for
  compute_group_advantages(rewards: (N*G,), group_size, normalize_std: bool, eps) -> (N*G,)
  grpo_loss(logprobs, old_logprobs, ref_logprobs, advantages, loss_mask,
            clip_eps, kl_coef, loss_agg: "token_mean" | "seq_mean" | "dr_grpo") -> (loss, stats dict)
Docstrings must cite DeepSeekMath Sec 4 (GRPO), the Dr. GRPO paper for loss_agg="dr_grpo",
and explain the k3 KL estimator. stats must include clip_frac, kl, zero_signal_group_frac.
Body: raise NotImplementedError. Write tests: advantages for [1,0,0,1] → [+1,-1,-1,+1];
all-equal group → all zeros; ratio==1 on first step; clipping activates only beyond 1±eps;
masked tokens contribute nothing to the gradient; kl==0 when policy==reference.
Do NOT implement.
```

### P5 [CLAUDE]: GSM8K reward + generation wrapper
```
Implement nano_agentrl/rewards.py (GSM8K numeric answer extraction + correctness reward,
plus an optional format reward) and nano_agentrl/generation.py: a wrapper that samples
G completions per prompt with stop strings, using vLLM if available else HF generate,
returning texts and token ids. Add a method to load new policy weights into vLLM
(document the approach and its cost). Tests for rewards on tricky strings.
```

### P6 [CLAUDE]: Trainer plumbing (uses my functions)
```
Implement nano_agentrl/trainer.py and scripts/train.py for SINGLE-TURN GRPO on GSM8K.
The trainer must call my hand-written get_token_logprobs, compute_group_advantages and
grpo_loss — do not re-implement them. You own: data loading, batching, frozen reference
model, AdamW (optional 8-bit via bitsandbytes), bf16, gradient checkpointing, gradient
accumulation, ppo_epochs updates per batch, weight sync to the generator, periodic eval,
checkpointing every N steps (resumable), W&B logging with the metric names in CLAUDE.md,
and timing/memory stats. Config: configs/gsm8k_0.5b.yaml. Add a --dry_run flag that runs
2 steps with a tiny random model on CPU, and a test that calls it.
```

### P7 [CLAUDE]: TRL sanity check
```
Create scripts/sanity_trl.py that runs TRL's GRPOTrainer on the same GSM8K setup and config
values, logging to W&B under a different run group, so I can compare curves with my trainer.
Add a short section in docs/experiment_log.md on how to compare them fairly.
```

---

## Milestone 2: Retriever + multi-turn rollouts

### P8 [CLAUDE]: Retrieval server
```
Implement nano_agentrl/retriever/server.py and client.py: FastAPI server serving BM25
over the wiki-18 corpus (download instructions for PeterJinGo/wiki-18-corpus and
PeterJinGo/wiki-18-bm25-index in README), POST /search {queries: list[str], topk: int}
→ list of lists of {id, title, text}, batched, with request timing. Add a tiny in-memory
corpus fixture so tests run without downloads. Add a /health endpoint and a CLI benchmark
that reports queries/sec.
```

### P9 [CLAUDE]: Eval harness + baselines
```
Implement nano_agentrl/eval/em.py (SQuAD normalization, EM against multiple golds),
nano_agentrl/eval/evaluate.py and scripts/evaluate.py: fixed-seed 500-question dev subsets
for NQ and HotpotQA; three baselines (direct answer, retrieve-once RAG, ReAct-style prompting
with search) for Qwen2.5-3B-Instruct; and evaluating a released Search-R1 checkpoint from the
PeterJinGo HF collection with the same harness. Save results to docs/results.md as a table.
```

### P10 [TUM]: rollout.py (skeleton + tests)
```
For nano_agentrl/rollout.py (HAND-WRITTEN): create signatures + detailed docstrings for a
batched multi-turn rollout: generate until </search> or </answer>, parse the query, call the
retriever client, append <information>...</information>, continue, up to max_turns; handle
trajectories finishing at different times; build input_ids, attention_mask and loss_mask
(1 = policy tokens, 0 = prompt and retrieved tokens); count searches and invalid actions.
Use the prompt template in nano_agentrl/prompts.py (copy it from the Search-R1 repo and cite it).
Body: raise NotImplementedError. Write tests with a fake generator and fake retriever:
mask is exactly 0 over every <information> span, max_turns cutoff works, malformed tags are
counted as invalid, batch with mixed finishing turns is assembled correctly.
Do NOT implement.
```

### P11 [CLAUDE]: Trajectory viewer
```
Create scripts/render_trajectory.py: takes a saved rollout batch (jsonl) and writes a
self-contained HTML file where policy tokens are normal, masked (retrieved) tokens are grey,
tags are coloured, and each trajectory shows its reward and number of searches.
Make the trainer save a few sample trajectories every eval step.
```

### P12 [CLAUDE]: Wire multi-turn into the trainer
```
Extend trainer.py so that, when config.task == "search", it uses my rollout.py and the EM
reward, logs num_searches, invalid_format_rate and zero_signal_group_frac, and saves sample
trajectories. Create configs/search_0.5b.yaml, search_1.5b.yaml and search_3b.yaml with
memory-saving settings documented inline. Keep --dry_run working.
```

---

## Milestone 3 + research: Reliability analysis

### P13 [CLAUDE]: pass^k and failure taxonomy
```
Implement nano_agentrl/eval/pass_k.py (pass@k and pass^k from τ-bench, unbiased estimators
from n samples, with tests) and nano_agentrl/eval/failure_taxonomy.py: an LLM-as-judge that
labels failed trajectories as one of {bad_query, ignored_relevant_doc, hallucinated_answer,
over_searching, format_error, other} with a short rationale; cache judge outputs; report
agreement on a 50-example hand-labelled set I will create. Add scripts/reliability_sweep.py
that evaluates every saved checkpoint and writes a CSV.
```

### P14 [CLAUDE]: Plots + ablation runner
```
Create scripts/plot.py producing publication-quality figures from W&B exports / CSVs:
(1) EM vs step, (2) pass^k (k=1..8) vs step, (3) accuracy vs reliability over training,
(4) searches per question vs step, (5) failure taxonomy stacked bars per checkpoint.
Also scripts/run_ablations.sh for: loss mask on/off, GRPO vs Dr.GRPO, G=4 vs 8, base vs instruct.
```

### P15 [CLAUDE]: README, blog draft, report skeleton
```
Write README.md: hook line, headline figure, results table (ours vs paper vs released
checkpoint), 3-command quickstart, "How it works" with a diagram of the rollout + loss mask,
file map linking to core files, findings, limitations, citation. Then draft docs/blog.md
("What broke and what I learned") from docs/experiment_log.md, and a LaTeX report skeleton
in report/ (arXiv style) with sections: Intro, Background, Implementation, Reproduction,
Reliability Analysis, Ablations, Limitations. Mark clearly where I must fill in numbers.
```

---

## Har waqt kaam aane wale prompts

**Concept samajhna ho**
```
Explain <concept> in simple English, step by step, with a tiny numeric example, and show
exactly which lines of my code correspond to it. Do not change any files.
```

**Training kaam nahi kar rahi**
```
Training is not improving. Here are the W&B metrics for the last 200 steps: <paste>.
Using the pitfalls in CLAUDE.md and the Search-R1 / DAPO / RAGEN papers, list the 5 most
likely causes ranked by probability, and for each give a cheap experiment to confirm it.
Do not edit my hand-written files.
```

**Code review**
```
Review my <file> against <paper section>. List bugs and risks as hints with line numbers.
Do not rewrite it.
```

**Interview practice**
```
Act as an interviewer at an AI lab. Ask me 10 deep questions about this repo — GRPO,
loss masking, my reliability findings, and debugging decisions — one at a time, and grade
each answer honestly.
```
