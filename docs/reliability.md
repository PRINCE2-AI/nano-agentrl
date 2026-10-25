# Reliability analysis

Research question: **as RL raises accuracy, does the agent also become more reliable?**
Accuracy (EM, pass@1) can rise while the agent stays erratic on questions it "sometimes"
gets right. We measure the following on every checkpoint (`scripts/reliability_sweep.py`).

| Metric | What it means | Source |
|---|---|---|
| pass@k | P(at least one of k samples correct): capability ceiling | Chen et al. 2021 |
| pass^k | P(all k samples correct): consistency | τ-bench, Yao et al. 2024 |
| gap pass@k − pass^k | how much success depends on luck | ours |
| num_searches | tool use per question | Search-R1 |
| invalid_format_rate | broken `<search>` / `<answer>` protocol | ours |
| failure taxonomy | why failures happen (6 labels) | `eval/failure_taxonomy.py` |

Example: a question answered correctly in 4 of 8 samples has pass@1 0.5 and pass@8 1.0,
but pass^8 = C(4,8)/C(8,8) = 0. Reliability is about moving questions from "sometimes"
to "always".

## Protocol

1. Train with `save_every: 50` (every checkpoint is kept).
2. `python scripts/reliability_sweep.py --run_dir runs/search_3b --split nq_dev_500 hotpotqa_dev_500 --max_examples 200 --n_samples 8 --use_vllm --judge rule`
3. Hand-label 50 failures (`failure_taxonomy export`), then judge them with Claude and with
   the rules, and report the agreement (accuracy, Cohen's κ) here.
4. Re-run the sweep with `--judge claude` only if Claude agrees with the hand labels better
   than the rules do. Judgements are cached, so re-runs are free.
5. Plot with `scripts/plot.py` (figures 2, 3 and 5).

## Judge agreement on 50 hand-labelled failures

TODO (Prince): fill in after labelling `docs/hand_labels.jsonl`.

| Judge | Accuracy | Cohen's κ |
|---|---|---|
| rule | – | – |
| claude-opus-5-5 (effort low) | – | – |

## Findings: 0.5B search agent (Kaggle T4, 2026-10-13)

Sweep: checkpoints at steps 75 and 90 of `configs/search_0.5b_t4.yaml` (only the last two
were kept). 100 NQ + 100 HotpotQA questions (from the dev_500 subsets), 8 samples each at
temperature 1.0, with the rollout limits of training (2 turns, 300 tokens/turn), and the
rule judge. Data: `runs/search_0.5b_t4/reliability.csv`. Figures:
`docs/figures/search_0.5b_t4/reliability_*` and
[fig0_capability_vs_consistency](figures/search_0.5b_t4/fig0_capability_vs_consistency.png)
(pass@1 vs pass@8 vs pass^8, drawn by `scripts/plot.py` from the machine-readable
[reliability_summary.csv](reliability_summary.csv)).

| step | split | pass@1 | pass@8 | pass^4 | pass^8 | searches | invalid fmt |
|---|---|---|---|---|---|---|---|
| 75 | NQ | 0.078 | 0.21 | 0.011 | 0.00 | 0.90 | 0.106 |
| 90 | NQ | 0.086 | 0.21 | 0.035 | 0.03 | 0.98 | 0.032 |
| 75 | HotpotQA | 0.105 | 0.27 | 0.030 | 0.01 | 0.90 | 0.121 |
| 90 | HotpotQA | 0.141 | 0.28 | 0.059 | 0.04 | 0.98 | 0.049 |

Failure taxonomy (rule judge, share of failed trajectories):

| step | split | bad query | ignored relevant doc | hallucinated (no search) | format / over-search |
|---|---|---|---|---|---|
| 75 | NQ | 0.68 | 0.20 | 0.12 | <0.01 |
| 90 | NQ | 0.72 | 0.24 | 0.04 | <0.01 |
| 75 | HotpotQA | 0.68 | 0.19 | 0.12 | <0.01 |
| 90 | HotpotQA | 0.71 | 0.24 | 0.04 | <0.01 |

1. **Capability far exceeds reliability.** pass@8 is 21–28%: the agent can answer about a
   quarter of the questions *sometimes*. pass^8 is only 3–4%: almost none *every time*.
   Most solvable questions are coin flips, with a pass@8 − pass^8 gap of 18–24 points.
2. **Late training improves consistency faster than accuracy.** From step 75 to 90, pass@1
   rose by 1–4 points, while pass^4 and pass^8 grew 2–4× (from a small base). pass@8 did
   not move: the agent is not reaching new questions, it is getting more reliable on the
   ones it could already solve.
3. **pass@8 ≈ RAG's gold-in-answer** (NQ 21 vs 22, HotpotQA 28 vs 27.4). Sampling 8 times
   recovers roughly what one-shot RAG "contains". The ceiling looks set by retrieval, not
   by the policy.
4. **Most failures are retrieval failures.** In ~70% of failures the gold never appears in
   the retrieved passages ("bad query" in the rule judge: the docs lacked the answer, which
   can also mean BM25 or the corpus is at fault, not only the query). ~24% are *reading*
   failures: the gold was in the docs and the agent answered something else. Answering
   without searching fell from 12% to 4% of failures between steps 75 and 90.

Caveats: two checkpoints only, one seed, 100 questions per split (pass@1 ±~1–2 points with
8 samples), and the rule judge rather than the LLM judge (no hand-label agreement yet).

Next:
- Retrieval recall@3 of the agent's query vs the raw question (tests finding 4 and the RAG
  gap). Tooling is in `scripts/retrieval_recall.py`; it needs the retriever server and a
  GPU sweep to fill in.
- A run with `keep_last_checkpoints: 0` so the sweep can trace pass^k across the whole run.
- Claude judge + 50 hand labels to validate the taxonomy.
