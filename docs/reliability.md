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

## Findings

TODO: fill in from the sweep CSVs. Do not write conclusions before the runs exist.
