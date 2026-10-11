"""Unbiased pass@k and pass^k (tau-bench, Yao et al. 2024) estimators from n samples.

For one question with n sampled answers of which c are correct:

    pass@k = P(at least one of k samples is correct) = 1 - C(n - c, k) / C(n, k)
             (Chen et al. 2021, "Evaluating LLMs Trained on Code", Eq. 1)
    pass^k = P(all k samples are correct)            = C(c, k) / C(n, k)
             (Yao et al. 2024, "tau-bench", Sec. 3)

Both are unbiased estimates for k <= n; the metric is their mean over questions.
pass@k rewards an agent that *sometimes* succeeds; pass^k rewards one that succeeds
*reliably*. pass^1 == pass@1 == the expected accuracy of a single sample.

Example, n = 4, c = 2:   pass@2 = 1 - C(2,2)/C(4,2) = 1 - 1/6 = 0.833
                         pass^2 = C(2,2)/C(4,2)     = 1/6     = 0.167
"""

from __future__ import annotations

from collections import defaultdict
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    return comb(c, k) / comb(n, k)


def correct_counts(records: list[dict]) -> dict[str, tuple[int, int]]:
    """qid -> (n samples, c correct) from eval records (eval/evaluate.py)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        counts[str(r["qid"])][0] += 1
        counts[str(r["qid"])][1] += int(bool(r["correct"]))
    return {q: (n, c) for q, (n, c) in counts.items()}


def pass_metrics(records: list[dict], ks: list[int] | None = None) -> dict[str, float]:
    """{'pass@k': mean, 'pass^k': mean} for every k (default 1..n). Every question must
    have the same number of samples n."""
    counts = correct_counts(records)
    if not counts:
        return {}
    ns = {n for n, _ in counts.values()}
    if len(ns) != 1:
        raise ValueError(f"questions have different sample counts: {sorted(ns)}")
    n = ns.pop()
    ks = ks or list(range(1, n + 1))
    out = {}
    for k in ks:
        out[f"pass@{k}"] = sum(pass_at_k(n, c, k) for n, c in counts.values()) / len(counts)
        out[f"pass^{k}"] = sum(pass_hat_k(n, c, k) for n, c in counts.values()) / len(counts)
    return out
