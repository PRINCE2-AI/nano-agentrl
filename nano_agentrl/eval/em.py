"""SQuAD-style answer normalization and Exact Match against multiple gold answers.

Same normalization as SQuAD v1.1's official script and Search-R1's
verl/utils/reward_score/qa_em.py: lowercase, drop punctuation, drop articles
(a, an, the), collapse whitespace.
"""

from __future__ import annotations

import re
import string

_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """'The  Eiffel Tower!' -> 'eiffel tower'."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, golds: str | list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold answer, else 0.0."""
    if prediction is None:
        return 0.0
    if isinstance(golds, str):
        golds = [golds]
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in golds))
