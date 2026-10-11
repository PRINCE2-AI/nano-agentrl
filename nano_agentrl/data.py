"""Dataset loading. Every split becomes a list of {"question": str, "golds": list[str]}.

Split names used in configs and on the command line:
    gsm8k_train, gsm8k_test     openai/gsm8k ("main")
    arithmetic_train/_test      synthetic, for --dry_run and tests (nano_agentrl.tiny)
Search-task splits (nq / hotpotqa) are added in nano_agentrl/data.py by milestone 2.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from nano_agentrl.rewards import extract_gsm8k_gold
from nano_agentrl.tiny import make_arithmetic_dataset

Example = dict  # {"question": str, "golds": list[str], ...optional extra fields}


def _load_gsm8k(split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "golds": [extract_gsm8k_gold(r["answer"])]} for r in ds]


SPLITS: dict[str, Callable[[], list[Example]]] = {
    "gsm8k_train": lambda: _load_gsm8k("train"),
    "gsm8k_test": lambda: _load_gsm8k("test"),
    "arithmetic_train": lambda: make_arithmetic_dataset(256, seed=0),
    "arithmetic_test": lambda: make_arithmetic_dataset(64, seed=1),
}

# Config `train_dataset` values that map to a split name.
TRAIN_DATASETS = {"openai/gsm8k": "gsm8k_train", "arithmetic": "arithmetic_train"}


def subsample(data: list[Example], n: int | None, seed: int) -> list[Example]:
    """Fixed-seed subset of size n (the whole list if n is None or >= len)."""
    if n is None or n >= len(data):
        return data
    return random.Random(seed).sample(data, n)


def load_split(name: str, n: int | None = None, seed: int = 0) -> list[Example]:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; known: {sorted(SPLITS)}")
    return subsample(SPLITS[name](), n, seed)


def load_train_dataset(train_dataset: str, n: int | None = None, seed: int = 0) -> list[Example]:
    return load_split(TRAIN_DATASETS.get(train_dataset, train_dataset), n, seed)
