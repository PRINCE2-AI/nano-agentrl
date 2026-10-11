"""Dataset loading. Every split becomes a list of {"question": str, "golds": list[str]}.

Split names used in configs and on the command line:
    gsm8k_train, gsm8k_test      openai/gsm8k ("main")
    search_train                 PeterJinGo/nq_hotpotqa_train, split "train" (NQ + HotpotQA)
    <source>_test                every question of one source in the Search-R1 eval file
                                 (PeterJinGo/nq_hotpotqa_train, split "test")
    <source>_dev_<N>             fixed-seed (seed 0) subset of N questions of <source>_test
        sources: nq (NQ test, 3610), hotpotqa (HotpotQA dev, 7405), triviaqa, popqa,
                 2wikimultihopqa, musique, bamboogle
    arithmetic_train/_test       synthetic, for --dry_run and tests (nano_agentrl.tiny)
    tinyqa_train/_test           6 questions over the tiny in-memory corpus (search dry runs)
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from functools import lru_cache

from nano_agentrl.rewards import extract_gsm8k_gold
from nano_agentrl.tiny import make_arithmetic_dataset, make_tiny_qa_dataset

Example = dict  # {"question": str, "golds": list[str], ...optional extra fields}

SEARCH_R1_DATASET = "PeterJinGo/nq_hotpotqa_train"
SEARCH_SOURCES = ("nq", "hotpotqa", "triviaqa", "popqa", "2wikimultihopqa", "musique", "bamboogle")
_SEARCH_SPLIT_RE = re.compile(rf"^({'|'.join(SEARCH_SOURCES)})_(test|dev_(\d+))$")


def _load_gsm8k(split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "golds": [extract_gsm8k_gold(r["answer"])]} for r in ds]


@lru_cache(maxsize=2)
def _load_search_r1(split: str) -> tuple[Example, ...]:
    """All rows of the Search-R1 parquet ("train" or "test"), cached per process.

    Downloads only that one file (test 70 MB, train 356 MB) into the HF cache.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SEARCH_R1_DATASET, f"{split}.parquet", repo_type="dataset")
    columns = ["id", "question", "golden_answers", "data_source"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    return tuple(
        {
            "id": r["id"],
            "question": r["question"],
            "golds": list(r["golden_answers"]),
            "data_source": r["data_source"],
        }
        for r in rows
    )


def parse_search_split(name: str) -> tuple[str, int | None] | None:
    """'nq_dev_500' -> ('nq', 500); 'hotpotqa_test' -> ('hotpotqa', None); else None."""
    m = _SEARCH_SPLIT_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(3)) if m.group(3) else None


SPLITS: dict[str, Callable[[], list[Example]]] = {
    "gsm8k_train": lambda: _load_gsm8k("train"),
    "gsm8k_test": lambda: _load_gsm8k("test"),
    "search_train": lambda: list(_load_search_r1("train")),
    "arithmetic_train": lambda: make_arithmetic_dataset(256, seed=0),
    "arithmetic_test": lambda: make_arithmetic_dataset(64, seed=1),
    "tinyqa_train": make_tiny_qa_dataset,
    "tinyqa_test": make_tiny_qa_dataset,
}

# Config `train_dataset` values that map to a split name.
TRAIN_DATASETS = {
    "openai/gsm8k": "gsm8k_train",
    SEARCH_R1_DATASET: "search_train",
    "arithmetic": "arithmetic_train",
    "tinyqa": "tinyqa_train",
}


def subsample(data: list[Example], n: int | None, seed: int) -> list[Example]:
    """Fixed-seed subset of size n (the whole list if n is None or >= len)."""
    if n is None or n >= len(data):
        return data
    return random.Random(seed).sample(data, n)


def load_split(name: str, n: int | None = None, seed: int = 0) -> list[Example]:
    """Examples of split `name`, optionally a fixed-seed subset of size n."""
    search = parse_search_split(name)
    if search is not None:
        source, dev_size = search
        rows = [r for r in _load_search_r1("test") if r["data_source"] == source]
        rows = subsample(rows, dev_size, seed=0)  # the dev subset is always the same
        return subsample(rows, n, seed)
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; known: {sorted(SPLITS)} or <source>_dev_<N>")
    return subsample(SPLITS[name](), n, seed)


def load_train_dataset(train_dataset: str, n: int | None = None, seed: int = 0) -> list[Example]:
    return load_split(TRAIN_DATASETS.get(train_dataset, train_dataset), n, seed)
