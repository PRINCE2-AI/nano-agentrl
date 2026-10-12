"""Run a model or checkpoint over a dataset split and report EM and reliability metrics.

Three modes (the baselines of Search-R1 Table 2 that need no extra training):
    direct  answer from parametric knowledge only, one generation
    rag     retrieve top-k docs for the question once, put them in the prompt, one generation
    search  the Search-R1 multi-turn loop (rollout.py): the model decides when and what to
            search. On an instruct model without RL this is the ReAct-style prompting
            baseline; on a Search-R1 / our checkpoint it is the trained agent.

Every (question, sample) becomes one record:
    {"qid", "sample", "question", "golds", "prediction", "correct", "num_searches",
     "num_invalid", "num_turns", "finish_reason", "valid_format", "text"}
With n_samples > 1 (temperature > 0) the records feed pass@k / pass^k (eval/pass_k.py).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from nano_agentrl.prompts import build_chat_prompt, format_docs, search_r1_prompt
from nano_agentrl.rewards import compute_reward, extract_answer, has_valid_format
from nano_agentrl.rollout import Trajectory, run_search_rollouts

MODES = ("direct", "rag", "search")

DIRECT_TEMPLATE = (
    "Answer the given question. Provide the answer inside <answer> and </answer>, without "
    "detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)
RAG_TEMPLATE = (
    "Answer the given question using the documents below. Provide the answer inside <answer> "
    "and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n"
    "Documents:\n{docs}\nQuestion: {question}\n"
)


def build_prompt(mode: str, question: str, docs: list[dict] | None = None) -> str:
    if mode == "direct":
        return DIRECT_TEMPLATE.format(question=question.strip())
    if mode == "rag":
        return RAG_TEMPLATE.format(docs=format_docs(docs or []), question=question.strip())
    if mode == "search":
        return search_r1_prompt(question)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def generate_trajectories(
    generator: Any,
    tokenizer: Any,
    examples: list[dict],
    mode: str,
    retriever: Any = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_turns: int = 4,
    max_new_tokens: int = 500,
    topk: int = 3,
    max_obs_tokens: int = 500,
    max_total_tokens: int = 4096,
) -> list[Trajectory]:
    """n_samples trajectories per example, example-major."""
    if mode in ("rag", "search") and retriever is None:
        raise ValueError(f"mode {mode!r} needs a retriever")
    docs = retriever.search([ex["question"] for ex in examples], topk) if mode == "rag" else None
    prompt_ids = []
    for i, ex in enumerate(examples):
        text = build_chat_prompt(tokenizer, build_prompt(mode, ex["question"], docs and docs[i]))
        prompt_ids.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    golds = [ex["golds"] for ex in examples]

    if mode == "search":
        return run_search_rollouts(
            generator,
            tokenizer,
            retriever,
            prompt_ids,
            golds,
            n=n_samples,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            topk=topk,
            max_obs_tokens=max_obs_tokens,
            max_total_tokens=max_total_tokens,
        )
    outputs = generator.generate(
        prompt_ids,
        n=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_strings=["</answer>"],
    )
    return [
        Trajectory(
            prompt_index=o.prompt_index,
            prompt_ids=prompt_ids[o.prompt_index],
            response_ids=o.token_ids,
            response_mask=[1] * len(o.token_ids),
            text=o.text,
            golds=golds[o.prompt_index],
            finish_reason=o.finish_reason,
            num_turns=1,
        )
        for o in outputs
    ]


def to_records(
    trajectories: list[Trajectory], examples: list[dict], n_samples: int, offset: int = 0
) -> list[dict]:
    """One record per trajectory. qid falls back to the global example index (offset + i)."""
    records = []
    for j, t in enumerate(trajectories):
        ex = examples[t.prompt_index]
        records.append(
            {
                "qid": ex.get("id", offset + t.prompt_index),
                "sample": j % n_samples,
                "question": ex["question"],
                "golds": t.golds,
                "prediction": extract_answer(t.scored_text()),
                "correct": compute_reward(t.scored_text(), t.golds, "search") == 1.0,
                "num_searches": t.num_searches,
                "num_invalid": t.num_invalid,
                "num_turns": t.num_turns,
                "finish_reason": t.finish_reason,
                "valid_format": has_valid_format(t.scored_text(), "search"),
                "text": t.text,
            }
        )
    return records


def evaluate_examples(
    generator: Any,
    tokenizer: Any,
    examples: list[dict],
    mode: str,
    retriever: Any = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    batch_size: int = 32,
    **rollout_kwargs: Any,
) -> list[dict]:
    """Records for every (example, sample), in batches of `batch_size` examples."""
    records: list[dict] = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s : s + batch_size]
        trajs = generate_trajectories(
            generator, tokenizer, chunk, mode, retriever, n_samples, temperature, **rollout_kwargs
        )
        records += to_records(trajs, chunk, n_samples, offset=s)
    return records


def summarize(records: list[dict]) -> dict[str, Any]:
    """EM (mean over all samples), behaviour stats and finish-reason counts."""
    return {
        "em": float(np.mean([r["correct"] for r in records])) if records else 0.0,
        "num_questions": len({r["qid"] for r in records}),
        "num_samples": len(records),
        "num_searches": float(np.mean([r["num_searches"] for r in records])) if records else 0.0,
        "invalid_format_rate": (
            float(np.mean([not r["valid_format"] for r in records])) if records else 0.0
        ),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
    }


def save_records(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- results table


RESULTS_START, RESULTS_END = "<!-- results:start -->", "<!-- results:end -->"


def update_results(json_path: str | Path, md_path: str | Path, row: dict[str, Any]) -> None:
    """Upsert `row` ({"model", "mode", "split", "em", "n", ...}) into results.json and
    re-render the table between the markers in results.md (one column per split)."""
    json_path, md_path = Path(json_path), Path(md_path)
    rows = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
    key = (row["model"], row["mode"], row["split"])
    rows = [r for r in rows if (r["model"], r["mode"], r["split"]) != key] + [row]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    splits = sorted({r["split"] for r in rows})
    table = {}
    for r in rows:
        table.setdefault((r["model"], r["mode"]), {})[r["split"]] = r
    lines = [
        "| Model | Mode | " + " | ".join(splits) + " |",
        "|---|---|" + "---|" * len(splits),
    ]
    for (model, mode), cells in table.items():
        vals = [
            f"{cells[s]['em'] * 100:.1f} (n={cells[s]['n']})" if s in cells else "–" for s in splits
        ]
        lines.append(f"| `{model}` | {mode} | " + " | ".join(vals) + " |")
    block = f"{RESULTS_START}\n" + "\n".join(lines) + f"\n{RESULTS_END}"

    text = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Results\n\n"
    if RESULTS_START in text:
        head, _, rest = text.partition(RESULTS_START)
        _, _, tail = rest.partition(RESULTS_END)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    md_path.write_text(text, encoding="utf-8")
