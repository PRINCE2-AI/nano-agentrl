"""Eval harness: prompts, records, summary, results table, data split names, dry-run CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nano_agentrl.data import load_split, parse_search_split
from nano_agentrl.eval.evaluate import (
    build_prompt,
    evaluate_examples,
    summarize,
    update_results,
)
from nano_agentrl.generation import Generator
from nano_agentrl.retriever.client import TINY_CORPUS, InMemoryRetriever
from nano_agentrl.tiny import make_tiny_qa_dataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts/evaluate.py")
evaluate_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_script)


def test_parse_search_split():
    assert parse_search_split("nq_dev_500") == ("nq", 500)
    assert parse_search_split("hotpotqa_test") == ("hotpotqa", None)
    assert parse_search_split("2wikimultihopqa_dev_100") == ("2wikimultihopqa", 100)
    assert parse_search_split("gsm8k_test") is None


def test_load_split_subsample_is_fixed():
    a = load_split("arithmetic_test", n=5, seed=0)
    b = load_split("arithmetic_test", n=5, seed=0)
    assert a == b and len(a) == 5
    with pytest.raises(ValueError):
        load_split("no_such_split")


def test_build_prompt_modes():
    docs = [{"title": "Paris", "text": "Capital of France."}]
    assert "Doc 1(Title: Paris) Capital of France." in build_prompt("rag", "q?", docs)
    assert build_prompt("direct", " q? ").endswith("Question: q?\n")
    assert build_prompt("search", "q?").startswith("Answer the given question. You must conduct")
    with pytest.raises(ValueError):
        build_prompt("cot", "q")


class EchoGenerator:
    """Answers with the title of the first doc in the prompt (perfect reader), else 'unknown'."""

    def __init__(self, tok):
        self.tok = tok

    def generate(self, prompt_ids, n=1, max_new_tokens=500, temperature=0.0, stop_strings=None):
        from nano_agentrl.generation import GenerationOutput

        outs = []
        for i, ids in enumerate(prompt_ids):
            text = self.tok.decode(ids)
            ans = text.split("(Title: ")[1].split(")")[0] if "(Title: " in text else "unknown"
            reply = f"<answer>{ans}</answer>"
            for _ in range(n):
                outs.append(GenerationOutput(i, self.tok(reply)["input_ids"], reply, "stop"))
        return outs


def test_rag_beats_direct_with_perfect_reader(char_tokenizer):
    examples = [ex for ex in make_tiny_qa_dataset() if ex["golds"][0] in ("Paris", "Berlin")]
    examples = [
        {"question": "capital of France", "golds": ["Paris"]},
        {"question": "capital of Germany", "golds": ["Berlin"]},
    ]
    retriever = InMemoryRetriever(TINY_CORPUS)
    gen = EchoGenerator(char_tokenizer)
    direct = summarize(evaluate_examples(gen, char_tokenizer, examples, "direct"))
    rag = summarize(evaluate_examples(gen, char_tokenizer, examples, "rag", retriever, topk=1))
    assert direct["em"] == 0.0 and rag["em"] == 1.0
    assert rag["num_questions"] == 2 and rag["invalid_format_rate"] == 0.0


def test_search_mode_records_with_samples(tiny_model, char_tokenizer):
    examples = make_tiny_qa_dataset()[:3]
    records = evaluate_examples(
        Generator(tiny_model, char_tokenizer),
        char_tokenizer,
        examples,
        "search",
        InMemoryRetriever(TINY_CORPUS),
        n_samples=2,
        temperature=1.0,
        batch_size=2,
        max_turns=2,
        max_new_tokens=8,
        max_total_tokens=1000,  # tiny GPT-2 has 1024 positions
    )
    assert len(records) == 6
    assert [r["sample"] for r in records] == [0, 1] * 3
    assert {"prediction", "correct", "num_searches", "finish_reason", "text"} <= set(records[0])
    s = summarize(records)
    assert s["num_questions"] == 3 and s["num_samples"] == 6


def test_update_results_table(tmp_path):
    js, md = tmp_path / "r.json", tmp_path / "r.md"
    md.write_text("# Results\n\nIntro text.\n", encoding="utf-8")
    update_results(
        js, md, {"model": "m1", "mode": "rag", "split": "nq_dev_500", "em": 0.25, "n": 500}
    )
    update_results(
        js, md, {"model": "m1", "mode": "rag", "split": "hotpotqa_dev_500", "em": 0.5, "n": 500}
    )
    update_results(
        js, md, {"model": "m1", "mode": "rag", "split": "nq_dev_500", "em": 0.3, "n": 500}
    )
    text = md.read_text(encoding="utf-8")
    assert "Intro text." in text
    assert "| `m1` | rag | 50.0 (n=500) | 30.0 (n=500) |" in text  # upserted, splits sorted
    assert len(json.loads(js.read_text())) == 2


def test_evaluate_cli_dry_run(tmp_path):
    summaries = evaluate_script.main(
        [
            "--dry_run",
            "--mode",
            "search",
            "--split",
            "tinyqa_test",
            "--max_turns",
            "1",
            "--out_dir",
            str(tmp_path / "out"),
            "--results_json",
            str(tmp_path / "r.json"),
            "--results_md",
            str(tmp_path / "r.md"),
        ]
    )
    assert summaries["tinyqa_test"]["num_questions"] == 6
    assert (tmp_path / "out/tinyqa_test_search.jsonl").exists()
    # A tiny random model must never land in the published results table.
    assert not (tmp_path / "r.json").exists()
    assert not (tmp_path / "r.md").exists()


def test_evaluate_cli_records_protocol(tmp_path):
    summaries = evaluate_script.main(
        [
            "--dry_run",
            "--mode",
            "search",
            "--split",
            "tinyqa_test",
            "--max_turns",
            "1",
            "--max_new_tokens",
            "8",
            "--max_obs_tokens",
            "100",
            "--max_total_tokens",
            "900",
            "--topk",
            "2",
            "--out_dir",
            str(tmp_path / "out"),
            "--no_results_table",
        ]
    )
    protocol = summaries["tinyqa_test"]["protocol"]
    assert protocol["max_turns"] == 1
    assert protocol["max_new_tokens"] == 8
    assert protocol["max_obs_tokens"] == 100
    assert protocol["max_total_tokens"] == 900
    assert protocol["topk"] == 2
    assert protocol["dry_run"] is True
    saved = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert saved["tinyqa_test"]["protocol"] == protocol


def test_evaluate_cli_inherits_protocol_from_config(tmp_path):
    summaries = evaluate_script.main(
        [
            "--dry_run",
            "--mode",
            "search",
            "--config",
            str(ROOT / "configs/search_0.5b_t4.yaml"),
            "--split",
            "tinyqa_test",
            "--out_dir",
            str(tmp_path / "out"),
            "--no_results_table",
        ]
    )
    protocol = summaries["tinyqa_test"]["protocol"]
    assert protocol["max_turns"] == 2
    assert protocol["max_obs_tokens"] == 200  # dry-run cap for the tiny 1024-position model
    assert protocol["max_total_tokens"] == 1000
    assert protocol["max_new_tokens"] == 16  # dry-run cap wins over the config's 300
    assert protocol["config"].endswith("search_0.5b_t4.yaml")


def test_evaluate_cli_flags_override_config(tmp_path):
    summaries = evaluate_script.main(
        [
            "--dry_run",
            "--mode",
            "search",
            "--config",
            str(ROOT / "configs/search_0.5b_t4.yaml"),
            "--split",
            "tinyqa_test",
            "--max_turns",
            "1",
            "--out_dir",
            str(tmp_path / "out"),
            "--no_results_table",
        ]
    )
    assert summaries["tinyqa_test"]["protocol"]["max_turns"] == 1
