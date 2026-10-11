"""scripts/render_trajectory.py: masked tokens grey, tags coloured, HTML escaped."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("render", ROOT / "scripts/render_trajectory.py")
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def test_masked_tokens_are_grey_and_tags_coloured():
    tokens = ["<search>", "q", "</search>", "<information>", "doc", "</information>", "<answer>"]
    mask = [1, 1, 1, 0, 0, 0, 1]
    out = render.render_tokens(tokens, mask)
    assert out.count('class="env"') == 1
    env = out.split('class="env"')[1].split("</span></span>")[0]
    assert "doc" in env and "q" not in env
    assert 'class="tag t-search"' in out and 'class="tag t-answer"' in out


def test_html_is_escaped():
    out = render.render_tokens(["<script>alert(1)</script>"], [1])
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_render_text_greys_information_blocks():
    out = render.render_text("a<information>doc</information>b")
    assert '<span class="env">' in out and out.startswith("a")


def test_cli_writes_html_from_trainer_samples(tmp_path):
    row = {
        "prompt_text": "Q?",
        "response_tokens": ["<answer>", "Paris", "</answer>"],
        "response_mask": [1, 1, 1],
        "reward": 1.0,
        "num_searches": 0,
        "golds": ["Paris"],
        "finish_reason": "answer",
    }
    eval_row = {"question": "Q2?", "text": "<answer>x</answer>", "correct": False, "golds": ["y"]}
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(eval_row) + "\n", encoding="utf-8")
    out = render.main([str(path)])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "2 trajectories" in text and "reward 1" in text and "reward 0" in text
