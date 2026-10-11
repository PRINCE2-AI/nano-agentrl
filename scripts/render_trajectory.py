"""Render a saved rollout batch (jsonl) as a self-contained HTML file showing masked tokens.

Input: a jsonl written by the trainer (runs/<run>/samples/step_*.jsonl, one trajectory per
line with response_tokens + response_mask) or by scripts/evaluate.py (records with `text`
only; there the <information> blocks are greyed out by pattern instead of by mask).

    python scripts/render_trajectory.py runs/search_0.5b/samples/step_000100_nq_dev_500.jsonl
    # -> writes the same path with .html; open it in a browser

Policy tokens are shown normally, masked (retrieved / environment) tokens in grey, and the
tags <think> <search> <information> <answer> are coloured. Each trajectory shows its
reward, number of searches and finish reason.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

TAG_RE = re.compile(r"(</?(?:think|search|information|answer)>)")
INFO_RE = re.compile(r"(<information>.*?</information>)", re.DOTALL)

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#8a8f98; --card:#f6f7f9; --border:#e3e5e8;
        --env:#9aa0a6; --env-bg:#eef0f2; --think:#7b4fd6; --search:#0a7cc4;
        --info:#6b7280; --answer:#1a8a3a; --bad:#c62828; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --fg:#e8e8ea; --muted:#9aa0a6; --card:#1f2125; --border:#2d3036;
          --env:#7d838c; --env-bg:#262a30; --think:#b495ff; --search:#5bb8f5;
          --info:#9aa0a6; --answer:#5fd17f; --bad:#ff7a7a; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 20px; font-size: 13px; }
.legend span { margin-right: 14px; }
.traj { background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:14px 16px; margin: 0 0 18px; }
.meta { display:flex; flex-wrap:wrap; gap:8px 16px; font-size:13px; margin-bottom:10px; }
.pill { padding:1px 8px; border-radius:999px; border:1px solid var(--border); }
.ok { color: var(--answer); border-color: var(--answer); }
.no { color: var(--bad); border-color: var(--bad); }
details { margin-bottom: 8px; color: var(--muted); font-size: 13px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0;
      font: 13px/1.55 ui-monospace, "Cascadia Code", Consolas, monospace; }
.env { color: var(--env); background: var(--env-bg); }
.tag { font-weight: 700; }
.t-think { color: var(--think); } .t-search { color: var(--search); }
.t-information { color: var(--info); } .t-answer { color: var(--answer); }
"""


def _colour_tags(escaped: str) -> str:
    """Wrap already-escaped tag text like &lt;search&gt; in coloured spans."""

    def sub(m: re.Match) -> str:
        return f'<span class="tag t-{m.group(2)}">{m.group(1)}</span>'

    return re.sub(r"(&lt;/?(think|search|information|answer)&gt;)", sub, escaped)


def render_tokens(tokens: list[str], mask: list[int]) -> str:
    """Group consecutive tokens with the same mask; masked groups get class 'env'."""
    parts, buf, cur = [], [], None
    for tok, m in zip(tokens, mask, strict=True):
        if cur is not None and m != cur:
            parts.append((cur, "".join(buf)))
            buf = []
        buf.append(tok)
        cur = m
    if buf:
        parts.append((cur, "".join(buf)))
    out = []
    for m, text in parts:
        body = _colour_tags(html.escape(text))
        out.append(
            body if m == 1 else f'<span class="env" title="masked (loss_mask=0)">{body}</span>'
        )
    return "".join(out)


def render_text(text: str) -> str:
    """No token mask available: grey out <information> blocks by pattern."""
    out = []
    for i, chunk in enumerate(INFO_RE.split(text)):
        body = _colour_tags(html.escape(chunk))
        out.append(f'<span class="env">{body}</span>' if i % 2 else body)
    return "".join(out)


def render_one(row: dict, index: int) -> str:
    reward = row.get("reward", 1.0 if row.get("correct") else 0.0)
    ok = reward >= 1.0
    if "response_tokens" in row and "response_mask" in row:
        body = render_tokens(row["response_tokens"], row["response_mask"])
        n_policy = sum(row["response_mask"])
        n_env = len(row["response_mask"]) - n_policy
        counts = f"<span>policy tokens {n_policy} · masked {n_env}</span>"
    else:
        body = render_text(row.get("text", ""))
        counts = ""
    prompt = row.get("prompt_text") or row.get("question", "")
    golds = ", ".join(row.get("golds", []))
    return f"""
<section class="traj">
  <div class="meta">
    <strong>#{index}</strong>
    <span class="pill {"ok" if ok else "no"}">reward {reward:g}</span>
    <span>searches {row.get("num_searches", 0)}</span>
    <span>invalid {row.get("num_invalid", 0)}</span>
    <span>finish {html.escape(str(row.get("finish_reason", "")))}</span>
    <span>gold: {html.escape(golds)}</span>
    {counts}
  </div>
  <details><summary>prompt</summary><pre>{html.escape(prompt)}</pre></details>
  <pre>{body}</pre>
</section>"""


def render_html(rows: list[dict], title: str) -> str:
    cards = "".join(render_one(r, i) for i, r in enumerate(rows))
    mean_r = sum(r.get("reward", float(bool(r.get("correct")))) for r in rows) / max(len(rows), 1)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(rows)} trajectories · mean reward {mean_r:.3f}</p>
<p class="sub legend"><span>normal = policy tokens (trained on)</span>
<span class="env">grey = masked environment tokens</span>
<span class="tag t-think">&lt;think&gt;</span><span class="tag t-search">&lt;search&gt;</span>
<span class="tag t-information">&lt;information&gt;</span>
<span class="tag t-answer">&lt;answer&gt;</span></p>
{cards}
</main></body></html>
"""


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Render trajectories (jsonl) to HTML")
    parser.add_argument("path", help="jsonl of trajectories")
    parser.add_argument("--out", default=None, help="output .html (default: next to input)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    path = Path(args.path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = rows[: args.limit] if args.limit else rows
    out = Path(args.out) if args.out else path.with_suffix(".html")
    out.write_text(render_html(rows, path.stem), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
