"""Produce publication figures from training metrics and reliability-sweep CSVs.

    python scripts/plot.py --metrics runs/search_3b/metrics.jsonl \
        --reliability runs/search_3b/reliability.csv --out_dir docs/figures

Inputs:
    --metrics      one or more metrics.jsonl (from the trainer) or W&B CSV exports
                   (a step column + metric columns). Label = parent folder name, or
                   pass --labels.
    --reliability  reliability.csv from scripts/reliability_sweep.py
Figures (each saved as .png and .pdf):
    1. fig1_em_vs_step          eval EM per split vs step (one line per run x split)
    2. fig2_pass_hat_k          pass^k for k = 1, 2, 4, 8 vs step (one blue ramp: k is ordered)
    3. fig3_acc_vs_reliability  pass@1 (accuracy) vs pass^k (reliability), path in step order
    4. fig4_searches            searches per question vs step (train, smoothed)
    5. fig5_failure_taxonomy    failure-label shares per checkpoint (stacked bars)

Colour follows the dataviz reference palette: categorical slots in fixed order, a single
validated ordinal ramp for k, one y-axis per figure, hairline grid, direct labels plus a
legend (several categorical colours sit below 3:1 contrast, so they never carry identity
alone).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (light mode): categorical slots in fixed order, ink, chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
K_RAMP = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}  # validated ordinal ramp
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
FAILURE_LABELS = [
    "bad_query",
    "ignored_relevant_doc",
    "hallucinated_answer",
    "over_searching",
    "format_error",
    "other",
]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.dpi": 200,
    }
)


# ---------------------------------------------------------------- loading


def load_metrics(path: str | Path) -> list[dict]:
    """metrics.jsonl rows, or a W&B CSV export normalized to {'step': .., metric: ..}."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            step_key = next(k for k in ("step", "_step", "Step") if k in r)
            row = {"step": int(float(r[step_key]))}
            for k, v in r.items():
                if k == step_key or v in ("", None):
                    continue
                key = k.split(" - ")[-1]  # W&B: "<run name> - eval/em_nq"
                try:
                    row[key] = float(v)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted((r["step"], r[key]) for r in rows if key in r and r[key] is not None)
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts, strict=True)
    return np.array(x), np.array(y, dtype=float)


def load_reliability(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "split":
                r[k] = float(v) if v not in ("", None) else np.nan
    return rows


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window or window <= 1:
        return y
    kernel = np.ones(window) / window
    pad = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


# ---------------------------------------------------------------- helpers


def end_label(ax, x, y, text, color) -> None:
    """Direct label at the last point of a line, in secondary ink (never the series colour)."""
    if len(x):
        ax.annotate(
            text,
            (x[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
        ax.plot([x[-1]], [y[-1]], "o", ms=4, color=color, mec=SURFACE, mew=1.5)


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def percent_axis(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


# ---------------------------------------------------------------- figures


def fig_em_vs_step(runs: dict[str, list[dict]], out_dir: Path) -> list[Path] | None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    slot = 0
    for label, rows in runs.items():
        keys = sorted({k for r in rows for k in r if k.startswith("eval/em_")})
        for key in keys:
            x, y = series(rows, key)
            if not len(x):
                continue
            name = f"{label} · {key.removeprefix('eval/em_')}" if len(runs) > 1 else key[8:]
            color = SERIES[slot % len(SERIES)]
            ax.plot(x, y, color=color, marker="o", ms=3.5, mec=SURFACE, mew=1, label=name)
            end_label(ax, x, y, name, color)
            slot += 1
    if slot == 0:
        plt.close(fig)
        return None
    ax.set_title("Exact match vs training step")
    ax.set_xlabel("training step")
    ax.set_ylabel("EM (greedy)")
    percent_axis(ax)
    ax.set_ylim(bottom=0)
    if slot > 1:
        ax.legend(loc="lower right")
    return save(fig, out_dir, "fig1_em_vs_step")


def fig_pass_hat_k(rel: list[dict], out_dir: Path, split: str) -> list[Path] | None:
    rows = sorted((r for r in rel if r["split"] == split), key=lambda r: r["step"])
    ks = [k for k in K_RAMP if f"pass_hat_{k}" in (rows[0] if rows else {})]
    if not rows or not ks:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    steps = np.array([r["step"] for r in rows])
    for k in ks:
        y = np.array([r[f"pass_hat_{k}"] for r in rows])
        ax.plot(steps, y, color=K_RAMP[k], marker="o", ms=3.5, mec=SURFACE, mew=1, label=f"k = {k}")
        end_label(ax, steps, y, f"pass^{k}", K_RAMP[k])
    ax.set_title(f"Reliability: pass^k vs training step ({split})")
    ax.set_xlabel("training step")
    ax.set_ylabel("pass^k  (all k samples correct)")
    percent_axis(ax)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=len(ks))
    return save(fig, out_dir, "fig2_pass_hat_k")


def fig_acc_vs_reliability(rel: list[dict], out_dir: Path, split: str) -> list[Path] | None:
    rows = sorted((r for r in rel if r["split"] == split), key=lambda r: r["step"])
    kmax = max((k for k in K_RAMP if rows and f"pass_hat_{k}" in rows[0]), default=None)
    if not rows or kmax is None:
        return None
    x = np.array([r["pass_at_1"] for r in rows])
    y = np.array([r[f"pass_hat_{kmax}"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    lim = max(0.05, float(np.nanmax([x.max(), y.max()])) * 1.1)
    ax.plot([0, lim], [0, lim], color=AXIS, lw=0.8, ls="--")
    ax.annotate(
        f"pass^{kmax} = pass@1 (perfectly consistent)",
        (lim * 0.98, lim * 0.98),
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
        rotation=0,
    )
    ax.plot(x, y, color=SERIES[0], lw=1.2, alpha=0.6)
    ax.plot(x, y, "o", ms=5, color=SERIES[0], mec=SURFACE, mew=1.5)
    for i in (0, len(rows) - 1):
        ax.annotate(
            f"step {int(rows[i]['step'])}",
            (x[i], y[i]),
            xytext=(5, -10 if i else 5),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"Accuracy vs reliability ({split})")
    ax.set_xlabel("accuracy: pass@1")
    ax.set_ylabel(f"reliability: pass^{kmax}")
    percent_axis(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    return save(fig, out_dir, "fig3_acc_vs_reliability")


def fig_searches(runs: dict[str, list[dict]], out_dir: Path, window: int) -> list[Path] | None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    drawn = 0
    for i, (label, rows) in enumerate(runs.items()):
        x, y = series(rows, "train/num_searches")
        if not len(x):
            continue
        color = SERIES[i % len(SERIES)]
        ax.plot(x, y, color=color, lw=0.8, alpha=0.25)
        ys = smooth(y, window)
        ax.plot(x, ys, color=color, label=label)
        if len(runs) > 1:
            end_label(ax, x, ys, label, color)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_title("Searches per question during training")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"searches / trajectory ({window}-step mean)")
    ax.set_ylim(bottom=0)
    if drawn > 1:
        ax.legend(loc="upper left")
    return save(fig, out_dir, "fig4_searches")


def fig_failure_taxonomy(rel: list[dict], out_dir: Path, split: str) -> list[Path] | None:
    rows = sorted((r for r in rel if r["split"] == split), key=lambda r: r["step"])
    if not rows or not any(r.get(f"fail_{lab}", 0) for r in rows for lab in FAILURE_LABELS):
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    idx = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for i, lab in enumerate(FAILURE_LABELS):
        share = np.array([r.get(f"fail_{lab}", 0.0) or 0.0 for r in rows])
        ax.bar(
            idx,
            share,
            bottom=bottom,
            width=0.7,
            color=SERIES[i],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=lab.replace("_", " "),
        )
        bottom += share
    ax.set_xticks(idx, [str(int(r["step"])) for r in rows])
    ax.set_title(f"Why does the agent fail? ({split})")
    ax.set_xlabel("checkpoint (training step)")
    ax.set_ylabel("share of failed trajectories")
    percent_axis(ax)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    handles, names = ax.get_legend_handles_labels()  # top of legend = top of stack
    ax.legend(handles[::-1], names[::-1], loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return save(fig, out_dir, "fig5_failure_taxonomy")


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Paper figures from metrics and sweep CSVs")
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--reliability", default=None)
    parser.add_argument("--split", default=None, help="split for figs 2, 3, 5 (default: first)")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out_dir", default="docs/figures")
    args = parser.parse_args(argv)

    labels = args.labels or [Path(p).parent.name or Path(p).stem for p in args.metrics]
    runs = {lab: load_metrics(p) for lab, p in zip(labels, args.metrics, strict=True)}
    out_dir = Path(args.out_dir)
    written: list[Path] = []
    for result in (fig_em_vs_step(runs, out_dir), fig_searches(runs, out_dir, args.smooth)):
        written += result or []
    if args.reliability:
        rel = load_reliability(args.reliability)
        split = args.split or rel[0]["split"]
        for fn in (fig_pass_hat_k, fig_acc_vs_reliability, fig_failure_taxonomy):
            written += fn(rel, out_dir, split) or []
    for p in written:
        print(f"wrote {p}")
    return written


if __name__ == "__main__":
    main()
