"""scripts/plot.py writes every figure from synthetic metrics and a synthetic sweep CSV."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot", ROOT / "scripts/plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "search_0.5b"
    run.mkdir()
    rows = []
    for s in range(1, 41):
        rows.append({"step": s, "train/reward_mean": s / 40, "train/num_searches": 1 + s / 20})
        if s % 10 == 0:
            rows.append({"step": s, "eval/em_nq": 0.1 + s / 100, "eval/em_hotpotqa": s / 200})
    metrics = run / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    wandb_csv = tmp_path / "wandb.csv"
    with open(wandb_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Step", "run-x - eval/em_nq", "run-x - train/num_searches"])
        for s in (10, 20, 30):
            w.writerow([s, s / 100, ""])

    rel = tmp_path / "reliability.csv"
    fields = (
        ["step", "split", "pass_at_1", "pass_at_8"]
        + [f"pass_hat_{k}" for k in (1, 2, 4, 8)]
        + [f"fail_{lab}" for lab in plot.FAILURE_LABELS]
    )
    with open(rel, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, step in enumerate((50, 100, 150)):
            acc = 0.2 + 0.1 * i
            row = {
                "step": step,
                "split": "nq_dev_500",
                "pass_at_1": acc,
                "pass_at_8": acc * 2.5,
                "pass_hat_1": acc,
                "pass_hat_2": acc * 0.8,
                "pass_hat_4": acc * 0.6,
                "pass_hat_8": acc * 0.4,
            }
            row.update({f"fail_{lab}": 1 / 6 for lab in plot.FAILURE_LABELS})
            w.writerow(row)
    return metrics, wandb_csv, rel


def test_all_figures_written(tmp_path):
    metrics, wandb_csv, rel = write_inputs(tmp_path)
    out = plot.main(
        [
            "--metrics",
            str(metrics),
            str(wandb_csv),
            "--reliability",
            str(rel),
            "--out_dir",
            str(tmp_path / "figs"),
        ]
    )
    names = {p.stem for p in out}
    assert names == {
        "fig0_capability_vs_consistency",
        "fig1_em_vs_step",
        "fig2_pass_hat_k",
        "fig3_acc_vs_reliability",
        "fig4_searches",
        "fig5_failure_taxonomy",
        "fig6_training_dynamics",
    }
    assert all(p.stat().st_size > 1000 for p in out)


def test_wandb_csv_loader(tmp_path):
    _, wandb_csv, _ = write_inputs(tmp_path)
    rows = plot.load_metrics(wandb_csv)
    assert rows[0] == {"step": 10, "eval/em_nq": 0.1}


def test_missing_inputs_are_skipped(tmp_path):
    assert plot.main(["--out_dir", str(tmp_path / "figs")]) == []
