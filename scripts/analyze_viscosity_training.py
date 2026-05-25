"""Analyze viscosity training outputs with plots and worst-error tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "viscosity_training" / "sanity_cnn"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "index": int(row["index"]),
                "viscosity_index": int(row.get("viscosity_index", "-1") or -1),
                "action_index": int(row.get("action_index", "-1") or -1),
                "action_id": row.get("action_id", ""),
                "target_log10_mu": float(row["target_log10_mu"]),
                "pred_log10_mu": float(row["pred_log10_mu"]),
                "retrieval_log10_mu": float(row["retrieval_log10_mu"]),
                "abs_error_log10_mu": float(row["abs_error_log10_mu"]),
                "retrieval_abs_error_log10_mu": float(row["retrieval_abs_error_log10_mu"]),
            }
        )
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_worst_errors(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "split",
        "index",
        "run_id",
        "action_id",
        "target_log10_mu",
        "pred_log10_mu",
        "target_mu",
        "pred_mu",
        "abs_error_log10_mu",
        "multiplicative_error",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            target_mu = 10.0 ** row["target_log10_mu"]
            pred_mu = 10.0 ** row["pred_log10_mu"]
            writer.writerow(
                {
                    "rank": rank,
                    "split": row["split"],
                    "index": row["index"],
                    "run_id": row["run_id"],
                    "action_id": row.get("action_id", ""),
                    "target_log10_mu": f"{row['target_log10_mu']:.8f}",
                    "pred_log10_mu": f"{row['pred_log10_mu']:.8f}",
                    "target_mu": f"{target_mu:.8g}",
                    "pred_mu": f"{pred_mu:.8g}",
                    "abs_error_log10_mu": f"{row['abs_error_log10_mu']:.8f}",
                    "multiplicative_error": f"{10.0 ** row['abs_error_log10_mu']:.4f}",
                    "video_path": row["video_path"],
                }
            )


def _plot_predictions(path: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    colors = {"val": "#2b6cb0", "test": "#c05621"}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        x = [row["target_log10_mu"] for row in split_rows]
        y = [row["pred_log10_mu"] for row in split_rows]
        ax.scatter(x, y, s=34, alpha=0.85, label=split, color=colors[split], edgecolors="white", linewidths=0.5)
    all_values = [row["target_log10_mu"] for row in rows] + [row["pred_log10_mu"] for row in rows]
    lo = min(all_values) - 0.06
    hi = max(all_values) + 0.06
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0, linestyle="--", label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true log10(mu)")
    ax.set_ylabel("predicted log10(mu)")
    ax.set_title("Viscosity Prediction Sanity Baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    summary = (
        f"test MAE={metrics['test_mae_log10_mu']:.3f}\n"
        f"test x-error={metrics['test_typical_multiplicative_error']:.2f}x\n"
        f"rho={metrics['test_spearman']:.2f}"
    )
    ax.text(
        0.97,
        0.04,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_training_curve(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    val_mae = [row["val_mae_log10_mu"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ax1.plot(epochs, val_mae, color="#2b6cb0", marker="o", markersize=3, label="val MAE log10(mu)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val MAE log10(mu)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss, color="#718096", linewidth=1.5, label="train loss")
    ax2.set_ylabel("train loss", color="#4a5568")
    ax2.tick_params(axis="y", labelcolor="#4a5568")
    ax1.set_title("Sanity Baseline Training Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_error_by_viscosity(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    colors = {"val": "#2b6cb0", "test": "#c05621"}
    for split in ("val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        ax.scatter(
            [row["target_log10_mu"] for row in split_rows],
            [row["abs_error_log10_mu"] for row in split_rows],
            s=28,
            alpha=0.82,
            label=split,
            color=colors[split],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("true log10(mu)")
    ax.set_ylabel("absolute error log10(mu)")
    ax.set_title("Error by Viscosity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_predictions_by_action(path: Path, rows: list[dict[str, Any]]) -> None:
    action_ids = sorted({row.get("action_id", "") for row in rows if row.get("action_id", "")})
    if not action_ids:
        return
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#718096"]
    fig, ax = plt.subplots(figsize=(6.4, 5.3), dpi=150)
    for idx, action_id in enumerate(action_ids):
        action_rows = [row for row in rows if row.get("action_id", "") == action_id]
        ax.scatter(
            [row["target_log10_mu"] for row in action_rows],
            [row["pred_log10_mu"] for row in action_rows],
            s=26,
            alpha=0.82,
            label=action_id,
            color=palette[idx % len(palette)],
            edgecolors="white",
            linewidths=0.4,
        )
    all_values = [row["target_log10_mu"] for row in rows] + [row["pred_log10_mu"] for row in rows]
    lo = min(all_values) - 0.06
    hi = max(all_values) + 0.06
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0, linestyle="--", label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true log10(mu)")
    ax.set_ylabel("predicted log10(mu)")
    ax.set_title("Predictions by Action")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = sorted({row.get(key, "") for row in rows})
    output: list[dict[str, Any]] = []
    for group in groups:
        group_rows = [row for row in rows if row.get(key, "") == group]
        if not group_rows:
            continue
        errors = np.asarray([row["abs_error_log10_mu"] for row in group_rows], dtype=np.float64)
        output.append(
            {
                key: group,
                "count": int(len(group_rows)),
                "mae_log10_mu": float(errors.mean()),
                "typical_multiplicative_error": float(10.0 ** errors.mean()),
                "max_error_log10_mu": float(errors.max()),
            }
        )
    return output


def _viscosity_bin_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    min_value = min(row["target_log10_mu"] for row in rows)
    max_value = max(row["target_log10_mu"] for row in rows)
    edges = [min_value, -2.5, -2.0, max_value + 1.0e-9]
    labels = ["low", "middle", "high"]
    output: list[dict[str, Any]] = []
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        bin_rows = [row for row in rows if lo <= row["target_log10_mu"] < hi]
        if not bin_rows:
            continue
        errors = np.asarray([row["abs_error_log10_mu"] for row in bin_rows], dtype=np.float64)
        output.append(
            {
                "bin": label,
                "range_log10_mu": [float(lo), float(hi)],
                "count": int(len(bin_rows)),
                "mae_log10_mu": float(errors.mean()),
                "typical_multiplicative_error": float(10.0 ** errors.mean()),
                "max_error_log10_mu": float(errors.max()),
            }
        )
    return output


def _decision(metrics: dict[str, Any]) -> str:
    test_mae = float(metrics["test_mae_log10_mu"])
    if test_mae < 0.15:
        return "strong_signal"
    if test_mae <= 0.35:
        return "usable_but_needs_model_or_preprocessing_work"
    return "weak_signal_or_dataset_issue"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    metrics = _read_json(output_dir / "final_metrics.json")
    history = _read_jsonl(output_dir / "metrics.jsonl")
    rows = _read_predictions(output_dir / "predictions_val.csv") + _read_predictions(
        output_dir / "predictions_test.csv"
    )
    worst_rows = sorted(rows, key=lambda row: row["abs_error_log10_mu"], reverse=True)[: args.top_k]

    prediction_plot = output_dir / "prediction_scatter.png"
    training_plot = output_dir / "training_curve.png"
    error_by_viscosity_plot = output_dir / "error_by_viscosity.png"
    action_plot = output_dir / "prediction_scatter_by_action.png"
    worst_errors = output_dir / "worst_errors.csv"
    summary_path = output_dir / "analysis_summary.json"
    _plot_predictions(prediction_plot, rows, metrics)
    _plot_training_curve(training_plot, history)
    _plot_error_by_viscosity(error_by_viscosity_plot, rows)
    _plot_predictions_by_action(action_plot, rows)
    _write_worst_errors(worst_errors, worst_rows)
    action_metrics = _group_metrics(rows, "action_id")
    viscosity_bin_metrics = _viscosity_bin_metrics(rows)

    summary = {
        "decision": _decision(metrics),
        "metrics": metrics,
        "action_metrics": action_metrics,
        "viscosity_bin_metrics": viscosity_bin_metrics,
        "prediction_plot": str(prediction_plot),
        "training_plot": str(training_plot) if history else None,
        "error_by_viscosity_plot": str(error_by_viscosity_plot),
        "prediction_by_action_plot": str(action_plot) if action_plot.exists() else None,
        "worst_errors": str(worst_errors),
        "worst_error_log10_mu": worst_rows[0]["abs_error_log10_mu"] if worst_rows else math.nan,
        "worst_error_multiplicative": 10.0 ** worst_rows[0]["abs_error_log10_mu"] if worst_rows else math.nan,
        "num_predictions": len(rows),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
