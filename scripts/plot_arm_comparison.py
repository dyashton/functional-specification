#!/usr/bin/env python3
"""Plot matched Arm A versus Arm B G1-G5 evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Arm A": "#0f766e", "Arm B": "#6b7280"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _metric_values(summary: dict, metric: str) -> float:
    if metric in summary["headline"]:
        return float(summary["headline"][metric])
    rows = summary.get("G1_G2", [])
    return _mean_finite([float(row.get(metric, float("nan"))) for row in rows])


def _bar_panel(
    ax: plt.Axes,
    labels: list[str],
    arm_a: list[float],
    arm_b: list[float],
    *,
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    x = np.arange(len(labels))
    width = 0.36
    bars_a = ax.bar(x - width / 2, arm_a, width, label="Arm A", color=COLORS["Arm A"])
    bars_b = ax.bar(x + width / 2, arm_b, width, label="Arm B", color=COLORS["Arm B"])
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if lower_is_better:
        ax.text(
            0.98,
            0.94,
            "lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#6b7280",
        )
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#374151",
                )


def plot_comparison(comparison_dir: Path, out: Path) -> None:
    arm_a = _load(comparison_dir / "arm_a" / "gen_eval_summary.json")
    arm_b = _load(comparison_dir / "arm_b" / "gen_eval_summary.json")

    higher_labels = ["G1 match", "G2 entropy", "G3 confidence", "G4 LogP"]
    higher_metrics = [
        "g1_mean_match",
        "g2_mean_entropy",
        "g3_conf_acceptance_rho",
        "g4_logp_rho",
    ]
    higher_a = [_metric_values(arm_a, metric) for metric in higher_metrics]
    higher_b = [_metric_values(arm_b, metric) for metric in higher_metrics]
    g5_a = [_metric_values(arm_a, "g5_behavior_std")]
    g5_b = [_metric_values(arm_b, "g5_behavior_std")]

    fig, (ax_higher, ax_lower) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [3.2, 1.0]}
    )
    _bar_panel(
        ax_higher,
        higher_labels,
        higher_a,
        higher_b,
        title="Arm A vs Arm B: G1–G4",
        ylabel="Metric value",
    )
    _bar_panel(
        ax_lower,
        ["G5\nbehavior σ"],
        g5_a,
        g5_b,
        title="Repeatability",
        ylabel="Std.",
        lower_is_better=True,
    )
    handles, labels = ax_higher.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Functional Specification generator comparison", fontsize=13, y=1.08)
    fig.text(
        0.5,
        -0.01,
        "Higher is better for objective match, diversity, confidence calibration, and LogP control.",
        ha="center",
        fontsize=8.5,
        color="#6b7280",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("runs/gen/arm_comparison"),
        help="Directory produced by scripts/compare_arms.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/gen/arm_comparison/fig_arm_comparison"),
    )
    args = parser.parse_args()
    plot_comparison(args.comparison_dir, args.out)


if __name__ == "__main__":
    main()
