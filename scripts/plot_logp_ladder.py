#!/usr/bin/env python3
"""G4/G6 figure: LogP ladder control + Spec vs property vs random."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows_by_kind(rows: list[dict], kind: str) -> list[dict]:
    return sorted(
        [r for r in rows if r.get("kind") == kind and np.isfinite(r.get("logp_mean", float("nan")))],
        key=lambda r: float(r["target_logp"]),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g4", type=Path, default=Path("runs/gen/eval/gen_eval_summary.json"))
    p.add_argument("--g6a", type=Path, default=Path("runs/gen/g6a/g6a_summary.json"))
    p.add_argument("--g6b", type=Path, default=Path("runs/gen/g6b/g6b_summary.json"))
    p.add_argument("--out", type=Path, default=Path("runs/gen/fig_logp_ladder"))
    args = p.parse_args()

    g6a = json.loads(args.g6a.read_text())
    # g4 / g6b available for other panels; left panel uses G6a only (cleaner)

    style = {
        "true_S": {"c": "#0f766e", "ls": "-", "marker": "o", "label": r"$S$", "z": 4, "lw": 2.2},
        "property_y": {"c": "#b45309", "ls": "--", "marker": "s", "label": "property", "z": 3, "lw": 1.8},
        "random_S": {"c": "#9ca3af", "ls": ":", "marker": "^", "label": "random", "z": 2, "lw": 1.6},
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # --- Panel A: three curves only, means (no error bars / overlays) ---
    ax = axes[0]
    lo, hi = -1.0, 5.2
    ax.plot([lo, hi], [lo, hi], color="#e5e7eb", lw=1.2, zorder=1)

    rho_bits = []
    for kind in ("true_S", "property_y", "random_S"):
        rows = _rows_by_kind(g6a["rows"], kind)
        if not rows:
            continue
        x = np.asarray([r["target_logp"] for r in rows], dtype=float)
        y = np.asarray([r["logp_mean"] for r in rows], dtype=float)
        st = style[kind]
        rho = g6a["headline"][kind]["logp_pearson"]
        rho_bits.append((st["label"], rho, st["c"]))
        ax.plot(
            x,
            y,
            color=st["c"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.6,
            markeredgecolor=st["c"],
            lw=st["lw"],
            label=st["label"],
            zorder=st["z"],
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Target LogP")
    ax.set_ylabel("Realized mean LogP")
    ax.set_title("Behavior navigation")
    ax.legend(frameon=False, fontsize=9, loc="upper left", handlelength=2.2)
    # ρ as a quiet annotation (keeps legend short)
    ann = "\n".join(f"{lab}:  $\\rho$ = {rho:.2f}" for lab, rho, _ in rho_bits)
    ax.text(
        0.98,
        0.04,
        ann,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4b5563",
        linespacing=1.45,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    # --- Panel B: diversity (Murcko H only) ---
    ax = axes[1]
    kinds = ["true_S", "property_y", "random_S"]
    labels = [style[k]["label"] for k in kinds]
    colors = [style[k]["c"] for k in kinds]
    H = [g6a["headline"][k]["mean_murcko_entropy"] for k in kinds]
    xpos = np.arange(len(kinds))
    ax.bar(xpos, H, width=0.55, color=colors, alpha=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean Murcko entropy $H$")
    ax.set_ylim(0, max(H) * 1.25)
    ax.set_title("Diversity under control")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Functional Specification as a controllable design interface",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    out = args.out
    png = out if out.suffix == ".png" else Path(str(out) + ".png")
    pdf = out.with_suffix(".pdf") if out.suffix in {".png", ".pdf"} else Path(str(out) + ".pdf")
    if out.suffix == ".pdf":
        png = out.with_suffix(".png")
        pdf = out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"→ {png}")
    print(f"→ {pdf}")


if __name__ == "__main__":
    main()
