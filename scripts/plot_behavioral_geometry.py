#!/usr/bin/env python3
"""Render H_structure vs v_beh overlay for S vs ECFP neighborhoods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("runs/p1/spec_specificity"))
    p.add_argument("--ks", type=int, nargs="+", default=[32, 64])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    fig, axes = plt.subplots(1, len(args.ks), figsize=(4.8 * len(args.ks), 4.3), sharey=True)
    if len(args.ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, args.ks):
        df = pd.read_csv(args.dir / f"soft_neighborhoods_k{k}.csv")
        ax.scatter(
            df.H_murcko_ECFP,
            df.v_beh_ECFP,
            s=32,
            alpha=0.7,
            c="#6b7280",
            label="ECFP neighborhood",
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            df.H_murcko_S,
            df.v_beh_S,
            s=32,
            alpha=0.85,
            c="#0f766e",
            label="$S$ neighborhood",
            edgecolors="none",
            zorder=3,
        )
        ax.set_xlabel(r"Murcko scaffold entropy $H$")
        ax.set_title(f"$k={k}$ neighborhoods (n={len(df)} probes)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Behavioral variance $v_{\mathrm{beh}}$")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        r"Same structural diversity, tighter behavior in $S$-space",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    out = args.out or (args.dir / "fig_behavioral_geometry")
    png = out if out.suffix == ".png" else Path(str(out) + ".png")
    pdf = out.with_suffix(".pdf") if out.suffix in {".png", ".pdf"} else Path(str(out) + ".pdf")
    if out.suffix == ".pdf":
        png = out.with_suffix(".png")
        pdf = out
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"→ {png}")
    print(f"→ {pdf}")


if __name__ == "__main__":
    main()
