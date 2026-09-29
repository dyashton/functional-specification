#!/usr/bin/env python3
"""Codebook size sweep — executes |C| phase via run_mvp_ablations (or dry plan JSON)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.config import load_config
from functionalspec.eval.harness import dump_json
from functionalspec.metrics.thresholds import THRESHOLDS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/ablations"))
    p.add_argument(
        "--execute",
        action="store_true",
        help="Run codebook phase via scripts/run_mvp_ablations.py",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--recon-epochs", type=int, default=15)
    p.add_argument("--n-probes", type=int, default=200)
    args = p.parse_args()
    cfg = load_config(args.config)
    sizes = cfg["model"]["codebook_sizes"]
    plan = {
        "codebook_sizes": sizes,
        "primary": cfg["model"]["primary_codebook"],
        "metrics_per_size": ["e2_gap", "delta_R_k64", "code_usage", "g4_rho"],
        "selection_rule": "E2 pass then best mean delta_R at k=64; take top-2 for Arm A",
        "thresholds": {
            "e2_gap_pass": THRESHOLDS.e2_gap_pass,
            "e3_r_vs_base_frac": THRESHOLDS.e3_r_vs_base_frac,
        },
        "execute_command": (
            f"uv run python scripts/run_mvp_ablations.py --phase codebook --out {args.out}"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dump_json(plan, args.out / "codebook_sweep_plan.json")
    print(plan)
    if args.execute:
        import subprocess

        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/run_mvp_ablations.py",
                "--phase",
                "codebook",
                "--out",
                str(args.out),
                "--epochs",
                str(args.epochs),
                "--recon-epochs",
                str(args.recon_epochs),
                "--n-probes",
                str(args.n_probes),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
