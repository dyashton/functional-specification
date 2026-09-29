#!/usr/bin/env python3
"""E9: frozen S vs ECFP low-n transfer on held-out tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.e9_run import run_e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--e9-dir", type=Path, default=Path("data/processed/e9"))
    p.add_argument("--out", type=Path, default=Path("runs/p1/e9"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-mols", type=int, default=None, help="Cap per task (smoke)")
    args = p.parse_args()

    summary = run_e9(
        checkpoint=args.checkpoint,
        e9_dir=args.e9_dir,
        out_dir=args.out,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        max_mols=args.max_mols,
    )
    print(
        f"E9 pass={summary['pass']} wins={summary['n_wins']}/{summary['n_tasks']} "
        f"→ {args.out / 'e9_summary.json'}"
    )


if __name__ == "__main__":
    main()
