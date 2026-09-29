#!/usr/bin/env python3
"""E3: freeze S*, sample many molecules, score structural diversity / behavioral variance."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.e3_run import run_e3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/p2_selfies/p2_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p2_selfies/e3"))
    p.add_argument("--n-specs", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = run_e3(
        checkpoint=args.checkpoint,
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        n_specs=args.n_specs,
        n_samples=args.n_samples,
        split=args.split,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )
    print(
        f"E3 pass_rate={summary['pass_rate']:.2f} "
        f"({summary['n_pass']}/{summary['n_specs']}) → {args.out / 'e3_summary.json'}"
    )


if __name__ == "__main__":
    main()
