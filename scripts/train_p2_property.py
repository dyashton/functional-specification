#!/usr/bin/env python3
"""Train G6a Model B: property-conditional FiLM SELFIES decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.train.p2_property import train_p2_property


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p2_property"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cond-dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = train_p2_property(
        p1_checkpoint=args.p1_checkpoint,
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cond_dropout=args.cond_dropout,
        device=args.device,
        seed=args.seed,
    )
    print(f"best validity={meta['best_sample_validity']:.3f} → {meta['best_checkpoint']}")


if __name__ == "__main__":
    main()
