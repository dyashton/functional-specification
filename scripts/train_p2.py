#!/usr/bin/env python3
"""Train P2: freeze P1 planner, train Arm A (S → SELFIES by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.train.p2 import train_p2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p2_selfies"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--arm-hidden", type=int, default=512)
    p.add_argument("--arm-layers", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--representation",
        choices=["selfies", "smiles"],
        default="selfies",
        help="Arm A string representation (selfies recommended for validity)",
    )
    p.add_argument("--cond-dropout", type=float, default=0.1)
    args = p.parse_args()

    meta = train_p2(
        p1_checkpoint=args.p1_checkpoint,
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_len=args.max_len,
        arm_hidden=args.arm_hidden,
        arm_layers=args.arm_layers,
        num_workers=args.num_workers,
        seed=args.seed,
        representation=args.representation,
        cond_dropout=args.cond_dropout,
    )
    print(f"representation={meta['representation']} vocab={meta['vocab_size']}")
    print(f"best sample validity={meta['best_sample_validity']:.3f}")
    print(f"checkpoint: {meta['best_checkpoint']}")


if __name__ == "__main__":
    main()
