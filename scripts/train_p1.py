#!/usr/bin/env python3
"""Train P1: Functional Specification bottleneck (surrogate + contrastive + VQ)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.train.p1 import train_p1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p1"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--codebook-size", type=int, default=None)
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--commitment-cost", type=float, default=None, help="VQ beta")
    p.add_argument("--w-ecfp", type=float, default=None, help="ECFP subtract weight in positives")
    p.add_argument("--no-vq", action="store_true", help="Continuous slots (bypass quantization)")
    p.add_argument("--contrastive-weight", type=float, default=0.1)
    p.add_argument("--topk-pos", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = train_p1(
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        codebook_size=args.codebook_size,
        contrastive_weight=args.contrastive_weight,
        topk_pos=args.topk_pos,
        num_workers=args.num_workers,
        seed=args.seed,
        num_slots=args.num_slots,
        commitment_cost=args.commitment_cost,
        w_ecfp=args.w_ecfp,
        no_vq=args.no_vq,
    )
    print(f"best val Spearman={meta['best_val_spearman']:.4f}")
    print(f"checkpoint: {meta['best_checkpoint']}")


if __name__ == "__main__":
    main()
