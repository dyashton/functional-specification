#!/usr/bin/env python3
"""Train Phase II ContextFAN (EnvEncoder + IR→Spec query). Generator frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.train.phase2_fan import train_phase2_fan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--co2-root", type=Path, default=Path("../CO2_IE_Dataset"))
    p.add_argument("--out", type=Path, default=Path("runs/phase2_co2"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = train_phase2_fan(
        p1_checkpoint=args.p1_checkpoint,
        bank_path=args.bank,
        co2_root=args.co2_root,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    print(f"n_posed={meta['n_posed']} best_val={meta['best_val_loss']:.4f} → {meta['checkpoint']}")


if __name__ == "__main__":
    main()
