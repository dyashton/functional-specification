#!/usr/bin/env python3
"""G6b: scramble / random flat_S ablation (decoder uses Spec?)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.g6_ablation import run_g6b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--generator", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/gen/g6b"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = run_g6b(
        bank_path=args.bank,
        generator_ckpt=args.generator,
        p1_checkpoint=args.p1_checkpoint,
        out_dir=args.out,
        n=args.n,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )
    h = summary["headline"]
    print(
        f"G6b true ρ={h['true_S']['logp_pearson']:.3f} "
        f"scramble ρ={h['scramble_S']['logp_pearson']:.3f} "
        f"random ρ={h['random_S']['logp_pearson']:.3f} → {args.out / 'g6b_summary.json'}"
    )


if __name__ == "__main__":
    main()
