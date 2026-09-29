#!/usr/bin/env python3
"""G6a: matched S vs property-y vs random-S on LogP ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.g6_ablation import run_g6a


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--s-checkpoint", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--y-checkpoint", type=Path, default=Path("runs/p2_property/p2_property_best.pt"))
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/gen/g6a"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = run_g6a(
        bank_path=args.bank,
        s_checkpoint=args.s_checkpoint,
        y_checkpoint=args.y_checkpoint,
        p1_checkpoint=args.p1_checkpoint,
        out_dir=args.out,
        n=args.n,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )
    h = summary["headline"]
    print(
        f"G6a S ρ={h['true_S']['logp_pearson']:.3f} hit={h['true_S']['mean_hit_rate_pm05']:.3f} H={h['true_S']['mean_murcko_entropy']:.2f} | "
        f"y ρ={h['property_y']['logp_pearson']:.3f} hit={h['property_y']['mean_hit_rate_pm05']:.3f} H={h['property_y']['mean_murcko_entropy']:.2f} | "
        f"rand ρ={h['random_S']['logp_pearson']:.3f} hit={h['random_S']['mean_hit_rate_pm05']:.3f} "
        f"→ {args.out / 'g6a_summary.json'}"
    )


if __name__ == "__main__":
    main()
