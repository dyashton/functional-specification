#!/usr/bin/env python3
"""Run G1–G5 generation evaluations (match, diversity, confidence, LogP ladder, repeatability)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.gen_eval import run_gen_eval


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--generator", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/gen/eval"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--g5-batches", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compose", action="store_true", help="FAN retrieve-only (no Spec compose)")
    args = p.parse_args()

    summary = run_gen_eval(
        bank_path=args.bank,
        generator_ckpt=args.generator,
        p1_checkpoint=args.p1_checkpoint,
        out_dir=args.out,
        n=args.n,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
        g5_batches=args.g5_batches,
        compose=not args.no_compose,
    )
    h = summary["headline"]
    print(
        f"G1 match={h['g1_mean_match']:.3f} G2 H={h['g2_mean_entropy']:.2f} "
        f"G3 ρ={h['g3_conf_acceptance_rho']} G4 LogP ρ={h['g4_logp_rho']} "
        f"G5 beh_std={h['g5_behavior_std']} → {args.out / 'gen_eval_summary.json'}"
    )


if __name__ == "__main__":
    main()
