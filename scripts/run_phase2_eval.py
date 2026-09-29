#!/usr/bin/env python3
"""Run Phase II I1–I5 evaluation gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.phase2_eval import run_phase2_eval


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fan", type=Path, default=Path("runs/phase2_co2/fan_context.pt"))
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--generator", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--co2-root", type=Path, default=Path("../CO2_IE_Dataset"))
    p.add_argument("--out", type=Path, default=Path("runs/phase2_co2/eval"))
    p.add_argument("--n-gen", type=int, default=32)
    p.add_argument("--n-strategies", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    s = run_phase2_eval(
        fan_ckpt=args.fan,
        bank_path=args.bank,
        generator_ckpt=args.generator,
        co2_root=args.co2_root,
        out_dir=args.out,
        n_gen=args.n_gen,
        n_strategies=args.n_strategies,
        device=args.device,
    )
    h = s["headline"]
    print(
        f"I1 ctx_aff={h['ctx_affinity']:.3f} rand={h['random_affinity']:.3f} "
        f"I2 H={h['diversity_H']:.2f} I3 cos={h['strategy_mean_cosine']:.3f} "
        f"env>rand={h['env_beats_random']} → {args.out / 'phase2_eval.json'}"
    )


if __name__ == "__main__":
    main()
