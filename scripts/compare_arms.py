#!/usr/bin/env python3
"""Run matched G1-G5 evaluations for Arm A and Arm B."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.gen_eval import run_gen_eval
from functionalspec.eval.harness import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--g5-batches", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compose", action="store_true")
    args = parser.parse_args()
    common = {
        "bank_path": args.bank,
        "p1_checkpoint": args.p1_checkpoint,
        "n": args.n,
        "temperature": args.temperature,
        "device": args.device,
        "seed": args.seed,
        "g5_batches": args.g5_batches,
        "compose": not args.no_compose,
    }
    arm_a = run_gen_eval(
        generator_ckpt=args.arm_a,
        out_dir=args.out / "arm_a",
        **common,
    )
    arm_b = run_gen_eval(
        generator_ckpt=args.arm_b,
        out_dir=args.out / "arm_b",
        **common,
    )
    comparison = {
        "settings": {
            "bank": str(args.bank),
            "p1_checkpoint": str(args.p1_checkpoint),
            "n": args.n,
            "temperature": args.temperature,
            "seed": args.seed,
            "compose": not args.no_compose,
        },
        "arm_a": arm_a["headline"],
        "arm_b": arm_b["headline"],
    }
    dump_json(comparison, args.out / "arm_comparison.json")
    print(f"Arm A: {arm_a['headline']}")
    print(f"Arm B: {arm_b['headline']}")


if __name__ == "__main__":
    main()
