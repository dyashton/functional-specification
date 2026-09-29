#!/usr/bin/env python3
"""DesignObjective → FAN → FunctionalSpecification → generate molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.gen.fan import FunctionalAbstractionNetwork
from functionalspec.gen.generate import generate_from_spec, load_generator
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec_bank import SpecBank


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objective", type=str, required=True, help="LogP=3,TPSA=50 or seed:CCO")
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--generator", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-compose", action="store_true")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("runs/gen/demo"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bank = SpecBank(args.bank)
    enc, _ = load_planner_for_embed(args.p1_checkpoint, device)
    fan = FunctionalAbstractionNetwork(bank, lam=args.lam, encoder=enc, device=device)
    obj = DesignObjective.from_string(args.objective)
    spec = fan.plan(obj, top_k=args.top_k, compose=not args.no_compose)
    print(
        f"outcome={spec.outcome} confidence={spec.confidence:.3f} "
        f"R_internal={spec.specificity_R_internal:.3f} dist={spec.behavior_distance:.3f} "
        f"provenance={spec.provenance}"
    )
    if spec.outcome == "unsupported":
        print("FAN: unsupported design objective — not generating.")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "spec.json").write_text(
            json.dumps(
                {
                    "outcome": spec.outcome,
                    "confidence": spec.confidence,
                    "provenance": spec.provenance,
                },
                indent=2,
            )
        )
        return

    model, tok, ckpt = load_generator(args.generator, device)
    max_len = int(ckpt.get("max_len", 150))
    result = generate_from_spec(
        spec,
        model,
        tok,
        n=args.n,
        temperature=args.temperature,
        tol=args.tol,
        max_len=max_len,
        device=device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accepted.smi").write_text("\n".join(result.accepted) + ("\n" if result.accepted else ""))
    (args.out / "all.smi").write_text("\n".join(result.smiles) + ("\n" if result.smiles else ""))
    summary = {
        "objective": args.objective,
        "outcome": result.outcome,
        "confidence": result.confidence,
        "n_sampled": result.n_sampled,
        "n_accepted": result.n_accepted,
        "acceptance_rate": result.acceptance_rate,
        "empirical_v_beh": result.empirical_v_beh,
        "murcko_entropy": result.murcko_entropy,
        "predicted_behavior": result.predicted_behavior,
        "provenance": spec.provenance,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"accepted={result.n_accepted}/{result.n_sampled} "
        f"rate={result.acceptance_rate:.2f} v_beh={result.empirical_v_beh:.3f} "
        f"H_murcko={result.murcko_entropy:.2f} → {args.out}"
    )


if __name__ == "__main__":
    main()
