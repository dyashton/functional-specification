#!/usr/bin/env python3
"""Build on-manifold Functional Specification bank from P1 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.gen.spec_bank import build_spec_bank


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1-checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    meta = build_spec_bank(
        p1_checkpoint=args.p1_checkpoint,
        corpus_a=args.corpus_a,
        out_path=args.out,
        batch_size=args.batch_size,
        device=args.device,
        knn_specificity=args.knn,
    )
    print(f"bank n={meta['n']} R_mean={meta['R_mean']:.3f} → {args.out}")


if __name__ == "__main__":
    main()
