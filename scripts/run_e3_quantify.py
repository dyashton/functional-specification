#!/usr/bin/env python3
"""E3 quantification: Murcko/BRICS entropy, NN Tanimoto, vs Recon-VQ control."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.e3_quantify import quantify_e3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e3-dir", type=Path, default=Path("runs/p2_selfies/e3"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--recon-epochs", type=int, default=20)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    quantify_e3(
        e3_dir=args.e3_dir,
        corpus_a=args.corpus_a,
        out_path=args.out,
        recon_epochs=args.recon_epochs,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
