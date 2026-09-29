#!/usr/bin/env python3
"""E2 probes: surrogate vs structure from embedding matrices (CSV/NPY)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from functionalspec.eval.harness import dump_json, e2_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=Path, required=True, help=".npy (n, d_s)")
    p.add_argument("--Y-surrogate", type=Path, required=True, help=".npy (n, d_f)")
    p.add_argument("--Y-structure", type=Path, required=True, help=".npy (n, d_struct)")
    p.add_argument("--S-recon", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e2.json"))
    args = p.parse_args()

    S = np.load(args.S)
    Yf = np.load(args.Y_surrogate)
    Ys = np.load(args.Y_structure)
    Sr = np.load(args.S_recon) if args.S_recon else None
    report = e2_report(S, Yf, Ys, Sr)
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
