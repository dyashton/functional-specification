#!/usr/bin/env python3
"""E3 one-S→many: metrics + optional matched-behavior baseline.

Input CSVs must contain SMILES and surrogate columns (default: MW,LogP,TPSA,QED,HBA,HBD,nRot).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from functionalspec.eval.harness import dump_json, e3_report

DEFAULT_SURR = ["MW", "LogP", "TPSA", "QED", "HBA", "HBD", "nRot"]


def load_pair(path: Path, surr_cols: list[str]) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    cols = [c for c in surr_cols if c in df.columns]
    if not cols:
        raise SystemExit(f"No surrogate columns found in {path}; need one of {surr_cols}")
    smiles = df["SMILES"].astype(str).tolist()
    Y = df[cols].to_numpy(dtype=float)
    return smiles, Y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours-csv", type=Path, required=True)
    p.add_argument("--baseline-csv", type=Path, default=None)
    p.add_argument("--surrogates", nargs="+", default=DEFAULT_SURR)
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e3.json"))
    args = p.parse_args()

    smiles, Y = load_pair(args.ours_csv, args.surrogates)
    if args.baseline_csv:
        bs, BY = load_pair(args.baseline_csv, args.surrogates)
        report = e3_report(smiles, Y, bs, BY)
    else:
        report = e3_report(smiles, Y)
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
