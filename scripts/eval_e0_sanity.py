#!/usr/bin/env python3
"""E0 sanity on a SMILES column CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from functionalspec.eval.harness import dump_json, e0_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-csv", type=Path, required=True)
    p.add_argument("--col", default="SMILES")
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e0.json"))
    args = p.parse_args()
    df = pd.read_csv(args.smiles_csv)
    report = e0_report(df[args.col].astype(str).tolist())
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
