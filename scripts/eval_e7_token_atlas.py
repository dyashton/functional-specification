#!/usr/bin/env python3
"""E7 token atlas: per-token surrogate mean/std from a long table.

CSV columns: token_id, SMILES, <surrogate cols...>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SURR = ["MW", "LogP", "TPSA", "QED", "HBA", "HBD", "nRot"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--token-col", default="token_id")
    p.add_argument("--surrogates", nargs="+", default=DEFAULT_SURR)
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e7_token_atlas.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    cols = [c for c in args.surrogates if c in df.columns]
    rows = []
    for tid, g in df.groupby(args.token_col):
        row = {"token_id": tid, "n": len(g)}
        for c in cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std())
        rows.append(row)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} tokens)")


if __name__ == "__main__":
    main()
