#!/usr/bin/env python3
"""Download MoleculeNet held-out tasks + build flexibility CSV for E9."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.data.prepare_e9 import prepare_e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed/e9"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--max-flex-pool", type=int, default=8000)
    args = p.parse_args()

    meta = prepare_e9(args.out, corpus_a_dir=args.corpus_a, max_flex_pool=args.max_flex_pool)
    print("E9 prepare done:")
    for tid, info in meta["tasks"].items():
        print(f"  {tid}: n={info['n']} label={info['label_col']} type={info['task_type']}")


if __name__ == "__main__":
    main()
