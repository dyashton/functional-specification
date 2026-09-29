#!/usr/bin/env python3
"""Prepare Corpus A (bootstrap: scored GuacaMol components)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.config import load_config
from functionalspec.data.corpus_a import prepare_corpus_a
from functionalspec.data.e9_tasks import write_task_catalog


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests")
    args = p.parse_args()

    cfg = load_config(args.config)
    sp = cfg["corpus_a"]["scaffold_split"]
    df = prepare_corpus_a(
        args.input,
        args.out,
        train_frac=sp["train"],
        val_frac=sp["val"],
        test_frac=sp["test"],
        seed=sp["seed"],
        max_rows=args.max_rows,
    )
    write_task_catalog(args.out.parent / "e9_task_catalog.json")
    print(f"Corpus A: {len(df)} molecules → {args.out}")


if __name__ == "__main__":
    main()
