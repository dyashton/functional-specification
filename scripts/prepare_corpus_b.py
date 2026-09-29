#!/usr/bin/env python3
"""Prepare Corpus B from CO2 IE compiled.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.config import load_config
from functionalspec.data.corpus_b import prepare_corpus_b
from functionalspec.data.e9_tasks import write_task_catalog


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/corpus_b"))
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    b = cfg["corpus_b"]
    sp = b["scaffold_split"]
    df = prepare_corpus_b(
        args.input,
        args.out,
        strong_max=b["strong_ie_max"],
        weak_min=b["weak_ie_min"],
        train_frac=sp["train"],
        val_frac=sp["val"],
        test_frac=sp["test"],
        seed=sp["seed"],
    )
    write_task_catalog(args.out.parent / "e9_task_catalog.json")
    print(f"Corpus B: {len(df)} molecules → {args.out}")
    print(df["band"].value_counts().to_dict())


if __name__ == "__main__":
    main()
