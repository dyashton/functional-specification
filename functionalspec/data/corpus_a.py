"""Corpus A: large diverse surrogate-labeled molecules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from functionalspec.data.descriptors import compute_surrogates
from functionalspec.data.splits import murcko_scaffold, scaffold_split, write_split_frame

BRIDGE_EXTRA_COLS = ("apol", "basicity", "PFC_composite", "TopoPSA", "HallKierAlpha", "LogP", "MW")


def load_bridge_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SMILES" not in df.columns:
        raise ValueError(f"Expected SMILES column in {path}")
    df = df.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    return df


def enrich_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        smi = str(r["SMILES"])
        extra = {c: r[c] for c in BRIDGE_EXTRA_COLS if c in df.columns and pd.notna(r[c])}
        surr = compute_surrogates(smi, extra=extra)
        if surr is None:
            continue
        scaf = murcko_scaffold(smi)
        rows.append({"SMILES": smi, "scaffold": scaf, **surr})
    return pd.DataFrame(rows)


def prepare_corpus_a(
    input_csv: Path,
    out_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_rows: int | None = None,
) -> pd.DataFrame:
    df = load_bridge_csv(input_csv)
    if max_rows is not None:
        df = df.head(max_rows)
    enriched = enrich_surrogates(df)
    splits = scaffold_split(
        enriched["SMILES"].tolist(),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    write_split_frame(enriched, splits, out_dir)
    return enriched
