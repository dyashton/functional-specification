"""Corpus B: CO2 gold IE hosts with strong/weak bands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from functionalspec.data.descriptors import compute_surrogates, ecfp_bits, tanimoto
from functionalspec.data.splits import murcko_scaffold, save_json, scaffold_split, write_split_frame


def load_compiled_ie(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept EI or EInt
    if "EI" not in df.columns and "EInt" in df.columns:
        df = df.rename(columns={"EInt": "EI"})
    if "SMILES" not in df.columns or "EI" not in df.columns:
        raise ValueError(f"Need SMILES and EI columns in {path}")
    df = df.dropna(subset=["SMILES", "EI"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    return df


def assign_bands(df: pd.DataFrame, strong_max: float = -6.0, weak_min: float = -3.0) -> pd.DataFrame:
    out = df.copy()

    def band(ei: float) -> str:
        if ei <= strong_max:
            return "strong"
        if ei >= weak_min:
            return "weak"
        return "mid"

    out["band"] = out["EI"].astype(float).map(band)
    return out


def enrich_corpus_b(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        smi = str(r["SMILES"])
        surr = compute_surrogates(smi)
        if surr is None:
            continue
        rows.append(
            {
                "SMILES": smi,
                "EI": float(r["EI"]),
                "band": r["band"],
                "source": r.get("source", ""),
                "scaffold": murcko_scaffold(smi),
                **surr,
            }
        )
    return pd.DataFrame(rows)


def build_strong_weak_pairs(
    df: pd.DataFrame,
    max_pairs: int = 500,
    hard_neg_tanimoto_min: float = 0.4,
) -> list[dict]:
    """Contrastive pair catalog: strong–strong positives; strong–weak hard negatives."""
    strong = df[df["band"] == "strong"].reset_index(drop=True)
    weak = df[df["band"] == "weak"].reset_index(drop=True)
    pairs: list[dict] = []

    # Positives: all strong–strong (capped)
    for i in range(len(strong)):
        for j in range(i + 1, len(strong)):
            pairs.append(
                {
                    "type": "positive",
                    "smiles_a": strong.loc[i, "SMILES"],
                    "smiles_b": strong.loc[j, "SMILES"],
                    "ei_a": float(strong.loc[i, "EI"]),
                    "ei_b": float(strong.loc[j, "EI"]),
                }
            )
            if len([p for p in pairs if p["type"] == "positive"]) >= max_pairs:
                break
        if len([p for p in pairs if p["type"] == "positive"]) >= max_pairs:
            break

    # Hard negatives: high ECFP similarity, weak IE
    fps_s = [ecfp_bits(s) for s in strong["SMILES"]]
    fps_w = [ecfp_bits(s) for s in weak["SMILES"]]
    hard = 0
    for i, fa in enumerate(fps_s):
        if fa is None:
            continue
        for j, fb in enumerate(fps_w):
            if fb is None:
                continue
            t = tanimoto(fa, fb)
            if t >= hard_neg_tanimoto_min:
                pairs.append(
                    {
                        "type": "hard_negative",
                        "smiles_a": strong.loc[i, "SMILES"],
                        "smiles_b": weak.loc[j, "SMILES"],
                        "ei_a": float(strong.loc[i, "EI"]),
                        "ei_b": float(weak.loc[j, "EI"]),
                        "tanimoto": t,
                    }
                )
                hard += 1
                if hard >= max_pairs:
                    break
        if hard >= max_pairs:
            break

    return pairs


def prepare_corpus_b(
    input_csv: Path,
    out_dir: Path,
    strong_max: float = -6.0,
    weak_min: float = -3.0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    raw = load_compiled_ie(input_csv)
    banded = assign_bands(raw, strong_max=strong_max, weak_min=weak_min)
    enriched = enrich_corpus_b(banded)
    splits = scaffold_split(
        enriched["SMILES"].tolist(),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    write_split_frame(enriched, splits, out_dir)

    counts = enriched["band"].value_counts().to_dict()
    save_json(
        {
            "strong_ie_max": strong_max,
            "weak_ie_min": weak_min,
            "counts": {k: int(v) for k, v in counts.items()},
            "n": int(len(enriched)),
            "ei_mean": float(enriched["EI"].mean()),
            "ei_median": float(enriched["EI"].median()),
        },
        out_dir / "bands.json",
    )
    pairs = build_strong_weak_pairs(enriched)
    save_json(pairs, out_dir / "strong_weak_pairs.json")
    return enriched
