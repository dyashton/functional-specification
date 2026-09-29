"""Scaffold splits and I/O helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def scaffold_hash(scaffold: str) -> int:
    return int(hashlib.md5(scaffold.encode()).hexdigest(), 16)


def scaffold_split(
    smiles: Iterable[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Assign molecule indices by Murcko scaffold buckets (deterministic given seed)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    buckets: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        scaf = murcko_scaffold(smi) or f"__invalid_{i}__"
        buckets.setdefault(scaf, []).append(i)

    keys = sorted(buckets.keys(), key=lambda k: scaffold_hash(k) ^ seed)
    n = sum(len(v) for v in buckets.values())
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train, val, test = [], [], []
    for k in keys:
        idxs = buckets[k]
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    assigned = set(train) | set(val) | set(test)
    leftover = [i for i in range(n) if i not in assigned]
    for i in leftover:
        if len(train) < n_train:
            train.append(i)
        elif len(val) < n_val:
            val.append(i)
        else:
            test.append(i)

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_split_frame(df: pd.DataFrame, splits: dict[str, list[int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_parquet(out_dir / "molecules.parquet", index=False)
    save_json(splits, out_dir / "splits.json")
    for name, idxs in splits.items():
        df.iloc[idxs].to_csv(out_dir / f"{name}.csv", index=False)
