"""Download / build standardized E9 held-out task CSVs."""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from functionalspec.data.descriptors import SURROGATE_ALWAYS, compute_surrogates
from functionalspec.data.e9_tasks import HELDOUT_TASKS, document_label_correlations, write_task_catalog
from functionalspec.data.splits import save_json

# DeepChem / MoleculeNet mirrors (no deepchem dep required)
URLS = {
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
}


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "functionalspec-e9/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_csv_bytes(raw: bytes, gz: bool = False) -> pd.DataFrame:
    if gz:
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def _valid_smiles(s: str) -> bool:
    mol = Chem.MolFromSmiles(str(s))
    return mol is not None and mol.GetNumAtoms() > 0


def _standardize_table(smiles: list[str], labels: list[float], label_col: str) -> pd.DataFrame:
    rows = []
    for s, y in zip(smiles, labels):
        if not _valid_smiles(s):
            continue
        if not np.isfinite(y):
            continue
        rows.append({"SMILES": str(s), label_col: float(y)})
    return pd.DataFrame(rows).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)


def build_solubility() -> pd.DataFrame:
    df = _read_csv_bytes(_fetch(URLS["esol"]))
    # Delaney: measured log solubility
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    y_col = "measured log solubility in mols per litre"
    return _standardize_table(df[smi_col].tolist(), df[y_col].astype(float).tolist(), "solubility")


def build_toxicity() -> pd.DataFrame:
    df = _read_csv_bytes(_fetch(URLS["clintox"]), gz=True)
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    # CT_TOX = clinical toxicity (1 = toxic)
    y = df["CT_TOX"].astype(float).tolist()
    return _standardize_table(df[smi_col].tolist(), y, "toxic")


def build_permeability() -> pd.DataFrame:
    """Lipophilicity (exp logD) as membrane-partition / permeability proxy."""
    df = _read_csv_bytes(_fetch(URLS["lipophilicity"]))
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    return _standardize_table(df[smi_col].tolist(), df["exp"].astype(float).tolist(), "permeability")


def build_heldout_binding() -> pd.DataFrame:
    """BACE pIC50 — held-out bioassay distinct from CO2 IE."""
    df = _read_csv_bytes(_fetch(URLS["bace"]))
    smi_col = "mol" if "mol" in df.columns else ("smiles" if "smiles" in df.columns else "SMILES")
    y_col = "pIC50" if "pIC50" in df.columns else "PIC50"
    return _standardize_table(df[smi_col].tolist(), df[y_col].astype(float).tolist(), "activity")


def fraction_csp3(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(Descriptors.FractionCSP3(mol))
    except Exception:
        return None


def build_flexibility(seed_smiles: list[str]) -> pd.DataFrame:
    """FractionCSP3 proxy — nRot is a train surrogate, so avoid it as the E9 label."""
    rows = []
    for s in tqdm(seed_smiles, desc="flexibility", leave=False):
        if not _valid_smiles(s):
            continue
        v = fraction_csp3(s)
        if v is None or not np.isfinite(v):
            continue
        rows.append({"SMILES": s, "flexibility": float(v)})
    return pd.DataFrame(rows).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)


def attach_train_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit train-style surrogates for leakage / correlation audit."""
    extra: dict[str, list[float]] = {c: [] for c in SURROGATE_ALWAYS}
    keep = []
    for s in df["SMILES"].astype(str):
        d = compute_surrogates(s)
        if d is None or any(c not in d or not np.isfinite(d[c]) for c in SURROGATE_ALWAYS):
            keep.append(False)
            for c in SURROGATE_ALWAYS:
                extra[c].append(np.nan)
            continue
        keep.append(True)
        for c in SURROGATE_ALWAYS:
            extra[c].append(float(d[c]))
    out = df.copy()
    for c, vals in extra.items():
        out[c] = vals
    return out.loc[keep].reset_index(drop=True)


def prepare_e9(
    out_dir: Path,
    corpus_a_dir: Path | None = None,
    max_flex_pool: int = 8000,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_task_catalog(out_dir.parent / "e9_task_catalog.json")

    builders = {
        "solubility": build_solubility,
        "toxicity": build_toxicity,
        "permeability": build_permeability,
        "heldout_binding": build_heldout_binding,
    }
    meta: dict[str, Any] = {"tasks": {}, "sources": {}}
    all_smiles: list[str] = []

    for task_id, fn in builders.items():
        print(f"Building {task_id}...")
        df = fn()
        df = attach_train_surrogates(df)
        path = out_dir / f"{task_id}.csv"
        df.to_csv(path, index=False)
        label = next(t.label_col for t in HELDOUT_TASKS if t.task_id == task_id)
        meta["tasks"][task_id] = {
            "path": str(path),
            "n": int(len(df)),
            "label_col": label,
            "task_type": next(t.task_type for t in HELDOUT_TASKS if t.task_id == task_id),
        }
        meta["sources"][task_id] = {
            "solubility": "MoleculeNet ESOL (Delaney)",
            "toxicity": "MoleculeNet ClinTox (CT_TOX)",
            "permeability": "MoleculeNet Lipophilicity (exp logD; membrane-partition proxy)",
            "heldout_binding": "MoleculeNet BACE (pIC50)",
        }[task_id]
        all_smiles.extend(df["SMILES"].astype(str).tolist())
        print(f"  → {len(df)} rows → {path}")

    # Flexibility pool: union of held-out + optional corpus A
    pool = list(dict.fromkeys(all_smiles))
    if corpus_a_dir is not None:
        for split in ("train", "val"):
            p = corpus_a_dir / f"{split}.csv"
            if p.exists():
                pool.extend(pd.read_csv(p)["SMILES"].astype(str).tolist())
    pool = list(dict.fromkeys(pool))[:max_flex_pool]
    print(f"Building flexibility on {len(pool)} SMILES...")
    flex = build_flexibility(pool)
    flex = attach_train_surrogates(flex)
    flex_path = out_dir / "flexibility.csv"
    flex.to_csv(flex_path, index=False)
    meta["tasks"]["flexibility"] = {
        "path": str(flex_path),
        "n": int(len(flex)),
        "label_col": "flexibility",
        "task_type": "regression",
    }
    meta["sources"]["flexibility"] = "RDKit FractionCSP3 (nRot avoided — used in P1 train heads)"
    print(f"  → {len(flex)} rows → {flex_path}")

    # Leakage audit: correlate each task label with train surrogates
    corr_dir = out_dir / "correlations"
    corr_dir.mkdir(parents=True, exist_ok=True)
    leakage_flags: list[dict[str, Any]] = []
    for task_id, info in meta["tasks"].items():
        df = pd.read_csv(info["path"])
        label = info["label_col"]
        cols = [label] + [c for c in SURROGATE_ALWAYS if c in df.columns]
        document_label_correlations(df, cols, corr_dir / f"{task_id}_vs_surrogates.csv")
        # flag |rho| > 0.7 with any train surrogate
        for c in SURROGATE_ALWAYS:
            if c not in df.columns:
                continue
            mask = np.isfinite(df[label]) & np.isfinite(df[c])
            if mask.sum() < 10:
                continue
            rho = float(np.corrcoef(df.loc[mask, label], df.loc[mask, c])[0, 1])
            if abs(rho) > 0.7:
                leakage_flags.append({"task": task_id, "surrogate": c, "pearson": rho})

    meta["leakage_flags_rho_gt_0.7"] = leakage_flags
    save_json(meta, out_dir / "e9_prepare_meta.json")
    if leakage_flags:
        print(f"WARNING: {len(leakage_flags)} label↔surrogate |ρ|>0.7 pairs (see meta)")
    else:
        print("No label↔surrogate |ρ|>0.7 flags")
    return meta
