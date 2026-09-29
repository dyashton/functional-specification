"""E9 train vs held-out task suite definitions and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from functionalspec.data.splits import save_json


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    role: str  # "train" | "heldout"
    label_col: str
    description: str
    suggested_source: str
    task_type: str  # "regression" | "classification"


TRAIN_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("homo", "train", "homo", "HOMO energy", "QM9 / MoleculeNet", "regression"),
    TaskSpec("lumo", "train", "lumo", "LUMO energy", "QM9 / MoleculeNet", "regression"),
    TaskSpec("dipole", "train", "dipole", "Dipole moment", "QM9", "regression"),
    TaskSpec("co2_ie", "train", "EI", "Host–CO2 interaction energy", "Corpus B compiled.csv", "regression"),
    TaskSpec("pfc", "train", "PFC_composite", "Polar focal concavity proxy", "CO2 scored bridge", "regression"),
)

HELDOUT_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("solubility", "heldout", "solubility", "Aqueous solubility", "ESOL / AqSolDB", "regression"),
    TaskSpec("toxicity", "heldout", "toxic", "Clinical toxicity proxy", "MoleculeNet ClinTox", "classification"),
    TaskSpec("permeability", "heldout", "permeability", "Membrane permeability", "PAMPA / Caco-2", "regression"),
    TaskSpec(
        "heldout_binding",
        "heldout",
        "activity",
        "Held-out bioassay activity",
        "ChEMBL single-assay holdout",
        "regression",
    ),
    TaskSpec(
        "flexibility",
        "heldout",
        "flexibility",
        "Conformational flexibility proxy (prefer 3D RoG if nRot used in train)",
        "RDKit / conformer stats",
        "regression",
    ),
)


def task_catalog() -> dict:
    return {
        "train": [t.__dict__ for t in TRAIN_TASKS],
        "heldout": [t.__dict__ for t in HELDOUT_TASKS],
    }


def write_task_catalog(out_path: Path) -> None:
    save_json(task_catalog(), out_path)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def document_label_correlations(
    frame: pd.DataFrame,
    label_cols: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """Write pairwise correlations so E9 'unseen' leakage can be audited."""
    cols = [c for c in label_cols if c in frame.columns]
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            mat.loc[ci, cj] = pearson_safe(frame[ci].to_numpy(dtype=float), frame[cj].to_numpy(dtype=float))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out_path)
    save_json({"columns": cols, "matrix": mat.fillna(None).to_dict()}, out_path.with_suffix(".json"))
    return mat


# --- Dataset loaders (stubs until external data is fetched) ---

Loader = Callable[[Path], pd.DataFrame]


def load_corpus_b_as_co2_task(corpus_b_dir: Path) -> pd.DataFrame:
    path = corpus_b_dir / "molecules.parquet"
    if not path.exists():
        path_csv = corpus_b_dir / "train.csv"
        if path_csv.exists():
            return pd.read_csv(path_csv)
        raise FileNotFoundError(f"Missing corpus B at {corpus_b_dir}")
    return pd.read_parquet(path)


def load_task_table(task_id: str, data_root: Path) -> pd.DataFrame:
    """Load a standardized SMILES + label table for an E9 task.

    Expected layout: data_root / e9 / {task_id}.csv with columns SMILES, <label>
    """
    path = data_root / "e9" / f"{task_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"E9 task file not found: {path}. Place a CSV with SMILES and label columns "
            f"(see docs/DATA_SPEC.md)."
        )
    return pd.read_csv(path)
