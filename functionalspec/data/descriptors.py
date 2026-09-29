"""RDKit surrogate functional views and structure probes."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED

# Invalid SMILES are expected during E0 / generation eval — keep stderr clean.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


SURROGATE_ALWAYS = ("MW", "LogP", "TPSA", "QED", "HBA", "HBD", "nRot")


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def compute_surrogates(smiles: str, extra: dict[str, float] | None = None) -> dict[str, float] | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    out: dict[str, float] = {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "QED": float(QED.qed(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "nRot": float(Lipinski.NumRotatableBonds(mol)),
        "HallKierAlpha": float(Descriptors.HallKierAlpha(mol)),
    }
    if extra:
        for k, v in extra.items():
            if v is not None and k not in out:
                out[k] = float(v)
            elif v is not None and k in ("apol", "basicity", "PFC_composite", "TopoPSA"):
                out[k] = float(v)
    return out


def ecfp_bits(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    # Explicit bit iteration keeps dependency light
    on = fp.GetOnBits()
    arr[list(on)] = 1.0
    return arr


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    return inter / union if union > 0 else 0.0


def brics_fragment_multiset(smiles: str) -> dict[str, int] | None:
    """Structure probe only — not a planner target."""
    from collections import Counter

    from rdkit.Chem import BRICS

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        frags = BRICS.BRICSDecompose(mol)
    except Exception:
        return None
    return dict(Counter(frags))


def vectorize_surrogates(rows: list[dict[str, float]], keys: list[str]) -> np.ndarray:
    return np.asarray([[r.get(k, np.nan) for k in keys] for r in rows], dtype=np.float64)
