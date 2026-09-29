"""SMILES → graph tensors for the GINE encoder."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from functionalspec.data.descriptors import ecfp_bits, mol_from_smiles

# Fixed dims must match MoleculeEncoder defaults / train_p1 config.
NODE_DIM = 64
EDGE_DIM = 16

_ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H B C N O F Si P S Cl Br I
_HYB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(idx: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def atom_features(atom: Chem.Atom) -> list[float]:
    z = atom.GetAtomicNum()
    z_idx = _ATOM_LIST.index(z) if z in _ATOM_LIST else len(_ATOM_LIST)
    hyb = atom.GetHybridization()
    hyb_idx = _HYB.index(hyb) if hyb in _HYB else len(_HYB)
    feats = (
        _one_hot(z_idx, len(_ATOM_LIST) + 1)
        + _one_hot(atom.GetTotalDegree(), 6)
        + _one_hot(atom.GetFormalCharge() + 2, 5)
        + _one_hot(hyb_idx, len(_HYB) + 1)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()), float(atom.GetTotalNumHs(includeNeighbors=True))]
    )
    if len(feats) < NODE_DIM:
        feats = feats + [0.0] * (NODE_DIM - len(feats))
    return feats[:NODE_DIM]


def bond_features(bond: Chem.Bond) -> list[float]:
    bt = bond.GetBondType()
    types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    idx = types.index(bt) if bt in types else len(types)
    feats = _one_hot(idx, len(types) + 1) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    if len(feats) < EDGE_DIM:
        feats = feats + [0.0] * (EDGE_DIM - len(feats))
    return feats[:EDGE_DIM]


def smiles_to_graph(smiles: str) -> dict[str, torch.Tensor] | None:
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    rows, cols, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows.extend([i, j])
        cols.extend([j, i])
        attrs.extend([bf, bf])
    if attrs:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def smiles_to_fp(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    return ecfp_bits(smiles, n_bits=n_bits)
