"""Build simple CO₂ InteractionEnvironment graphs (MVP atom features only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM, _ATOM_LIST, _one_hot
from functionalspec.env.types import InteractionEnvironment
from functionalspec.env.xyz import Xyz, covalent_cutoff, read_xyz, split_host_co2

# Spatial host–CO2 edges within this cutoff (Å)
SPATIAL_CUTOFF = 4.5


def _atom_feat_from_symbol(symbol: str, *, is_co2: bool, is_host: bool) -> list[float]:
    """XYZ path: no RDKit atom — Z one-hot + role flags, pad to NODE_DIM."""
    z = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    z_idx = _ATOM_LIST.index(z) if z in _ATOM_LIST else len(_ATOM_LIST)
    feats = (
        _one_hot(z_idx, len(_ATOM_LIST) + 1)
        + [0.0] * 6  # degree placeholder
        + [0.0] * 5  # charge
        + [0.0] * 6  # hyb
        + [0.0, 0.0, 0.0]  # aromatic, ring, Hs
        + [float(is_co2), float(is_host)]
    )
    if len(feats) < NODE_DIM:
        feats = feats + [0.0] * (NODE_DIM - len(feats))
    return feats[:NODE_DIM]


def _edge_feat_covalent() -> list[float]:
    # Mark as covalent SINGLE-like
    feats = [1.0, 0.0, 0.0, 0.0, 0.0] + [0.0, 0.0] + [1.0, 0.0]  # last: covalent, spatial
    if len(feats) < EDGE_DIM:
        feats = feats + [0.0] * (EDGE_DIM - len(feats))
    return feats[:EDGE_DIM]


def _edge_feat_spatial(dist: float) -> list[float]:
    # Soft distance encoding in trailing dims
    feats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(dist / SPATIAL_CUTOFF)]
    if len(feats) < EDGE_DIM:
        feats = feats + [0.0] * (EDGE_DIM - len(feats))
    return feats[:EDGE_DIM]


def _graphs_from_xyz(xyz: Xyz, *, local_host_only: bool = True) -> InteractionEnvironment:
    """Host+CO₂ complex → interaction graph.

    If local_host_only: keep only host atoms within SPATIAL_CUTOFF of any CO₂ atom
    (plus all CO₂) — cheaper, matches 'local interaction site'.
    """
    host, co2 = split_host_co2(xyz)
    dmat = np.linalg.norm(host.coords[:, None, :] - co2.coords[None, :, :], axis=-1)
    near = np.where(dmat.min(axis=1) <= SPATIAL_CUTOFF)[0]
    if local_host_only and len(near) > 0:
        host = Xyz(symbols=[host.symbols[i] for i in near], coords=host.coords[near])
    elif local_host_only and len(near) == 0:
        # keep closest 8 host atoms
        order = np.argsort(dmat.min(axis=1))[: min(8, len(host.symbols))]
        host = Xyz(symbols=[host.symbols[i] for i in order], coords=host.coords[order])

    n_host = len(host.symbols)
    symbols = host.symbols + co2.symbols
    coords = np.vstack([host.coords, co2.coords])
    is_co2 = [False] * n_host + [True] * 3
    is_host = [True] * n_host + [False] * 3

    x = torch.tensor(
        [_atom_feat_from_symbol(s, is_co2=c, is_host=h) for s, c, h in zip(symbols, is_co2, is_host)],
        dtype=torch.float32,
    )

    rows, cols, attrs = [], [], []

    def add_edge(i: int, j: int, feat: list[float]) -> None:
        rows.extend([i, j])
        cols.extend([j, i])
        attrs.extend([feat, feat])

    # covalent within host / CO2
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            same_frag = (i < n_host and j < n_host) or (i >= n_host and j >= n_host)
            if same_frag and d <= covalent_cutoff(symbols[i], symbols[j]):
                add_edge(i, j, _edge_feat_covalent())
            elif (i < n_host) != (j < n_host) and d <= SPATIAL_CUTOFF:
                add_edge(i, j, _edge_feat_spatial(d))

    if attrs:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)

    return InteractionEnvironment(
        env_type="co2",
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        provenance="complex_xyz",
        meta={"n_host": n_host, "n_co2": 3},
    )


def co2_only_environment() -> InteractionEnvironment:
    """Shared CO₂ molecule graph (no host) — de novo / env-shared baseline."""
    # Linear O=C=O geometry
    symbols = ["O", "C", "O"]
    coords = np.array([[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]], dtype=np.float64)
    xyz = Xyz(symbols=symbols, coords=coords)
    # Fake host-empty complex: pad with dummy H far away then strip — simpler build inline
    x = torch.tensor(
        [_atom_feat_from_symbol(s, is_co2=True, is_host=False) for s in symbols],
        dtype=torch.float32,
    )
    rows, cols, attrs = [], [], []
    for i, j in ((0, 1), (1, 2)):
        rows.extend([i, j])
        cols.extend([j, i])
        bf = _edge_feat_covalent()
        attrs.extend([bf, bf])
    return InteractionEnvironment(
        env_type="co2",
        x=x,
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        provenance="co2_only",
        meta={"n_host": 0, "n_co2": 3},
    )


def environment_from_complex_xyz(path: Path, *, local_host_only: bool = True) -> InteractionEnvironment:
    return _graphs_from_xyz(read_xyz(path), local_host_only=local_host_only)


def environment_from_host_smiles_rdkit(smiles: str) -> InteractionEnvironment | None:
    """Fallback without pose: host covalent graph + tagged as host-only (no CO₂).

    Used only when XYZ missing — IR still varies by host chemotype for training
    proxies; inference for de novo should prefer co2_only_environment().
    """
    from functionalspec.data.featurize import smiles_to_graph

    g = smiles_to_graph(smiles)
    if g is None:
        return None
    # Tag last feature dims as host (already zero for co2 flag in atom_features —
    # overwrite trailing two slots if present)
    x = g["x"].clone()
    if x.size(1) >= 2:
        x[:, -2] = 0.0  # is_co2
        x[:, -1] = 1.0  # is_host
    return InteractionEnvironment(
        env_type="co2",
        x=x,
        edge_index=g["edge_index"],
        edge_attr=g["edge_attr"],
        provenance="host_smiles_fallback",
        meta={"smiles": smiles, "n_co2": 0},
    )
