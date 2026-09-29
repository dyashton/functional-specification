"""PyTorch dataset + collate for Corpus A/B graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from functionalspec.data.featurize import smiles_to_fp, smiles_to_graph
from functionalspec.data.motifs import MotifVocabulary

DEFAULT_SURROGATES = [
    "MW",
    "LogP",
    "TPSA",
    "QED",
    "HBA",
    "HBD",
    "nRot",
    "HallKierAlpha",
    "apol",
    "basicity",
    "PFC_composite",
]


class MoleculeGraphDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        surrogate_cols: list[str],
        y_mean: np.ndarray,
        y_std: np.ndarray,
        fp_bits: int = 2048,
        motif_vocab: MotifVocabulary | None = None,
    ):
        self.smiles: list[str] = []
        self.graphs: list[dict[str, torch.Tensor]] = []
        self.y: list[np.ndarray] = []
        self.fps: list[np.ndarray] = []
        self.surrogate_cols = surrogate_cols
        self.motif_vocab = motif_vocab

        for _, row in df.iterrows():
            smi = str(row["SMILES"])
            g = smiles_to_graph(smi)
            fp = smiles_to_fp(smi, n_bits=fp_bits)
            if g is None or fp is None:
                continue
            raw = np.asarray([float(row[c]) if pd.notna(row[c]) else np.nan for c in surrogate_cols], dtype=np.float64)
            if not np.isfinite(raw).all():
                continue
            yz = (raw - y_mean) / y_std
            self.smiles.append(smi)
            self.graphs.append(g)
            self.y.append(yz.astype(np.float32))
            self.fps.append(fp.astype(np.float32))

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        g = self.graphs[idx]
        item = {
            "smiles": self.smiles[idx],
            "x": g["x"],
            "edge_index": g["edge_index"],
            "edge_attr": g["edge_attr"],
            "y": torch.from_numpy(self.y[idx]),
            "fp": torch.from_numpy(self.fps[idx]),
        }
        if self.motif_vocab is not None:
            item["motif_targets"] = torch.tensor(
                self.motif_vocab.encode(self.smiles[idx]), dtype=torch.long
            )
        return item


def compute_y_stats(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Y = df[cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def load_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def available_surrogates(df: pd.DataFrame, preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or DEFAULT_SURROGATES
    return [c for c in preferred if c in df.columns]


def collate_graphs(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    xs, eidxs, eattrs, ys, fps, motif_targets, batch_idx = [], [], [], [], [], [], []
    smiles = []
    node_offset = 0
    for i, item in enumerate(batch):
        n = item["x"].size(0)
        xs.append(item["x"])
        ei = item["edge_index"] + node_offset
        eidxs.append(ei)
        eattrs.append(item["edge_attr"])
        ys.append(item["y"])
        fps.append(item["fp"])
        if "motif_targets" in item:
            motif_targets.append(item["motif_targets"])
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        smiles.append(item["smiles"])
        node_offset += n

    if eidxs and all(e.numel() > 0 for e in eidxs):
        edge_index = torch.cat(eidxs, dim=1)
        edge_attr = torch.cat(eattrs, dim=0)
    elif eidxs:
        nonempty = [(e, a) for e, a in zip(eidxs, eattrs) if e.numel() > 0]
        if nonempty:
            edge_index = torch.cat([e for e, _ in nonempty], dim=1)
            edge_attr = torch.cat([a for _, a in nonempty], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, batch[0]["edge_attr"].size(-1)), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 16), dtype=torch.float32)

    out = {
        "smiles": smiles,
        "x": torch.cat(xs, dim=0),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": torch.cat(batch_idx, dim=0),
        "y": torch.stack(ys, dim=0),
        "fp": torch.stack(fps, dim=0),
    }
    if motif_targets:
        out["motif_targets"] = torch.stack(motif_targets, dim=0)
    return out
