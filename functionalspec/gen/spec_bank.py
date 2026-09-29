"""Specification bank: on-manifold programs with behavior + specificity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.eval.harness import dump_json


def build_spec_bank(
    p1_checkpoint: Path,
    corpus_a: Path,
    out_path: Path,
    batch_size: int = 64,
    device: str | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    knn_specificity: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Embed corpus; store flat_S, y (raw), smiles; estimate neighborhood R."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_planner_for_embed(p1_checkpoint, device_t)
    cols = list(ckpt["surrogate_cols"])
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float64)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float64)

    dfs = [load_split_csv(corpus_a / f"{sp}.csv") for sp in splits if (corpus_a / f"{sp}.csv").exists()]
    import pandas as pd

    df = pd.concat(dfs, ignore_index=True)
    ds = MoleculeGraphDataset(df, cols, y_mean, y_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    Ss, Yz, smiles, indices = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="build spec bank"):
            out = model.forward_graphs(
                batch["x"].to(device_t),
                batch["edge_index"].to(device_t),
                batch["edge_attr"].to(device_t),
                batch["batch"].to(device_t),
            )
            Ss.append(out["flat_S"].cpu().numpy())
            indices.append(out["indices"].cpu().numpy())
            Yz.append(batch["y"].numpy())
            smiles.extend(batch["smiles"])

    S = np.concatenate(Ss, axis=0).astype(np.float32)
    Yz_arr = np.concatenate(Yz, axis=0).astype(np.float64)
    Y_raw = Yz_arr * y_std + y_mean
    idx = np.concatenate(indices, axis=0)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)

    # Neighborhood specificity proxy: R ~ (scaffold-free) use 1/v_beh of kNN in S
    rng = np.random.default_rng(seed)
    n = len(S)
    sample_n = min(n, 2000)
    probe = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    # For each molecule, estimate local v_beh from kNN (use all via chunked matmul for probes only,
    # then assign each mol the R of its nearest probe — coarse but cheap)
    k = min(knn_specificity, n - 1)
    R = np.full(n, np.nan, dtype=np.float64)
    density = np.full(n, np.nan, dtype=np.float64)
    for q in tqdm(probe, desc="specificity", leave=False):
        sims = Sn @ Sn[q]
        sims[q] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = np.unique(np.concatenate([top, [q]]))
        v = float(np.mean(np.var(Yz_arr[top], axis=0)))
        # structural proxy: mean pairwise cosine distance in S as diversity stand-in
        sub = Sn[top]
        # H-like: use 1 - mean NN sim among neighbors
        nn_sims = []
        for i in range(len(top)):
            srow = sub @ sub[i]
            srow[i] = -1
            nn_sims.append(float(srow.max()))
        d_struct = 1.0 - float(np.mean(nn_sims)) if nn_sims else 0.0
        r = d_struct / (v + 1e-6)
        dens = float(np.mean(np.sort(sims)[-k:]))  # mean sim to kNN before -inf
        for j in top:
            if not np.isfinite(R[j]) or r > R[j]:
                R[j] = r
            if not np.isfinite(density[j]):
                density[j] = dens
    # fill missing with median
    med_r = float(np.nanmedian(R[np.isfinite(R)])) if np.isfinite(R).any() else 1.0
    med_d = float(np.nanmedian(density[np.isfinite(density)])) if np.isfinite(density).any() else 0.0
    R = np.where(np.isfinite(R), R, med_r)
    density = np.where(np.isfinite(density), density, med_d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        flat_S=S,
        Y_raw=Y_raw.astype(np.float32),
        Y_z=Yz_arr.astype(np.float32),
        token_indices=idx.astype(np.int32),
        R=R.astype(np.float32),
        density=density.astype(np.float32),
        y_mean=y_mean.astype(np.float64),
        y_std=y_std.astype(np.float64),
        smiles=np.array(smiles, dtype=object),
    )
    meta = {
        "path": str(out_path),
        "n": int(n),
        "surrogate_cols": cols,
        "knn_specificity": knn_specificity,
        "p1_checkpoint": str(p1_checkpoint),
        "R_mean": float(R.mean()),
        "R_std": float(R.std()),
    }
    dump_json(meta, out_path.with_suffix(".meta.json"))
    return meta


class SpecBank:
    def __init__(self, path: Path):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=True)
        self.flat_S = data["flat_S"]
        self.Y_raw = data["Y_raw"]
        self.Y_z = data["Y_z"]
        self.token_indices = data["token_indices"]
        self.R = data["R"]
        self.density = data["density"]
        self.y_mean = data["y_mean"]
        self.y_std = data["y_std"]
        self.smiles = list(data["smiles"])
        meta_path = self.path.with_suffix(".meta.json")
        if meta_path.exists():
            import json

            self.meta = json.loads(meta_path.read_text())
            self.surrogate_cols = list(self.meta["surrogate_cols"])
        else:
            self.meta = {}
            self.surrogate_cols = []

    def __len__(self) -> int:
        return len(self.smiles)
