"""Train Phase II ContextFAN: IR → Spec query via contrastive strong/weak hosts.

Freezes P1 molecule encoder and Arm A generator. EnvEncoder + IRToQuery train only.
DesignObjective is NOT fed into EnvEncoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM, smiles_to_graph
from functionalspec.env.dataset import ie_band, load_posed_complexes
from functionalspec.env.encoder import EnvironmentEncoder
from functionalspec.env.graph_co2 import environment_from_complex_xyz
from functionalspec.env.types import InteractionEnvironment
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.gen.fan_context import ContextFAN, IRToQuery
from functionalspec.gen.spec_bank import SpecBank, build_spec_bank


class PoseSpecDataset(Dataset):
    def __init__(
        self,
        envs: list[InteractionEnvironment],
        flat_S: np.ndarray,
        bands: list[str],
    ):
        self.envs = envs
        self.flat_S = flat_S.astype(np.float32)
        self.bands = bands

    def __len__(self) -> int:
        return len(self.envs)

    def __getitem__(self, i: int) -> dict:
        return {
            "env": self.envs[i],
            "flat_S": torch.from_numpy(self.flat_S[i]),
            "strong": 1.0 if self.bands[i] == "strong" else (0.0 if self.bands[i] == "weak" else 0.5),
            "band": self.bands[i],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "envs": [b["env"] for b in batch],
        "flat_S": torch.stack([b["flat_S"] for b in batch]),
        "strong": torch.tensor([b["strong"] for b in batch], dtype=torch.float32),
        "bands": [b["band"] for b in batch],
    }


@torch.no_grad()
def encode_hosts_safe(planner, smiles_list: list[str], device: torch.device) -> np.ndarray:
    rows = []
    dim = None
    for s in tqdm(smiles_list, desc="encode host Specs"):
        g = smiles_to_graph(s)
        if g is None:
            rows.append(None)
            continue
        x = g["x"].to(device)
        ei = g["edge_index"].to(device)
        ea = g["edge_attr"].to(device)
        b = torch.zeros(x.size(0), dtype=torch.long, device=device)
        out = planner.forward_graphs(x, ei, ea, b)
        v = out["flat_S"][0].detach().cpu().numpy().astype(np.float32)
        dim = v.shape[0]
        rows.append(v)
    assert dim is not None
    return np.stack([r if r is not None else np.zeros(dim, np.float32) for r in rows], axis=0)


def train_phase2_fan(
    p1_checkpoint: Path,
    bank_path: Path,
    co2_root: Path,
    out_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 42,
    obj_dim: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(bank_path).exists():
        build_spec_bank(p1_checkpoint, Path("data/processed/corpus_a"), Path(bank_path))

    bank = SpecBank(bank_path)
    planner, _ = load_planner_for_embed(p1_checkpoint, device_t)
    planner.eval()
    for p in planner.parameters():
        p.requires_grad = False

    posed = load_posed_complexes(co2_root)
    # Keep those with usable SMILES + XYZ
    posed = [p for p in posed if p.smiles and p.xyz_path.is_file()]
    if len(posed) < 8:
        raise RuntimeError(f"Need more posed complexes, found {len(posed)}")

    envs, smiles, ies = [], [], []
    for p in tqdm(posed, desc="build env graphs"):
        try:
            envs.append(environment_from_complex_xyz(p.xyz_path))
            smiles.append(p.smiles)
            ies.append(p.ie_kcal_mol)
        except Exception:
            continue
    bands = [ie_band(e) for e in ies]
    flat_S = encode_hosts_safe(planner, smiles, device_t)

    # Split
    n = len(envs)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(4, n // 10)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def subset(ix):
        return PoseSpecDataset([envs[i] for i in ix], flat_S[ix], [bands[i] for i in ix])

    train_ds, val_ds = subset(train_idx), subset(val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    ir_dim = 256
    spec_dim = flat_S.shape[1]
    env_enc = EnvironmentEncoder(node_dim=NODE_DIM, edge_dim=EDGE_DIM, out_dim=ir_dim).to(device_t)
    proj = IRToQuery(ir_dim=ir_dim, spec_dim=spec_dim, obj_dim=obj_dim).to(device_t)
    opt = torch.optim.AdamW(list(env_enc.parameters()) + list(proj.parameters()), lr=lr, weight_decay=1e-4)

    def step_batch(batch, train: bool) -> float:
        env_enc.train(train)
        proj.train(train)
        irs = []
        for env in batch["envs"]:
            e = InteractionEnvironment(
                env_type=env.env_type,
                x=env.x.to(device_t),
                edge_index=env.edge_index.to(device_t),
                edge_attr=env.edge_attr.to(device_t),
                provenance=env.provenance,
                meta=env.meta,
            )
            irs.append(env_enc.encode(e).embedding)
        ir = torch.stack(irs, dim=0)
        S = F.normalize(batch["flat_S"].to(device_t), dim=-1)
        q = proj(ir, None)
        # InfoNCE: query matches own Spec among batch
        logits = q @ S.T / 0.07
        labels = torch.arange(q.size(0), device=device_t)
        loss_nce = F.cross_entropy(logits, labels)
        # Ranking: strong hosts should have higher self-sim than weak (soft)
        self_sim = (q * S).sum(-1)
        strong = batch["strong"].to(device_t)
        # encourage strong > mid > weak self-alignment
        loss_band = F.mse_loss(self_sim, strong)
        loss = loss_nce + 0.2 * loss_band
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(env_enc.parameters()) + list(proj.parameters()), 5.0)
            opt.step()
        return float(loss.item())

    history = []
    best = float("inf")
    ckpt_path = out_dir / "fan_context.pt"
    for epoch in range(1, epochs + 1):
        tr = [step_batch(b, True) for b in train_loader]
        with torch.no_grad():
            va = [step_batch(b, False) for b in val_loader]
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va))}
        history.append(row)
        print(f"epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}")
        if row["val_loss"] <= best:
            best = row["val_loss"]
            fan = ContextFAN(bank, env_enc, proj, device=device_t)
            fan.save(ckpt_path)

    meta = {
        "n_posed": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "band_counts": {b: int(sum(1 for x in bands if x == b)) for b in ("strong", "mid", "weak")},
        "best_val_loss": best,
        "checkpoint": str(ckpt_path),
        "history": history,
        "spec_dim": int(spec_dim),
        "ir_dim": ir_dim,
        "obj_dim": obj_dim,
        "note": "EnvEncoder never sees DesignObjective; Generator frozen",
    }
    (out_dir / "phase2_train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
