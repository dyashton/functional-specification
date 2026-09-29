"""P1 training: encoder + VQ + surrogate + multi-view contrastive (no generation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.config import load_config
from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import (
    MoleculeGraphDataset,
    available_surrogates,
    collate_graphs,
    compute_y_stats,
    load_split_csv,
)
from functionalspec.models.contrastive import info_nce, multi_view_positive_weight
from functionalspec.models.planner import FunctionalPlanner


def _cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.t()


def _tanimoto_sim_matrix(fp: torch.Tensor) -> torch.Tensor:
    # fp: (B, bits) in {0,1}
    inter = fp @ fp.t()
    cardinal = fp.sum(dim=1, keepdim=True)
    union = cardinal + cardinal.t() - inter
    return inter / union.clamp_min(1.0)


def build_positive_mask(
    y: torch.Tensor,
    fp: torch.Tensor,
    w_desc: float,
    w_ecfp: float,
    topk: int = 3,
) -> torch.Tensor:
    """Positives = top-k multi-view weights per row (exclude self)."""
    sim_desc = _cosine_sim_matrix(y)
    sim_ecfp = _tanimoto_sim_matrix(fp)
    weight = multi_view_positive_weight(
        sim_desc,
        sim_qm=None,
        sim_bind=None,
        sim_interaction=None,
        sim_ecfp=sim_ecfp,
        w_desc=w_desc,
        w_ecfp=w_ecfp,
    )
    b = y.size(0)
    weight = weight - torch.eye(b, device=y.device) * 1e9
    k = min(topk, max(b - 1, 1))
    idx = weight.topk(k, dim=1).indices
    mask = torch.zeros(b, b, dtype=torch.bool, device=y.device)
    mask.scatter_(1, idx, True)
    return mask


def spearman_mean(pred: np.ndarray, target: np.ndarray) -> float:
    rhos = []
    for j in range(target.shape[1]):
        if np.std(target[:, j]) < 1e-8:
            continue
        rho, _ = spearmanr(target[:, j], pred[:, j])
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


@torch.no_grad()
def evaluate(
    model: FunctionalPlanner,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds, targets, flats, fps = [], [], [], []
    total_surr = 0.0
    total_vq = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        ei = batch["edge_index"].to(device)
        ea = batch["edge_attr"].to(device)
        b = batch["batch"].to(device)
        y = batch["y"].to(device)
        out = model.forward_graphs(x, ei, ea, b)
        total_surr += float(F.mse_loss(out["surrogate_pred"], y).item()) * y.size(0)
        total_vq += float(out["vq_loss"].item()) * y.size(0)
        n += y.size(0)
        preds.append(out["surrogate_pred"].cpu().numpy())
        targets.append(y.cpu().numpy())
        flats.append(out["flat_S"].cpu().numpy())
        fps.append(batch["fp"].numpy())
    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(targets, axis=0)
    flat = np.concatenate(flats, axis=0)
    fp = np.concatenate(fps, axis=0)
    # Structure probe: predict a slice of ECFP from S (should stay weak if S is functional)
    probe_y = fp[:, :64]
    ridge = Ridge(alpha=1.0)
    # simple holdout split
    n_te = max(int(0.25 * len(flat)), 1)
    ridge.fit(flat[:-n_te], probe_y[:-n_te])
    struct_r2 = float(ridge.score(flat[-n_te:], probe_y[-n_te:]))
    return {
        "surrogate_mse": total_surr / max(n, 1),
        "vq_loss": total_vq / max(n, 1),
        "surrogate_spearman": spearman_mean(pred, tgt),
        "structure_probe_r2": struct_r2,
        "n": float(n),
    }


def train_p1(
    corpus_a_dir: Path,
    out_dir: Path,
    config_path: Path | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    codebook_size: int | None = None,
    contrastive_weight: float = 0.1,
    topk_pos: int = 3,
    num_workers: int = 0,
    seed: int = 42,
    num_slots: int | None = None,
    commitment_cost: float | None = None,
    w_ecfp: float | None = None,
    no_vq: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = load_config(config_path)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_df = load_split_csv(corpus_a_dir / "train.csv")
    val_df = load_split_csv(corpus_a_dir / "val.csv")
    surr_cols = available_surrogates(train_df, cfg["corpus_a"].get("surrogate_views"))
    if not surr_cols:
        raise RuntimeError("No surrogate columns found in corpus A")

    y_mean, y_std = compute_y_stats(train_df, surr_cols)
    train_ds = MoleculeGraphDataset(train_df, surr_cols, y_mean, y_std)
    val_ds = MoleculeGraphDataset(val_df, surr_cols, y_mean, y_std)
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty after featurization")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )

    mcfg = cfg["model"]
    ccfg = cfg["contrastive"]
    cb = 1 if no_vq else (codebook_size or int(mcfg["primary_codebook"]))
    n_slots = int(num_slots if num_slots is not None else mcfg["num_slots"])
    beta = float(commitment_cost if commitment_cost is not None else mcfg["commitment_cost"])
    w_ecfp_eff = float(ccfg["w_ecfp"] if w_ecfp is None else w_ecfp)
    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=n_slots,
        codebook_size=cb,
        n_surrogates=len(surr_cols),
        commitment_cost=beta,
        no_vq=no_vq,
        with_arm_a=False,
        with_arm_b=False,
    ).to(device_t)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_spear = -1e9
    best_path = out_dir / "p1_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "surr": 0.0, "vq": 0.0, "ctr": 0.0}
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"P1 epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device_t)
            ei = batch["edge_index"].to(device_t)
            ea = batch["edge_attr"].to(device_t)
            b = batch["batch"].to(device_t)
            y = batch["y"].to(device_t)
            fp = batch["fp"].to(device_t)

            out = model.forward_graphs(x, ei, ea, b)
            surr_loss = F.mse_loss(out["surrogate_pred"], y)
            vq_loss = out["vq_loss"]
            if contrastive_weight > 0:
                pos = build_positive_mask(
                    y,
                    fp,
                    w_desc=float(ccfg["w_desc"]),
                    w_ecfp=w_ecfp_eff,
                    topk=topk_pos,
                )
                ctr_loss = info_nce(out["z_proj"], pos, temperature=float(ccfg["temperature"]))
            else:
                ctr_loss = out["z_proj"].new_zeros(())
            loss = surr_loss + vq_loss + contrastive_weight * ctr_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = y.size(0)
            running["loss"] += float(loss.item()) * bs
            running["surr"] += float(surr_loss.item()) * bs
            running["vq"] += float(vq_loss.item()) * bs
            running["ctr"] += float(ctr_loss.item()) * bs
            n_seen += bs
            pbar.set_postfix(loss=float(loss.item()), spear="…")

        val_metrics = evaluate(model, val_loader, device_t)
        row = {
            "epoch": float(epoch),
            "train_loss": running["loss"] / max(n_seen, 1),
            "train_surr": running["surr"] / max(n_seen, 1),
            "train_vq": running["vq"] / max(n_seen, 1),
            "train_ctr": running["ctr"] / max(n_seen, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch}: val_spearman={val_metrics['surrogate_spearman']:.3f} "
            f"struct_R2={val_metrics['structure_probe_r2']:.3f} "
            f"surr_mse={val_metrics['surrogate_mse']:.4f}"
        )

        if val_metrics["surrogate_spearman"] > best_spear:
            best_spear = val_metrics["surrogate_spearman"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "surrogate_cols": surr_cols,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "codebook_size": cb,
                    "num_slots": n_slots,
                    "commitment_cost": beta,
                    "no_vq": no_vq,
                    "contrastive_weight": contrastive_weight,
                    "w_ecfp": w_ecfp_eff,
                    "config": cfg,
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )

    meta = {
        "best_val_spearman": best_spear,
        "best_checkpoint": str(best_path),
        "surrogate_cols": surr_cols,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "codebook_size": cb,
        "num_slots": n_slots,
        "commitment_cost": beta,
        "no_vq": no_vq,
        "contrastive_weight": contrastive_weight,
        "w_ecfp": w_ecfp_eff,
        "history": history,
    }
    (out_dir / "p1_history.json").write_text(json.dumps(meta, indent=2))
    return meta
