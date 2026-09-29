"""E2 runner: surrogate vs structure probes on frozen Functional Spec S."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.eval.harness import dump_json
from functionalspec.metrics.e2_probes import probe_multiregression
from functionalspec.metrics.thresholds import THRESHOLDS
from functionalspec.models.baselines import ReconVQBottleneck
from functionalspec.models.planner import FunctionalPlanner
from functionalspec.train.p2 import encode_flat_S


def probe_ecfp_bits(X: np.ndarray, bits: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Mean AUROC over binary ECFP bits via Ridge scores (fast; skip near-constant bits)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    aurocs = []
    for j in range(bits.shape[1]):
        y = bits[:, j].astype(np.float64)
        if y.min() == y.max():
            continue
        if min(int((y == 0).sum()), int((y == 1).sum())) < 8:
            continue
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y.astype(int)
            )
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
        if len(np.unique(yte)) < 2:
            continue
        pred = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
        aurocs.append(float(roc_auc_score(yte, pred)))
    mean_auroc = float(np.nanmean(aurocs)) if aurocs else 0.0
    return {
        "mean_auroc": mean_auroc,
        "n_bits": float(len(aurocs)),
        # Alias used for margin vs thresholds (higher = more structure info)
        "mean_spearman": mean_auroc,
    }


def load_planner_for_embed(ckpt_path: Path, device: torch.device) -> tuple[FunctionalPlanner, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    surr_cols = ckpt["surrogate_cols"]
    cb = int(ckpt.get("codebook_size", mcfg["primary_codebook"]))
    n_slots = int(ckpt.get("num_slots", mcfg["num_slots"]))
    beta = float(ckpt.get("commitment_cost", mcfg["commitment_cost"]))
    no_vq = bool(ckpt.get("no_vq", False))
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
    )
    # P2 checkpoints include arm_a keys — ignore them
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def embed_split(
    model: FunctionalPlanner,
    ds: MoleculeGraphDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns flat_S, y (z-scored surrogates), fp."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    Ss, Ys, Fps = [], [], []
    for batch in tqdm(loader, desc="embed", leave=False):
        flat_S = encode_flat_S(model, batch, device)
        Ss.append(flat_S.cpu().numpy())
        Ys.append(batch["y"].numpy())
        Fps.append(batch["fp"].numpy())
    return np.concatenate(Ss), np.concatenate(Ys), np.concatenate(Fps)


def train_recon_vq(
    fp: np.ndarray,
    codebook_size: int = 128,
    dim: int = 256,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Train fingerprint autoencoder VQ; return latent z for all rows."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    x = torch.from_numpy(fp).float()
    model = ReconVQBottleneck(in_dim=fp.shape[1], codebook_size=codebook_size, dim=dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = x.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x[idx].to(device)
            out = model(xb)
            loss = F.mse_loss(out["recon"], xb) + out["vq_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x[i : i + batch_size].to(device)
            zs.append(model(xb)["z"].cpu().numpy())
    return np.concatenate(zs, axis=0)


def nn_tanimoto_test(S: np.ndarray, fp: np.ndarray, n_query: int = 100, seed: int = 0) -> dict[str, float]:
    """Mean FP-Tanimoto of S-space NN vs FP-space NN (exclude self)."""
    rng = np.random.default_rng(seed)
    n = len(S)
    qs = rng.choice(n, size=min(n_query, n), replace=False)
    # normalize S for cosine
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)

    def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
        inter = float(np.minimum(a, b).sum())
        union = float(np.maximum(a, b).sum())
        return inter / union if union > 0 else 0.0

    s_sims, fp_sims = [], []
    for q in qs:
        # S NN
        sims = Sn @ Sn[q]
        sims[q] = -1e9
        j = int(np.argmax(sims))
        s_sims.append(tanimoto(fp[q], fp[j]))
        # FP NN
        inter = fp @ fp[q]
        card = fp.sum(axis=1)
        union = card + card[q] - inter
        tani = inter / np.maximum(union, 1e-8)
        tani[q] = -1.0
        k = int(np.argmax(tani))
        fp_sims.append(float(tani[k]))
    return {
        "mean_tanimoto_S_NN": float(np.mean(s_sims)),
        "mean_tanimoto_FP_NN": float(np.mean(fp_sims)),
        "margin": float(np.mean(fp_sims) - np.mean(s_sims)),
    }


def run_e2(
    checkpoint: Path,
    corpus_a_dir: Path,
    out_dir: Path,
    split: str = "val",
    batch_size: int = 64,
    fp_bits_probe: int = 128,
    recon_epochs: int = 30,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_planner_for_embed(checkpoint, device_t)
    surr_cols = ckpt["surrogate_cols"]
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float64)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float64)

    df = load_split_csv(corpus_a_dir / f"{split}.csv")
    ds = MoleculeGraphDataset(df, surr_cols, y_mean, y_std)
    S, Y, fp = embed_split(model, ds, device_t, batch_size=batch_size)

    # Structure target: first fp_bits_probe Morgan bits (binary → AUROC)
    Y_struct = (fp[:, :fp_bits_probe] > 0).astype(np.int64)

    surr = probe_multiregression(S, Y, seed=seed)
    ps_ours = probe_ecfp_bits(S, Y_struct, seed=seed)
    gap = float(surr["mean_spearman"] - ps_ours["mean_spearman"])
    report: dict[str, Any] = {
        "surrogate_probe": surr,
        "structure_probe": ps_ours,
        "gap_spearman": gap,
        "pass": gap >= THRESHOLDS.e2_gap_pass,
    }

    # Recon-VQ control on fingerprints
    print(f"Training Recon-VQ control on ECFP ({recon_epochs} epochs)...")
    z_recon = train_recon_vq(
        fp,
        codebook_size=int(ckpt.get("codebook_size", 128)),
        epochs=recon_epochs,
        device=device_t,
        seed=seed,
    )
    ps_recon = probe_ecfp_bits(z_recon, Y_struct, seed=seed)

    # FP-PCA control: strong structure baseline without relying on VQ training quality
    n_pca = min(128, fp.shape[0] - 1, fp.shape[1])
    z_fp = PCA(n_components=n_pca, random_state=seed).fit_transform(fp)
    ps_fp = probe_ecfp_bits(z_fp, Y_struct, seed=seed)

    struct_control = max(ps_recon["mean_spearman"], ps_fp["mean_spearman"])
    structure_margin = float(struct_control - ps_ours["mean_spearman"])

    report["recon_structure_probe"] = ps_recon
    report["fp_pca_structure_probe"] = ps_fp
    report["structure_control_spearman"] = struct_control
    report["structure_margin"] = structure_margin
    report["pass"] = bool(
        gap >= THRESHOLDS.e2_gap_pass
        and structure_margin >= THRESHOLDS.e2_structure_margin_vs_recon
    )

    # FP → surrogate baseline (how predictable are properties from structure alone)
    fp_surr = probe_multiregression(fp[:, :fp_bits_probe].astype(np.float64), Y, seed=seed)
    report["fp_surrogate_probe"] = fp_surr

    nn = nn_tanimoto_test(S, fp, seed=seed)
    report["nn_test"] = nn
    report["nn_pass"] = bool(nn["margin"] >= THRESHOLDS.e2_nn_tanimoto_margin)

    report["meta"] = {
        "checkpoint": str(checkpoint),
        "split": split,
        "n": int(len(S)),
        "surrogate_cols": surr_cols,
        "fp_bits_probe": fp_bits_probe,
        "thresholds": {
            "gap_pass": THRESHOLDS.e2_gap_pass,
            "structure_margin": THRESHOLDS.e2_structure_margin_vs_recon,
            "nn_margin": THRESHOLDS.e2_nn_tanimoto_margin,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report, out_dir / "e2_summary.json")
    np.savez_compressed(out_dir / "e2_arrays.npz", S=S, Y=Y, fp=fp, z_recon=z_recon)
    return report
