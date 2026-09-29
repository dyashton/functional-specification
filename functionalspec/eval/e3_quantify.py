"""Quantify E3 structural entropy vs Recon-VQ / corpus controls; label C1/C2/C3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from functionalspec.data.descriptors import SURROGATE_ALWAYS, compute_surrogates, ecfp_bits
from functionalspec.data.graph_dataset import load_split_csv
from functionalspec.eval.harness import dump_json
from functionalspec.metrics.diversity import structure_quantiles
from functionalspec.metrics.thresholds import THRESHOLDS, behavioral_variance
from functionalspec.models.baselines import ReconVQBottleneck

PROP_COLS = list(SURROGATE_ALWAYS) + ["HallKierAlpha"]


def _v_beh_from_df(df: pd.DataFrame, cols: list[str], y_mean: np.ndarray, y_std: np.ndarray) -> float:
    use = [c for c in cols if c in df.columns]
    if not use:
        return float("nan")
    Y = df[use].to_numpy(dtype=np.float64)
    idx = [PROP_COLS.index(c) for c in use if c in PROP_COLS]
    if len(idx) != len(use):
        mu, sd = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return behavioral_variance((Y - mu) / sd)
    return behavioral_variance((Y - y_mean[idx]) / y_std[idx])


def _classify_world(
    v_beh: float,
    n_scaf: float,
    top_scaf: float,
    valid: float,
    unique: float,
    passed: bool,
    matched: bool | None,
) -> str:
    if valid < 0.8 or unique < 0.4:
        return "A"
    # C1 before B: runaway sets can still be benzene-heavy without being motif codes
    if v_beh > 1.0:
        return "C1"
    if (n_scaf < 30 or top_scaf > 0.5) and v_beh <= THRESHOLDS.e3_beh_var_pass:
        return "B"
    if v_beh > THRESHOLDS.e3_beh_var_pass:
        return "C2"
    if not passed and matched is False:
        return "C2"
    if not passed:
        return "C2"
    return "OK"


def _corpus_pool(corpus_a: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    dfs = []
    for split in ("train", "val"):
        p = corpus_a / f"{split}.csv"
        if p.exists():
            dfs.append(load_split_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    smiles, fps, rows = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="corpus fps", leave=False):
        s = str(row["SMILES"])
        fp = ecfp_bits(s)
        if fp is None:
            continue
        vals = []
        ok = True
        for c in PROP_COLS:
            if c in row.index and pd.notna(row[c]):
                vals.append(float(row[c]))
            else:
                d = compute_surrogates(s)
                if d is None or c not in d or not np.isfinite(d[c]):
                    ok = False
                    break
                vals.append(float(d[c]))
        if not ok:
            continue
        smiles.append(s)
        fps.append(fp.astype(np.float32))
        rows.append(vals)
    return smiles, np.stack(fps), np.asarray(rows, dtype=np.float64)


def _train_recon(
    pool_fp: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> ReconVQBottleneck:
    torch.manual_seed(seed)
    model = ReconVQBottleneck(in_dim=pool_fp.shape[1], codebook_size=128, dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.from_numpy(pool_fp).float()
    n = x.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            xb = x[perm[i : i + 256]].to(device)
            out = model(xb)
            loss = F.mse_loss(out["recon"], xb) + out["vq_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _embed_fps(model: ReconVQBottleneck, fps: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(fps).float()
    zs = []
    for i in range(0, len(fps), 256):
        zs.append(model(x[i : i + 256].to(device))["z"].cpu().numpy())
    return np.concatenate(zs, axis=0)


def _nn_indices(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(pool - query[None, :], axis=1)
    return np.argsort(d)[:k]


def quantify_e3(
    e3_dir: Path,
    corpus_a: Path,
    out_path: Path | None = None,
    recon_epochs: int = 20,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    summary = json.loads((e3_dir / "e3_summary.json").read_text())
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Building corpus pool for Recon-VQ / behavior controls...")
    pool_smi, pool_fp, pool_Y = _corpus_pool(corpus_a)
    y_mean = pool_Y.mean(axis=0)
    y_std = np.where(pool_Y.std(axis=0) < 1e-8, 1.0, pool_Y.std(axis=0))

    print(f"Training Recon-VQ on corpus ECFP ({recon_epochs} epochs, n={len(pool_smi)})...")
    model = _train_recon(pool_fp, recon_epochs, device_t, seed)
    z_pool = _embed_fps(model, pool_fp, device_t)

    rows_out: list[dict[str, Any]] = []
    for r in summary["per_spec"]:
        sid = r["spec_id"]
        src = r["source_smiles"]
        df = pd.read_csv(r["samples_csv"])
        gen_smi = df["smiles"].astype(str).tolist()
        n_gen = len(gen_smi)

        ours_struct = structure_quantiles(gen_smi, seed=seed)
        ours_v = float(r.get("ours", {}).get("v_beh_global_z", r.get("ours", {}).get("v_beh", np.nan)))
        if not np.isfinite(ours_v):
            ours_v = _v_beh_from_df(df, PROP_COLS, y_mean, y_std)

        e0 = r.get("e0", {})
        world = _classify_world(
            ours_v,
            ours_struct["n_scaffolds"],
            ours_struct["top_scaffold_frac"],
            float(e0.get("validity", 1.0)),
            float(e0.get("unique_rate", 1.0)),
            bool(r.get("pass")),
            r.get("behavior_matched"),
        )

        # Recon-VQ NN control
        fp_src = ecfp_bits(src)
        if fp_src is None:
            recon_smi, recon_idx = [], np.array([], dtype=int)
        else:
            z_src = _embed_fps(model, fp_src.astype(np.float32)[None, :], device_t)[0]
            recon_idx = _nn_indices(z_src, z_pool, k=n_gen)
            recon_smi = [pool_smi[i] for i in recon_idx]
        recon_struct = structure_quantiles(recon_smi, seed=seed) if recon_smi else {}
        if len(recon_idx):
            recon_df = pd.DataFrame(
                {"smiles": recon_smi, **{c: pool_Y[recon_idx, j] for j, c in enumerate(PROP_COLS)}}
            )
            recon_v = _v_beh_from_df(recon_df, PROP_COLS, y_mean, y_std)
        else:
            recon_v = float("nan")

        # Behavior-NN corpus control (nearest to generated set mean in z-scored surrogates)
        use_cols = [c for c in PROP_COLS if c in df.columns]
        gen_mean = df[use_cols].to_numpy(dtype=np.float64).mean(axis=0)
        col_idx = [PROP_COLS.index(c) for c in use_cols]
        pool_z = (pool_Y[:, col_idx] - y_mean[col_idx]) / y_std[col_idx]
        tgt = (gen_mean - y_mean[col_idx]) / y_std[col_idx]
        b_idx = np.argsort(np.linalg.norm(pool_z - tgt[None, :], axis=1))[:n_gen]
        base_smi = [pool_smi[i] for i in b_idx]
        base_struct = structure_quantiles(base_smi, seed=seed)
        base_df = pd.DataFrame(
            {"smiles": base_smi, **{c: pool_Y[b_idx, PROP_COLS.index(c)] for c in PROP_COLS}}
        )
        base_v = _v_beh_from_df(base_df, PROP_COLS, y_mean, y_std)

        def delta(a: dict, b: dict, key: str) -> float | None:
            if key not in a or key not in b:
                return None
            return float(a[key] - b[key])

        row = {
            "spec_id": sid,
            "source_smiles": src,
            "pass": r.get("pass"),
            "behavior_matched": r.get("behavior_matched"),
            "world": world,
            "ours": {
                "v_beh": ours_v,
                **ours_struct,
                "delta_murcko_entropy_vs_recon": delta(ours_struct, recon_struct, "murcko_entropy"),
                "delta_brics_entropy_vs_recon": delta(ours_struct, recon_struct, "brics_entropy"),
                "delta_nn_tanimoto_vs_recon": delta(ours_struct, recon_struct, "mean_nn_tanimoto"),
            },
            "recon_vq_nn": {"v_beh": recon_v, **recon_struct},
            "corpus_behavior_nn": {"v_beh": base_v, **base_struct},
            "reasons": r.get("reasons"),
        }
        rows_out.append(row)
        print(
            f"spec {sid} world={world} "
            f"H_murcko={ours_struct['murcko_entropy']:.2f} "
            f"(recon {recon_struct.get('murcko_entropy', float('nan')):.2f}) "
            f"H_brics={ours_struct['brics_entropy']:.2f} "
            f"(recon {recon_struct.get('brics_entropy', float('nan')):.2f}) "
            f"NN_T={ours_struct['mean_nn_tanimoto']:.3f} "
            f"(recon {recon_struct.get('mean_nn_tanimoto', float('nan')):.3f}) "
            f"v_beh={ours_v:.3f}"
        )

    out = {
        "e3_dir": str(e3_dir),
        "n_corpus": len(pool_smi),
        "recon_epochs": recon_epochs,
        "world_legend": {
            "A": "weak decoder (validity/uniqueness)",
            "B": "motif/descriptor cluster (few scaffolds)",
            "C1": "underconstrained — behavior explodes",
            "C2": "imprecise spec — mild drift / unmatched gate",
            "C3": "decoder ignores S (not observed here)",
            "OK": "tight behavior + diverse scaffolds",
        },
        "specs": rows_out,
    }
    out_path = out_path or (e3_dir / "e3_quantify.json")
    dump_json(out, out_path)
    print(f"→ {out_path}")
    return out
