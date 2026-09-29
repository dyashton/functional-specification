"""Encoder-side functional neighborhoods: specificity of S-space partitions.

Exact discrete VQ codes are nearly unique on Corpus A, so we study:
  1) soft k-NN balls in continuous flat_S
  2) k-means partitions of flat_S
and compare each ball/cluster to an equal-sized ECFP neighborhood of the same center.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.eval.harness import dump_json
from functionalspec.metrics.diversity import structure_quantiles
from functionalspec.metrics.thresholds import behavioral_variance
from functionalspec.train.p2 import encode_flat_S

PROP_FALLBACK = ["MW", "LogP", "TPSA", "QED", "HBA", "HBD", "nRot"]


def _primary_specificity(h_murcko: float, h_brics: float, v_beh: float, eps: float = 1e-6) -> float:
    """High = structurally diverse + behaviorally specific."""
    return float((h_murcko + h_brics) / (v_beh + eps))


def _v_beh(Y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> float:
    Yz = (Y - y_mean) / y_std
    return behavioral_variance(Yz)


@torch.no_grad()
def embed_corpus(
    checkpoint: Path,
    corpus_a: Path,
    device: torch.device,
    batch_size: int = 64,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    model, ckpt = load_planner_for_embed(checkpoint, device)
    surr_cols = list(ckpt["surrogate_cols"])
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float64)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float64)

    dfs = [load_split_csv(corpus_a / f"{sp}.csv") for sp in splits if (corpus_a / f"{sp}.csv").exists()]
    df = pd.concat(dfs, ignore_index=True)
    ds = MoleculeGraphDataset(df, surr_cols, y_mean, y_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    Ss, Ys, Fps, smiles, codes = [], [], [], [], []
    for batch in tqdm(loader, desc="embed corpus"):
        out = model.forward_graphs(
            batch["x"].to(device),
            batch["edge_index"].to(device),
            batch["edge_attr"].to(device),
            batch["batch"].to(device),
        )
        Ss.append(out["flat_S"].cpu().numpy())
        idx = out["indices"].cpu().numpy()
        for row in idx:
            codes.append(tuple(int(x) for x in row))
        Ys.append(batch["y"].numpy())
        Fps.append(batch["fp"].numpy())
        smiles.extend(batch["smiles"])

    S = np.concatenate(Ss, axis=0)
    Y = np.concatenate(Ys, axis=0)
    fp = np.concatenate(Fps, axis=0)
    # L2-normalize S for cosine NN via dot product
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    return {
        "S": S,
        "Sn": Sn,
        "Y": Y,
        "fp": fp,
        "smiles": smiles,
        "codes": codes,
        "y_mean": np.zeros(Y.shape[1]),  # Y already z-scored in dataset
        "y_std": np.ones(Y.shape[1]),
        "surrogate_cols": surr_cols,
        "n_unique_codes": len(set(codes)),
    }


def _nn_idx_cosine(Sn: np.ndarray, q: int, k: int) -> np.ndarray:
    sims = Sn @ Sn[q]
    sims[q] = -np.inf
    # include self as center of neighborhood
    top = np.argpartition(-sims, min(k - 1, len(sims) - 1))[:k]
    # ensure q is in set
    if q not in top:
        top = np.concatenate([top[:-1], [q]])
    return top


def _nn_idx_tanimoto(fp: np.ndarray, q: int, k: int) -> np.ndarray:
    inter = fp @ fp[q]
    card = fp.sum(axis=1)
    union = card + card[q] - inter
    tani = inter / np.maximum(union, 1e-8)
    tani[q] = -1.0
    top = np.argpartition(-tani, min(k - 1, len(tani) - 1))[:k]
    if q not in top:
        top = np.concatenate([top[:-1], [q]])
    return top


def _metrics_for_idx(
    idx: np.ndarray,
    smiles: list[str],
    Y: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    seed: int,
) -> dict[str, float]:
    smi = [smiles[i] for i in idx]
    struct = structure_quantiles(smi, seed=seed)
    v = _v_beh(Y[idx], y_mean, y_std)
    h_m = float(struct["murcko_entropy"])
    h_b = float(struct["brics_entropy"])
    return {
        **struct,
        "v_beh": v,
        "specificity_R": _primary_specificity(h_m, h_b, v),
        "n_bucket": float(len(idx)),
    }


def run_soft_neighborhoods(
    emb: dict[str, Any],
    k: int = 64,
    n_probes: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(emb["smiles"])
    probes = rng.choice(n, size=min(n_probes, n), replace=False)
    rows = []
    for q in tqdm(probes, desc=f"S/ECFP neighborhoods k={k}"):
        s_idx = _nn_idx_cosine(emb["Sn"], int(q), k)
        e_idx = _nn_idx_tanimoto(emb["fp"], int(q), k)
        s_m = _metrics_for_idx(s_idx, emb["smiles"], emb["Y"], emb["y_mean"], emb["y_std"], seed)
        e_m = _metrics_for_idx(e_idx, emb["smiles"], emb["Y"], emb["y_mean"], emb["y_std"], seed)
        rows.append(
            {
                "probe_idx": int(q),
                "probe_smiles": emb["smiles"][int(q)],
                "k": k,
                "S_neighborhood": s_m,
                "ECFP_neighborhood": e_m,
                "delta_specificity_R": float(s_m["specificity_R"] - e_m["specificity_R"]),
                "delta_v_beh": float(s_m["v_beh"] - e_m["v_beh"]),
                "delta_murcko_entropy": float(s_m["murcko_entropy"] - e_m["murcko_entropy"]),
            }
        )
    return rows


def run_kmeans_partitions(
    emb: dict[str, Any],
    n_clusters: int = 100,
    min_size: int = 16,
    seed: int = 42,
) -> list[dict[str, Any]]:
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init=3)
    labels = km.fit_predict(emb["Sn"])
    rows = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) < min_size:
            continue
        m = _metrics_for_idx(idx, emb["smiles"], emb["Y"], emb["y_mean"], emb["y_std"], seed)
        # ECFP control: take centroid molecule's ECFP-NN of same size
        # use cluster medoid in S
        center = emb["Sn"][idx].mean(axis=0)
        center /= np.linalg.norm(center) + 1e-8
        sims = emb["Sn"][idx] @ center
        medoid_local = int(idx[int(np.argmax(sims))])
        e_idx = _nn_idx_tanimoto(emb["fp"], medoid_local, k=len(idx))
        e_m = _metrics_for_idx(e_idx, emb["smiles"], emb["Y"], emb["y_mean"], emb["y_std"], seed)
        rows.append(
            {
                "cluster_id": int(c),
                "S_partition": m,
                "ECFP_neighborhood": e_m,
                "delta_specificity_R": float(m["specificity_R"] - e_m["specificity_R"]),
                "delta_v_beh": float(m["v_beh"] - e_m["v_beh"]),
            }
        )
    return rows


def summarize_specificity(rows: list[dict[str, Any]], s_key: str) -> dict[str, Any]:
    Rs = np.array([r[s_key]["specificity_R"] for r in rows], dtype=float)
    vs = np.array([r[s_key]["v_beh"] for r in rows], dtype=float)
    hs = np.array([r[s_key]["murcko_entropy"] for r in rows], dtype=float)
    dR = np.array([r["delta_specificity_R"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "specificity_R_mean": float(np.nanmean(Rs)),
        "specificity_R_median": float(np.nanmedian(Rs)),
        "v_beh_mean": float(np.nanmean(vs)),
        "murcko_entropy_mean": float(np.nanmean(hs)),
        "frac_S_higher_R_than_ECFP": float(np.mean(dR > 0)),
        "delta_R_mean": float(np.nanmean(dR)),
        "R_percentiles": {str(p): float(np.nanpercentile(Rs, p)) for p in (10, 25, 50, 75, 90)},
    }


def _code_usage_stats(codes: list, codebook_size: int | None, no_vq: bool) -> dict[str, Any]:
    from collections import Counter

    code_sizes = np.array(list(Counter(codes).values()), dtype=np.float64) if codes else np.array([])
    n_unique = len(set(codes))
    n_mols = len(codes)
    # occupancy entropy over observed code multisets
    if len(code_sizes) and code_sizes.sum() > 0:
        p = code_sizes / code_sizes.sum()
        occ_H = float(-(p * np.log(p + 1e-12)).sum())
    else:
        occ_H = 0.0
    dead = None
    if codebook_size and not no_vq and codebook_size > 0:
        # slot-wise: approx dead rate unknown without per-slot histograms; report unused fraction of slot-tuples
        dead = max(0.0, 1.0 - n_unique / float(max(codebook_size**2, 1)))  # loose; prefer unique ratio
    return {
        "n_mols": n_mols,
        "n_unique_codes": n_unique,
        "n_codes_size_ge_2": int((code_sizes >= 2).sum()) if len(code_sizes) else 0,
        "n_codes_size_ge_8": int((code_sizes >= 8).sum()) if len(code_sizes) else 0,
        "max_code_size": int(code_sizes.max()) if len(code_sizes) else 0,
        "occupancy_entropy": occ_H,
        "unique_code_fraction": float(n_unique / max(n_mols, 1)),
        "codebook_size": codebook_size,
        "no_vq": no_vq,
        "dead_code_proxy": dead,
        "note": "Exact VQ codes are often nearly unique — soft neighborhoods / k-means are the analysis objects.",
    }


def _residualize_y_on_logp_tpsa_mw(Y: np.ndarray, cols: list[str]) -> np.ndarray:
    """OLS-residualize each surrogate column on LogP/TPSA/MW (z-scored Y already)."""
    name_to_i = {c: i for i, c in enumerate(cols)}
    bases = [name_to_i[c] for c in ("LogP", "TPSA", "MW") if c in name_to_i]
    if not bases:
        return Y.copy()
    X = Y[:, bases]
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    Y_res = Y.astype(np.float64).copy()
    for j in range(Y.shape[1]):
        if j in bases:
            # residualize non-base dims primarily; zero base dims' residual contribution to v_beh
            Y_res[:, j] = 0.0
            continue
        y = Y[:, j]
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            Y_res[:, j] = y - X @ beta
        except Exception:
            Y_res[:, j] = y
    return Y_res


def run_spec_specificity(
    checkpoint: Path,
    corpus_a: Path,
    out_dir: Path,
    k_list: tuple[int, ...] = (32, 64, 128),
    n_probes: int = 200,
    n_clusters: int = 100,
    min_cluster_size: int = 16,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 42,
    residualize_logp_tpsa_mw: bool = False,
) -> dict[str, Any]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = embed_corpus(checkpoint, corpus_a, device_t, batch_size=batch_size)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    no_vq = bool(ckpt.get("no_vq", False))
    cb = ckpt.get("codebook_size")
    discrete_occupancy = _code_usage_stats(emb["codes"], int(cb) if cb is not None else None, no_vq)
    print("Discrete code occupancy:", discrete_occupancy)

    if residualize_logp_tpsa_mw:
        emb = dict(emb)
        emb["Y"] = _residualize_y_on_logp_tpsa_mw(emb["Y"], list(emb["surrogate_cols"]))

    soft = {}
    soft_summaries = {}
    for k in k_list:
        rows = run_soft_neighborhoods(emb, k=k, n_probes=n_probes, seed=seed)
        soft[str(k)] = rows
        soft_summaries[str(k)] = summarize_specificity(rows, "S_neighborhood")
        suffix = "_resid" if residualize_logp_tpsa_mw else ""
        dump_json(rows, out_dir / f"soft_neighborhoods_k{k}{suffix}.json")
        pd.DataFrame(
            [
                {
                    "k": k,
                    "probe_smiles": r["probe_smiles"],
                    "H_murcko_S": r["S_neighborhood"]["murcko_entropy"],
                    "H_brics_S": r["S_neighborhood"]["brics_entropy"],
                    "v_beh_S": r["S_neighborhood"]["v_beh"],
                    "R_S": r["S_neighborhood"]["specificity_R"],
                    "H_murcko_ECFP": r["ECFP_neighborhood"]["murcko_entropy"],
                    "H_brics_ECFP": r["ECFP_neighborhood"]["brics_entropy"],
                    "v_beh_ECFP": r["ECFP_neighborhood"]["v_beh"],
                    "R_ECFP": r["ECFP_neighborhood"]["specificity_R"],
                    "delta_R": r["delta_specificity_R"],
                }
                for r in rows
            ]
        ).to_csv(out_dir / f"soft_neighborhoods_k{k}{suffix}.csv", index=False)
        print(f"k={k}: {soft_summaries[str(k)]}")

    partitions = run_kmeans_partitions(
        emb, n_clusters=n_clusters, min_size=min_cluster_size, seed=seed
    )
    part_summary = summarize_specificity(partitions, "S_partition")
    dump_json(partitions, out_dir / ("kmeans_partitions_resid.json" if residualize_logp_tpsa_mw else "kmeans_partitions.json"))
    pd.DataFrame(
        [
            {
                "cluster_id": r["cluster_id"],
                "n": r["S_partition"]["n_bucket"],
                "H_murcko_S": r["S_partition"]["murcko_entropy"],
                "H_brics_S": r["S_partition"]["brics_entropy"],
                "v_beh_S": r["S_partition"]["v_beh"],
                "R_S": r["S_partition"]["specificity_R"],
                "H_murcko_ECFP": r["ECFP_neighborhood"]["murcko_entropy"],
                "v_beh_ECFP": r["ECFP_neighborhood"]["v_beh"],
                "R_ECFP": r["ECFP_neighborhood"]["specificity_R"],
                "delta_R": r["delta_specificity_R"],
            }
            for r in partitions
        ]
    ).to_csv(out_dir / ("kmeans_partitions_resid.csv" if residualize_logp_tpsa_mw else "kmeans_partitions.csv"), index=False)
    print("k-means:", part_summary)

    report = {
        "checkpoint": str(checkpoint),
        "residualize_logp_tpsa_mw": residualize_logp_tpsa_mw,
        "discrete_code_occupancy": discrete_occupancy,
        "code_usage": discrete_occupancy,
        "definition": {
            "specificity_R": "(H_murcko + H_brics) / (v_beh + eps)",
            "interpretation": "High R = structurally diverse + behaviorally specific neighborhood",
            "note": "Continuous quantity; categories deferred until distribution is inspected",
        },
        "soft_neighborhood_summaries": soft_summaries,
        "kmeans_summary": part_summary,
        "artifacts": {
            "soft_csvs": [
                str(out_dir / f"soft_neighborhoods_k{k}{'_resid' if residualize_logp_tpsa_mw else ''}.csv")
                for k in k_list
            ],
            "kmeans_csv": str(
                out_dir / ("kmeans_partitions_resid.csv" if residualize_logp_tpsa_mw else "kmeans_partitions.csv")
            ),
        },
    }
    summary_name = "specificity_summary_resid.json" if residualize_logp_tpsa_mw else "specificity_summary.json"
    dump_json(report, out_dir / summary_name)
    np.savez_compressed(
        out_dir / ("embeddings_resid.npz" if residualize_logp_tpsa_mw else "embeddings.npz"),
        S=emb["S"],
        Y=emb["Y"],
        fp=emb["fp"],
        smiles=np.array(emb["smiles"], dtype=object),
    )
    print(f"→ {out_dir / summary_name}")
    return report
