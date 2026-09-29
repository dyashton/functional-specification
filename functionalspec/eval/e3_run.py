"""E3 runner: freeze S*, sample many molecules, score R vs matched-behavior corpus baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.data.descriptors import SURROGATE_ALWAYS, compute_surrogates
from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.data.smiles_tokenizer import MoleculeTokenizer
from functionalspec.eval.harness import dump_json, e0_report, e3_report
from functionalspec.metrics.thresholds import THRESHOLDS, behavior_match_ok
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.planner import FunctionalPlanner
from functionalspec.train.p2 import encode_flat_S


def load_p2(ckpt_path: Path, device: torch.device) -> tuple[FunctionalPlanner, MoleculeTokenizer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    tok = MoleculeTokenizer.from_dict(ckpt["tokenizer"])
    surr_cols = ckpt["surrogate_cols"]
    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=int(mcfg["num_slots"]),
        codebook_size=int(ckpt["codebook_size"]),
        n_surrogates=len(surr_cols),
        commitment_cost=float(mcfg["commitment_cost"]),
        with_arm_a=False,
        with_arm_b=False,
    )
    cond_dim = int(mcfg["num_slots"]) * int(mcfg["hidden_dim"])
    model.arm_a = SmilesConditionedDecoder(
        cond_dim=cond_dim,
        vocab_size=tok.vocab_size,
        hidden=int(ckpt.get("arm_hidden", 512)),
        num_layers=int(ckpt.get("arm_layers", 2)),
        cond_dropout=float(ckpt.get("cond_dropout", 0.0)),
    )
    # FiLM decoder may add keys absent from older Arm A checkpoints
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model, tok, ckpt


def surrogates_for_smiles(smiles: list[str], cols: list[str]) -> tuple[list[str], np.ndarray]:
    kept_s: list[str] = []
    rows: list[list[float]] = []
    for s in smiles:
        d = compute_surrogates(s)
        if d is None:
            continue
        vals = []
        ok = True
        for c in cols:
            if c not in d or not np.isfinite(d[c]):
                ok = False
                break
            vals.append(float(d[c]))
        if not ok:
            continue
        kept_s.append(s)
        rows.append(vals)
    if not rows:
        return [], np.zeros((0, len(cols)), dtype=np.float64)
    return kept_s, np.asarray(rows, dtype=np.float64)


def e3_surrogate_cols(ckpt_cols: list[str]) -> list[str]:
    """Only use descriptors we can recompute for generated molecules."""
    always = set(SURROGATE_ALWAYS) | {"HallKierAlpha"}
    cols = [c for c in ckpt_cols if c in always]
    return cols or list(SURROGATE_ALWAYS)


def zscore(Y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Y - mean) / std


def matched_corpus_baseline(
    pool_smiles: list[str],
    pool_Y: np.ndarray,
    target_mean_z: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n: int,
    tol: float,
) -> tuple[list[str], np.ndarray, bool]:
    """Pick up to n corpus molecules closest to target_mean_z; require set-mean match."""
    if len(pool_smiles) == 0:
        return [], np.zeros((0, len(target_mean_z))), False
    pool_z = zscore(pool_Y, y_mean, y_std)
    d = np.linalg.norm(pool_z - target_mean_z, axis=1) / np.sqrt(len(target_mean_z))
    order = np.argsort(d)
    # take nearest n, then shrink until mean matches (or keep nearest n and report match flag)
    take = order[: min(n, len(order))]
    # grow/shrink to satisfy match if possible
    best = take
    matched = False
    for k in range(min(32, len(take)), len(take) + 1):
        idx = take[:k]
        Y = pool_Y[idx]
        ok = behavior_match_ok(zscore(Y, y_mean, y_std).mean(0), target_mean_z, tol=tol)
        if ok:
            best = idx
            matched = True
            break
    # if never matched, still return nearest n and flag False
    if not matched:
        best = take[:n] if len(take) >= n else take
        Y = pool_Y[best]
        matched = behavior_match_ok(zscore(Y, y_mean, y_std).mean(0), target_mean_z, tol=tol)
    return [pool_smiles[i] for i in best], pool_Y[best], matched


@torch.no_grad()
def sample_from_flat_S(
    model: FunctionalPlanner,
    tok: MoleculeTokenizer,
    flat_S: torch.Tensor,
    n_samples: int,
    batch_size: int,
    max_len: int,
    temperature: float,
) -> list[str]:
    assert model.arm_a is not None
    out: list[str] = []
    # flat_S: (1, dim)
    while len(out) < n_samples:
        b = min(batch_size, n_samples - len(out))
        cond = flat_S.expand(b, -1)
        sampled = model.arm_a.sample(cond, max_len=max_len, temperature=temperature)
        for row in sampled:
            smi = tok.decode_to_smiles(row)
            if smi:
                out.append(smi)
    return out[:n_samples]


def run_e3(
    checkpoint: Path,
    corpus_a_dir: Path,
    out_dir: Path,
    n_specs: int = 5,
    n_samples: int = 1000,
    split: str = "val",
    batch_size: int = 64,
    temperature: float = 1.0,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, ckpt = load_p2(checkpoint, device_t)
    surr_cols = e3_surrogate_cols(list(ckpt["surrogate_cols"]))
    # Align y_mean/y_std to the reduced column set
    full_cols = list(ckpt["surrogate_cols"])
    y_mean_full = np.asarray(ckpt["y_mean"], dtype=np.float64)
    y_std_full = np.asarray(ckpt["y_std"], dtype=np.float64)
    col_idx = [full_cols.index(c) for c in surr_cols]
    y_mean = y_mean_full[col_idx]
    y_std = y_std_full[col_idx]
    max_len = int(ckpt.get("max_len", 150 if tok.representation == "selfies" else 120))

    df = load_split_csv(corpus_a_dir / f"{split}.csv")
    # pool for matched baseline = full train+val with surrogate cols
    pool_df = pd.concat(
        [load_split_csv(corpus_a_dir / "train.csv"), load_split_csv(corpus_a_dir / "val.csv")],
        ignore_index=True,
    )
    pool_smiles = pool_df["SMILES"].astype(str).tolist()
    pool_Y = pool_df[surr_cols].to_numpy(dtype=np.float64)

    # choose source molecules for S*
    # Dataset still needs full P1 surrogate columns for graph loading
    ds_cols = list(ckpt["surrogate_cols"])
    ds = MoleculeGraphDataset(df, ds_cols, np.asarray(ckpt["y_mean"], dtype=np.float64), np.asarray(ckpt["y_std"], dtype=np.float64))
    if len(ds) == 0:
        raise RuntimeError("Empty dataset for E3 sources")
    idxs = rng.choice(len(ds), size=min(n_specs, len(ds)), replace=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_spec: list[dict[str, Any]] = []

    for si, idx in enumerate(tqdm(idxs, desc="E3 specs")):
        item = ds[int(idx)]
        batch = collate_graphs([item])
        flat_S = encode_flat_S(model, batch, device_t)
        source = item["smiles"]

        samples = sample_from_flat_S(
            model,
            tok,
            flat_S,
            n_samples=n_samples,
            batch_size=batch_size,
            max_len=max_len,
            temperature=temperature,
        )
        kept, Y = surrogates_for_smiles(samples, surr_cols)
        e0 = e0_report(samples)

        if len(kept) < 10:
            report = {
                "spec_id": int(si),
                "source_smiles": source,
                "e0": e0,
                "n_kept": len(kept),
                "pass": False,
                "reasons": ["too few valid surrogate-labeled samples"],
            }
            per_spec.append(report)
            continue

        Yz = zscore(Y, y_mean, y_std)
        target_mean_z = Yz.mean(axis=0)
        base_smi, base_Y, matched_flag = matched_corpus_baseline(
            pool_smiles,
            pool_Y,
            target_mean_z,
            y_mean,
            y_std,
            n=len(kept),
            tol=THRESHOLDS.behavior_match_tol,
        )
        # e3_report uses raw surrogates then zscores internally via e3_metrics
        # For fair R, pass z-scored matrices into e3_metrics path — e3_report calls e3_metrics which zscores again.
        # So pass RAW Y for both; internal zscore is per-set. Better pass already comparable:
        # Use e3_metrics on z-scored by global stats by feeding Yz and zscored baseline.
        from functionalspec.metrics.diversity import e3_metrics
        from functionalspec.metrics.thresholds import diversity_behavior_ratio, e3_pass, structural_diversity

        # Custom metrics with global z-scoring for behavior variance
        m_ours = e3_metrics(kept, Y)
        # Override v_beh / R using global z-score variance (more comparable across sets)
        v_beh = float(np.mean(np.var(Yz, axis=0)))
        m_ours["v_beh_global_z"] = v_beh
        m_ours["R_global_z"] = diversity_behavior_ratio(m_ours["d_struct"], v_beh)

        if len(base_smi) > 0:
            base_Yz = zscore(base_Y, y_mean, y_std)
            m_base = e3_metrics(base_smi, base_Y)
            v_base = float(np.mean(np.var(base_Yz, axis=0)))
            m_base["v_beh_global_z"] = v_base
            m_base["R_global_z"] = diversity_behavior_ratio(m_base["d_struct"], v_base)
            ok, reasons = e3_pass(
                n_scaffolds=int(m_ours["n_scaffolds"]),
                v_beh=v_beh,
                r=float(m_ours["R_global_z"]),
                r_base=float(m_base["R_global_z"]),
                n_scaf_base=int(m_base["n_scaffolds"]),
            )
            # require behavior match of set means in z-space
            matched = behavior_match_ok(target_mean_z, base_Yz.mean(0), tol=THRESHOLDS.behavior_match_tol)
            report = {
                "spec_id": int(si),
                "source_smiles": source,
                "e0": e0,
                "ours": m_ours,
                "baseline": m_base,
                "behavior_matched": bool(matched or matched_flag),
                "pass": bool(ok and (matched or matched_flag)),
                "reasons": reasons if (matched or matched_flag) else ["behavior not matched to corpus baseline"],
            }
        else:
            report = {
                "spec_id": int(si),
                "source_smiles": source,
                "e0": e0,
                "ours": m_ours,
                "pass": False,
                "reasons": ["empty baseline"],
            }

        # save samples
        spec_csv = out_dir / f"spec_{si:02d}_samples.csv"
        pd.DataFrame({"smiles": kept, **{c: Y[:, j] for j, c in enumerate(surr_cols)}}).to_csv(spec_csv, index=False)
        report["samples_csv"] = str(spec_csv)
        per_spec.append(report)
        print(
            f"spec {si}: scaffolds={m_ours['n_scaffolds']:.0f} "
            f"R={m_ours.get('R_global_z', m_ours['R']):.2f} "
            f"v_beh={m_ours.get('v_beh_global_z', m_ours['v_beh']):.3f} "
            f"pass={report.get('pass')}"
        )

    n_pass = sum(1 for r in per_spec if r.get("pass"))
    summary = {
        "n_specs": len(per_spec),
        "n_pass": n_pass,
        "pass_rate": n_pass / max(len(per_spec), 1),
        "checkpoint": str(checkpoint),
        "n_samples": n_samples,
        "thresholds": {
            "min_scaffolds": THRESHOLDS.e3_min_scaffolds_pass,
            "beh_var_pass": THRESHOLDS.e3_beh_var_pass,
            "r_vs_base": THRESHOLDS.e3_r_vs_base_frac,
        },
        "per_spec": per_spec,
    }
    dump_json(summary, out_dir / "e3_summary.json")
    return summary
