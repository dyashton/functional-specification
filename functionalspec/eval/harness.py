"""Shared eval helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from functionalspec.metrics.diversity import e3_metrics, murcko_scaffold_count, validity_rate
from functionalspec.metrics.e2_probes import e2_gap, probe_multiregression, probe_regression
from functionalspec.metrics.thresholds import THRESHOLDS, behavior_match_ok, e3_pass
from functionalspec.metrics.e9_transfer import low_n_curve, rank_methods


def load_smiles_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def e0_report(smiles: list[str]) -> dict[str, Any]:
    thr = THRESHOLDS
    v = validity_rate(smiles)
    u = len(set(smiles)) / max(len(smiles), 1)
    status = "pass" if v >= thr.e0_validity_pass and u >= thr.e0_unique_pass else "fail"
    if v < thr.e0_validity_fail or u < thr.e0_unique_fail:
        status = "fail"
    return {"validity": v, "unique_rate": u, "status": status}


def e3_report(
    smiles: list[str],
    surrogates: np.ndarray,
    base_smiles: list[str] | None = None,
    base_surrogates: np.ndarray | None = None,
) -> dict[str, Any]:
    m = e3_metrics(smiles, surrogates)
    out: dict[str, Any] = {"ours": m}
    if base_smiles is not None and base_surrogates is not None:
        mb = e3_metrics(base_smiles, base_surrogates)
        out["baseline"] = mb
        matched = behavior_match_ok(
            surrogates.mean(axis=0),
            base_surrogates.mean(axis=0),
            tol=THRESHOLDS.behavior_match_tol,
        )
        out["behavior_matched"] = matched
        ok, reasons = e3_pass(
            n_scaffolds=int(m["n_scaffolds"]),
            v_beh=float(m["v_beh"]),
            r=float(m["R"]),
            r_base=float(mb["R"]),
            n_scaf_base=int(mb["n_scaffolds"]),
        )
        out["pass"] = bool(ok and matched)
        out["reasons"] = reasons if matched else ["behavior not matched — refuse E3 comparison"]
    else:
        out["pass"] = None
        out["reasons"] = ["no matched baseline provided"]
    return out


def e2_report(
    S: np.ndarray,
    Y_surrogate: np.ndarray,
    Y_structure: np.ndarray,
    S_recon: np.ndarray | None = None,
) -> dict[str, Any]:
    pf = probe_multiregression(S, Y_surrogate)
    # structure as multi-reg on fingerprint bits or scaffold embedding dims
    ps = probe_multiregression(S, Y_structure)
    gap = e2_gap(pf["mean_spearman"], ps["mean_spearman"])
    out = {
        "surrogate_probe": pf,
        "structure_probe": ps,
        "gap_spearman": gap,
        "pass": gap >= THRESHOLDS.e2_gap_pass,
    }
    if S_recon is not None:
        ps_r = probe_multiregression(S_recon, Y_structure)
        out["recon_structure_probe"] = ps_r
        out["structure_margin"] = ps_r["mean_spearman"] - ps["mean_spearman"]
        out["pass"] = bool(
            out["pass"] and out["structure_margin"] >= THRESHOLDS.e2_structure_margin_vs_recon
        )
    return out


def e6_token_delta(
    surrogates_with: np.ndarray,
    surrogates_without: np.ndarray,
) -> dict[str, Any]:
    mu_w = surrogates_with.mean(axis=0)
    mu_o = surrogates_without.mean(axis=0)
    # z-score using with-token stats
    sd = surrogates_with.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    delta = (mu_o - mu_w) / sd
    max_abs = float(np.max(np.abs(delta)))
    status = "pass" if max_abs >= THRESHOLDS.e6_delta_pass else "fail"
    if max_abs < THRESHOLDS.e6_delta_fail:
        status = "fail"
    return {"delta_z": delta.tolist(), "max_abs_delta": max_abs, "status": status}


def e8_path_report(validity_per_step: list[float], surrogate_path: np.ndarray) -> dict[str, Any]:
    # surrogate_path: (T, d)
    jumps = np.linalg.norm(np.diff(surrogate_path, axis=0), axis=1)
    total = float(np.sum(jumps) + 1e-8)
    smoothness = float(np.mean(jumps) / total * len(jumps))
    vmean = float(np.mean(validity_per_step))
    status = "pass" if vmean >= THRESHOLDS.e8_validity_pass else "fail"
    if vmean < THRESHOLDS.e8_validity_fail:
        status = "fail"
    return {"validity_mean": vmean, "smoothness": smoothness, "status": status}


def e9_report(
    method_features: dict[str, np.ndarray],
    y: np.ndarray,
    task_type: str = "regression",
    ns: tuple[int, ...] = (32, 64, 128, 256),
) -> dict[str, Any]:
    curves = {
        name: low_n_curve(X, y, ns=ns, task_type=task_type)
        for name, X in method_features.items()
    }
    ranks = {n: rank_methods(curves, n) for n in ns if any(n in c for c in curves.values())}
    return {"curves": curves, "ranks": ranks, "ns": list(ns)}


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=_default))
