"""Operational metric thresholds and E3 ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    e0_validity_pass: float = 0.80
    e0_validity_fail: float = 0.50
    e0_unique_pass: float = 0.40
    e0_unique_fail: float = 0.10

    e1_spearman_vs_fp_frac: float = 0.70
    e1_spearman_abs_fail: float = 0.40
    e1b_ie_spearman_pass: float = 0.25

    e2_gap_pass: float = 0.25
    e2_structure_margin_vs_recon: float = 0.15
    e2_nn_tanimoto_margin: float = 0.15

    e3_n_samples: int = 1000
    e3_min_scaffolds_pass: int = 30
    e3_scaffolds_fail: int = 5
    e3_beh_var_pass: float = 0.35
    e3_beh_var_fail: float = 1.0
    e3_r_vs_base_frac: float = 1.25
    e3_scaf_vs_base_frac: float = 1.5
    e3_ie_strong_mass_pass: float = 0.40
    e3_core_motif_fail_frac: float = 0.80
    behavior_match_tol: float = 0.15
    # Individual-molecule Spec ball for generate_from_spec (set-mean E3 still uses behavior_match_tol)
    gen_behavior_tol: float = 0.75
    eps: float = 1e-6

    e6_delta_pass: float = 0.40
    e6_delta_fail: float = 0.10

    e8_validity_pass: float = 0.70
    e8_validity_fail: float = 0.40

    e9_min_tasks_win: int = 3
    e9_n_tasks: int = 5
    e9_full_data_margin: float = 0.02  # S must be ≥ ECFP − this on full data


THRESHOLDS = Thresholds()


def structural_diversity(mean_pairwise_tanimoto: float) -> float:
    return 1.0 - float(mean_pairwise_tanimoto)


def behavioral_variance(zscored_surrogates: np.ndarray) -> float:
    """Mean variance across surrogate dims. Shape (n_mols, d)."""
    if zscored_surrogates.size == 0:
        return float("nan")
    return float(np.nanmean(np.nanvar(zscored_surrogates, axis=0)))


def diversity_behavior_ratio(d_struct: float, v_beh: float, eps: float = 1e-6) -> float:
    return float(d_struct) / (float(v_beh) + eps)


def behavior_match_ok(mean_a: np.ndarray, mean_b: np.ndarray, tol: float = 0.15) -> bool:
    d = max(len(mean_a), 1)
    return float(np.linalg.norm(mean_a - mean_b) / np.sqrt(d)) <= tol


def e3_pass(
    n_scaffolds: int,
    v_beh: float,
    r: float,
    r_base: float,
    n_scaf_base: int,
    thr: Thresholds = THRESHOLDS,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    if n_scaffolds <= thr.e3_scaffolds_fail:
        ok = False
        reasons.append(f"scaffolds={n_scaffolds} ≤ fail gate {thr.e3_scaffolds_fail}")
    if v_beh > thr.e3_beh_var_fail:
        ok = False
        reasons.append(f"beh_var={v_beh:.3f} > fail {thr.e3_beh_var_fail}")
    gain_r = r >= thr.e3_r_vs_base_frac * r_base
    gain_scaf = n_scaffolds >= thr.e3_scaf_vs_base_frac * n_scaf_base
    if not (gain_r or gain_scaf):
        ok = False
        reasons.append("no gain vs matched-behavior baseline on R or scaffold count")
    if n_scaffolds >= thr.e3_min_scaffolds_pass and v_beh <= thr.e3_beh_var_pass and (gain_r or gain_scaf):
        return True, ["pass"]
    if ok and n_scaffolds < thr.e3_min_scaffolds_pass:
        reasons.append(f"scaffolds={n_scaffolds} < pass target {thr.e3_min_scaffolds_pass} (soft)")
    return ok and n_scaffolds >= thr.e3_min_scaffolds_pass, reasons or ["borderline"]
