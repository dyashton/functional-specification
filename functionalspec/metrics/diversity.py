"""Diversity / scaffold / fingerprint metrics for E3."""

from __future__ import annotations

from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import descriptors first so RDKit error logs are disabled for invalid SMILES.
from functionalspec.data.descriptors import brics_fragment_multiset, ecfp_bits, tanimoto
from functionalspec.metrics.thresholds import (
    behavioral_variance,
    diversity_behavior_ratio,
    structural_diversity,
)


def validity_rate(smiles_list: list[str]) -> float:
    if not smiles_list:
        return 0.0
    ok = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ok += 1
    return ok / len(smiles_list)


def unique_rate(smiles_list: list[str]) -> float:
    if not smiles_list:
        return 0.0
    return len(set(smiles_list)) / len(smiles_list)


def shannon_entropy(counts: Counter[str] | dict[str, int]) -> float:
    if not counts:
        return 0.0
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def murcko_counts(smiles_list: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            continue
        # empty string = acyclic; still a bucket
        counts[sc if sc else "<acyclic>"] += 1
    return counts


def murcko_entropy(smiles_list: list[str]) -> float:
    return shannon_entropy(murcko_counts(smiles_list))


def murcko_scaffold_count(smiles_list: list[str]) -> int:
    return len(murcko_counts(smiles_list))


def brics_counts(smiles_list: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for s in smiles_list:
        multi = brics_fragment_multiset(s)
        if not multi:
            continue
        counts.update(multi)
    return counts


def brics_entropy(smiles_list: list[str]) -> float:
    return shannon_entropy(brics_counts(smiles_list))


def mean_pairwise_tanimoto(smiles_list: list[str], max_pairs: int = 5000, seed: int = 0) -> float:
    fps = []
    for s in smiles_list:
        fp = ecfp_bits(s)
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    sims = [tanimoto(fps[i], fps[j]) for i, j in pairs]
    return float(np.mean(sims))


def mean_nn_tanimoto(smiles_list: list[str], max_n: int = 2000, seed: int = 0) -> float:
    """Mean ECFP Tanimoto to each molecule's nearest neighbor in the set (excl. self)."""
    fps = []
    for s in smiles_list:
        fp = ecfp_bits(s)
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        fps = [fps[i] for i in idx]
        n = len(fps)
    mat = np.stack(fps, axis=0).astype(np.float64)
    # Tanimoto via bits
    inter = mat @ mat.T
    card = mat.sum(axis=1)
    union = card[:, None] + card[None, :] - inter
    tani = inter / np.maximum(union, 1e-8)
    np.fill_diagonal(tani, -1.0)
    nn = tani.max(axis=1)
    return float(np.mean(nn))


def structure_quantiles(smiles_list: list[str], seed: int = 0) -> dict[str, float]:
    """Scaffold / fragment / fingerprint diversity summary for a molecule set."""
    uniq = list(dict.fromkeys(smiles_list))
    m_counts = murcko_counts(uniq)
    b_counts = brics_counts(uniq)
    top_scaf = m_counts.most_common(1)[0][1] / max(sum(m_counts.values()), 1) if m_counts else 0.0
    top_brics = b_counts.most_common(1)[0][1] / max(sum(b_counts.values()), 1) if b_counts else 0.0
    tbar = mean_pairwise_tanimoto(uniq, seed=seed)
    return {
        "n_unique": float(len(uniq)),
        "n_scaffolds": float(len(m_counts)),
        "murcko_entropy": shannon_entropy(m_counts),
        "top_scaffold_frac": float(top_scaf),
        "brics_entropy": shannon_entropy(b_counts),
        "n_brics_types": float(len(b_counts)),
        "top_brics_frac": float(top_brics),
        "mean_pairwise_tanimoto": tbar,
        "mean_nn_tanimoto": mean_nn_tanimoto(uniq, seed=seed),
        "d_struct": structural_diversity(tbar),
    }


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def e3_metrics(
    smiles_list: list[str],
    surrogate_matrix: np.ndarray,
) -> dict[str, float]:
    """surrogate_matrix: (n, d) aligned with smiles_list (invalids already dropped)."""
    uniq = list(dict.fromkeys(smiles_list))
    tbar = mean_pairwise_tanimoto(uniq)
    d_struct = structural_diversity(tbar)
    yz = zscore(surrogate_matrix)
    v_beh = behavioral_variance(yz)
    r = diversity_behavior_ratio(d_struct, v_beh)
    return {
        "n": float(len(smiles_list)),
        "n_unique": float(len(uniq)),
        "validity": validity_rate(smiles_list),
        "unique_rate": unique_rate(smiles_list),
        "n_scaffolds": float(murcko_scaffold_count(uniq)),
        "mean_tanimoto": tbar,
        "d_struct": d_struct,
        "brics_entropy": brics_entropy(uniq),
        "v_beh": v_beh,
        "R": r,
        "mean_surrogate": float(np.nanmean(surrogate_matrix)),
    }
