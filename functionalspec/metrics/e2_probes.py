"""E2 linear probes: surrogate sufficiency vs structure memorization."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def probe_regression(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "r2": float(r2_score(yte, pred)),
        "spearman": _spearman(yte, pred),
    }


def probe_multiregression(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> dict[str, float]:
    scores = [probe_regression(X, Y[:, j], seed=seed) for j in range(Y.shape[1])]
    return {
        "mean_r2": float(np.nanmean([s["r2"] for s in scores])),
        "mean_spearman": float(np.nanmean([s["spearman"] for s in scores])),
    }


def probe_classification(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    labels = np.unique(y)
    if len(labels) < 2:
        return {"accuracy": float("nan"), "auroc": float("nan")}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out = {"accuracy": float(accuracy_score(yte, pred))}
    if len(labels) == 2:
        proba = clf.predict_proba(Xte)[:, 1]
        out["auroc"] = float(roc_auc_score(yte, proba))
    else:
        out["auroc"] = float("nan")
    return out


def e2_gap(p_f: float, p_s: float) -> float:
    return float(p_f - p_s)


def normalize01(values: dict[str, float]) -> dict[str, float]:
    xs = np.array(list(values.values()), dtype=float)
    lo, hi = np.nanmin(xs), np.nanmax(xs)
    if not np.isfinite(lo) or hi - lo < 1e-8:
        return {k: 0.5 for k in values}
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}
