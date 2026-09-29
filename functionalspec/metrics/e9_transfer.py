"""E9 frozen-representation transfer evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import spearmanr


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def fit_head(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "regression",
    head: str = "linear",
    seed: int = 0,
) -> tuple[object, dict[str, float]]:
    if task_type == "classification" and len(np.unique(y)) < 2:
        return None, {"accuracy": float("nan")}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    if task_type == "classification":
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return None, {"accuracy": float("nan")}
        if head == "mlp":
            model: object = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
        else:
            model = LogisticRegression(max_iter=500)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        return model, {"accuracy": float(accuracy_score(yte, pred))}

    if head == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
    else:
        model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, {"r2": float(r2_score(yte, pred)), "spearman": _spearman(yte, pred)}


def low_n_curve(
    X: np.ndarray,
    y: np.ndarray,
    ns: tuple[int, ...] = (32, 64, 128, 256),
    task_type: str = "regression",
    head: str = "linear",
    seed: int = 0,
    n_repeats: int = 5,
) -> dict[int, float]:
    """Mean primary score vs training subset size."""
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    for n in ns:
        if n >= len(y) or n < 8:
            continue
        scores = []
        for r in range(n_repeats):
            # Stratify-ish for classification: ensure both classes when possible
            if task_type == "classification":
                pos = np.where(y > 0.5)[0]
                neg = np.where(y <= 0.5)[0]
                if len(pos) == 0 or len(neg) == 0:
                    continue
                n_pos = max(1, min(len(pos), n // 2))
                n_neg = min(len(neg), n - n_pos)
                if n_pos + n_neg < n and len(pos) > n_pos:
                    n_pos = min(len(pos), n - n_neg)
                if n_pos + n_neg < 8:
                    continue
                idx = np.concatenate(
                    [
                        rng.choice(pos, size=n_pos, replace=False),
                        rng.choice(neg, size=n_neg, replace=False),
                    ]
                )
            else:
                idx = rng.choice(len(y), size=n, replace=False)
            Xn, yn = X[idx], y[idx]
            _, metrics = fit_head(Xn, yn, task_type=task_type, head=head, seed=seed + r)
            primary = metrics.get("spearman", metrics.get("accuracy", metrics.get("r2", float("nan"))))
            if np.isfinite(primary):
                scores.append(primary)
        out[n] = float(np.mean(scores)) if scores else float("nan")
    return out


def rank_methods(
    method_curves: dict[str, dict[int, float]],
    n: int,
) -> dict[str, int]:
    """1 = best. Higher score is better."""
    vals = {m: curve.get(n, float("nan")) for m, curve in method_curves.items()}
    order = sorted(
        vals.keys(),
        key=lambda m: (-vals[m] if np.isfinite(vals[m]) else 1e9),
    )
    return {m: i + 1 for i, m in enumerate(order)}
