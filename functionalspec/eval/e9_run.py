"""E9 runner: frozen S vs ECFP low-n transfer on held-out tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.data.e9_tasks import HELDOUT_TASKS, load_task_table
from functionalspec.data.featurize import smiles_to_fp, smiles_to_graph
from functionalspec.data.graph_dataset import collate_graphs
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.eval.harness import dump_json, e9_report
from functionalspec.metrics.thresholds import THRESHOLDS
from functionalspec.train.p2 import encode_flat_S


def _graph_items(smiles: list[str]) -> tuple[list[str], list[dict], list[np.ndarray]]:
    kept_s, items, fps = [], [], []
    for s in smiles:
        g = smiles_to_graph(s)
        fp = smiles_to_fp(s)
        if g is None or fp is None:
            continue
        kept_s.append(s)
        items.append(
            {
                "smiles": s,
                "x": g["x"],
                "edge_index": g["edge_index"],
                "edge_attr": g["edge_attr"],
                "y": torch.zeros(1),  # unused
                "fp": torch.from_numpy(fp.astype(np.float32)),
            }
        )
        fps.append(fp.astype(np.float32))
    return kept_s, items, fps


@torch.no_grad()
def embed_items(
    model,
    items: list[dict],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    outs = []
    for batch in tqdm(loader, desc="embed S", leave=False):
        outs.append(encode_flat_S(model, batch, device).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1), dtype=np.float32)


def run_e9(
    checkpoint: Path,
    e9_dir: Path,
    out_dir: Path,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 42,
    max_mols: int | None = None,
) -> dict[str, Any]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ckpt = load_planner_for_embed(checkpoint, device_t)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_task: list[dict[str, Any]] = []
    wins = 0

    for spec in HELDOUT_TASKS:
        path = e9_dir / f"{spec.task_id}.csv"
        if not path.exists():
            per_task.append({"task_id": spec.task_id, "pass": False, "reasons": [f"missing {path}"]})
            continue
        df = load_task_table(spec.task_id, e9_dir.parent)
        if max_mols is not None:
            df = df.head(max_mols)
        y_raw = df[spec.label_col].to_numpy(dtype=np.float64)
        smiles = df["SMILES"].astype(str).tolist()

        kept_s, items, fps = _graph_items(smiles)
        # align labels to kept
        smi_to_y = {s: y for s, y in zip(smiles, y_raw)}
        y = np.asarray([smi_to_y[s] for s in kept_s], dtype=np.float64)
        fp = np.stack(fps, axis=0)

        if spec.task_type == "classification":
            # binary labels
            y = (y > 0.5).astype(np.float64)
            if len(np.unique(y)) < 2:
                per_task.append(
                    {"task_id": spec.task_id, "pass": False, "reasons": ["single-class after filter"]}
                )
                continue

        print(f"{spec.task_id}: n={len(kept_s)} type={spec.task_type}")
        S = embed_items(model, items, device_t, batch_size=batch_size)

        report = e9_report({"S": S, "ECFP": fp}, y, task_type=spec.task_type)
        # Serialize int keys
        report["curves"] = {m: {str(k): v for k, v in curve.items()} for m, curve in report["curves"].items()}
        report["ranks"] = {str(k): v for k, v in report["ranks"].items()}

        from functionalspec.metrics.e9_transfer import fit_head

        _, full_s = fit_head(S, y, task_type=spec.task_type, seed=seed)
        _, full_fp = fit_head(fp, y, task_type=spec.task_type, seed=seed)

        def _primary(m: dict[str, float]) -> float:
            for k in ("spearman", "accuracy", "r2"):
                if k in m and np.isfinite(m[k]):
                    return float(m[k])
            return float("nan")

        full_s_score = _primary(full_s)
        full_fp_score = _primary(full_fp)
        full_ok = bool(
            np.isfinite(full_s_score)
            and np.isfinite(full_fp_score)
            and full_s_score >= full_fp_score - THRESHOLDS.e9_full_data_margin
        )

        # Low-n: only compare n where both scores are finite; tied-best counts
        s_wins = 0
        n_compared = 0
        for n_str, ranks in report["ranks"].items():
            s_sc = report["curves"]["S"].get(n_str)
            e_sc = report["curves"]["ECFP"].get(n_str)
            if s_sc is None or e_sc is None:
                continue
            if not (np.isfinite(s_sc) and np.isfinite(e_sc)):
                continue
            if "S" not in ranks or "ECFP" not in ranks:
                continue
            n_compared += 1
            if ranks["S"] <= ranks["ECFP"]:
                s_wins += 1
        low_n_ok = n_compared > 0 and s_wins >= max(1, (n_compared + 1) // 2)
        task_pass = bool(low_n_ok and full_ok)
        reasons = []
        if not low_n_ok:
            reasons.append(f"low-n S_wins={s_wins}/{n_compared}")
        if not full_ok:
            reasons.append(
                f"full-data S={full_s_score:.3f} < ECFP−{THRESHOLDS.e9_full_data_margin} "
                f"({full_fp_score - THRESHOLDS.e9_full_data_margin:.3f})"
            )
        if task_pass:
            wins += 1

        row = {
            "task_id": spec.task_id,
            "n": int(len(kept_s)),
            "task_type": spec.task_type,
            "label_col": spec.label_col,
            "curves": report["curves"],
            "ranks": report["ranks"],
            "s_wins_over_ecfp": s_wins,
            "n_compared": n_compared,
            "low_n_ok": low_n_ok,
            "full_ok": full_ok,
            "full_S_score": full_s_score,
            "full_ECFP_score": full_fp_score,
            "full_S": full_s,
            "full_ECFP": full_fp,
            "pass": task_pass,
            "reasons": reasons or ["pass"],
        }
        per_task.append(row)
        dump_json(row, out_dir / f"{spec.task_id}.json")
        print(
            f"  pass={task_pass} low_n={s_wins}/{n_compared} full_ok={full_ok} "
            f"S={full_s_score:.3f} ECFP={full_fp_score:.3f} reasons={reasons or ['pass']}"
        )

    summary = {
        "checkpoint": str(checkpoint),
        "n_tasks": len(per_task),
        "n_wins": wins,
        "pass": wins >= THRESHOLDS.e9_min_tasks_win,
        "threshold_min_wins": THRESHOLDS.e9_min_tasks_win,
        "full_data_margin": THRESHOLDS.e9_full_data_margin,
        "per_task": per_task,
    }
    dump_json(summary, out_dir / "e9_summary.json")
    return summary
