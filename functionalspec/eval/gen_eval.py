"""G1–G5 scientific evaluation for Spec → Generator MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from functionalspec.data.descriptors import compute_surrogates
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.eval.harness import dump_json
from functionalspec.gen.fan import FunctionalAbstractionNetwork
from functionalspec.gen.generate import generate_from_spec, load_generator
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FunctionalSpecification
from functionalspec.gen.spec_bank import SpecBank
from functionalspec.metrics.diversity import murcko_entropy, murcko_scaffold_count
from functionalspec.metrics.thresholds import THRESHOLDS


def _logp_of(smiles: list[str]) -> list[float]:
    vals = []
    for s in smiles:
        d = compute_surrogates(s)
        if d is not None and np.isfinite(d.get("LogP", float("nan"))):
            vals.append(float(d["LogP"]))
    return vals


def _match_rate(accepted: list[str], target: dict[str, float], tol_raw: dict[str, float]) -> float:
    if not accepted or not target:
        return 0.0
    hits = 0
    for s in accepted:
        d = compute_surrogates(s)
        if d is None:
            continue
        ok = True
        for k, v in target.items():
            if k not in d or not np.isfinite(d[k]):
                ok = False
                break
            if abs(float(d[k]) - v) > tol_raw.get(k, 0.5):
                ok = False
                break
        if ok:
            hits += 1
    return hits / max(len(accepted), 1)


def run_g1_g2(
    fan: FunctionalAbstractionNetwork,
    model,
    tok,
    objectives: list[str],
    *,
    n: int,
    temperature: float,
    device: torch.device,
    max_len: int,
    compose: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for obj_s in objectives:
        obj = DesignObjective.from_string(obj_s)
        spec = fan.plan(obj, compose=compose)
        res = generate_from_spec(
            spec, model, tok, n=n, temperature=temperature, device=device, max_len=max_len
        )
        # raw tolerances: LogP ±0.5, TPSA ±10, else ±0.5*std-ish
        tol_raw = {k: (0.5 if k == "LogP" else 10.0 if k == "TPSA" else 5.0) for k in obj.target_y}
        g1 = _match_rate(res.accepted, obj.target_y, tol_raw) if obj.target_y else float("nan")
        n_scaf = murcko_scaffold_count(res.accepted) if res.accepted else 0
        motif_ids = [
            motif_id
            for row in (res.meta.get("motif_ids") or [])
            for motif_id in row
            if motif_id > 1
        ]
        rows.append(
            {
                "id": "G1_G2",
                "objective": obj_s,
                "outcome": res.outcome,
                "confidence": res.confidence,
                "n_accepted": res.n_accepted,
                "acceptance_rate": res.acceptance_rate,
                "g1_match_rate": g1,
                "g2_murcko_entropy": res.murcko_entropy,
                "g2_n_scaffolds": n_scaf,
                "empirical_v_beh": res.empirical_v_beh,
                "motif_unique_codes": len(set(motif_ids)),
                "motif_code_coverage": (
                    len(set(motif_ids)) / max(res.meta.get("motif_vocab_size", 1), 1)
                    if motif_ids
                    else float("nan")
                ),
                "predicted_behavior": res.predicted_behavior,
                "provenance": spec.provenance,
            }
        )
    return rows


def run_g3(
    fan: FunctionalAbstractionNetwork,
    model,
    tok,
    objectives: list[str],
    *,
    n: int,
    temperature: float,
    device: torch.device,
    max_len: int,
    compose: bool = True,
) -> dict[str, Any]:
    confs, rates = [], []
    for obj_s in objectives:
        obj = DesignObjective.from_string(obj_s)
        spec = fan.plan(obj, compose=compose)
        res = generate_from_spec(
            spec, model, tok, n=n, temperature=temperature, device=device, max_len=max_len
        )
        confs.append(spec.confidence)
        rates.append(res.acceptance_rate)
    confs_a = np.asarray(confs, dtype=np.float64)
    rates_a = np.asarray(rates, dtype=np.float64)
    if len(confs_a) >= 3 and confs_a.std() > 1e-8 and rates_a.std() > 1e-8:
        rho = float(np.corrcoef(confs_a, rates_a)[0, 1])
    else:
        rho = float("nan")
    return {
        "id": "G3",
        "n_objectives": len(objectives),
        "confidence_vs_acceptance_pearson": rho,
        "mean_confidence": float(confs_a.mean()) if len(confs_a) else float("nan"),
        "mean_acceptance": float(rates_a.mean()) if len(rates_a) else float("nan"),
        "points": [{"objective": o, "confidence": c, "acceptance": a} for o, c, a in zip(objectives, confs, rates)],
    }


def run_g4_logp_ladder(
    fan: FunctionalAbstractionNetwork,
    model,
    tok,
    *,
    logp_targets: list[float],
    n: int,
    temperature: float,
    device: torch.device,
    max_len: int,
    compose: bool = True,
) -> dict[str, Any]:
    rungs = []
    for lp in logp_targets:
        obj = DesignObjective(target_y={"LogP": lp}, name=f"LogP={lp}")
        spec = fan.plan(obj, compose=compose)
        res = generate_from_spec(
            spec, model, tok, n=n, temperature=temperature, device=device, max_len=max_len
        )
        realized = _logp_of(res.accepted)
        rungs.append(
            {
                "target_logp": lp,
                "outcome": spec.outcome,
                "confidence": spec.confidence,
                "provenance": spec.provenance,
                "n_accepted": res.n_accepted,
                "realized_logp_mean": float(np.mean(realized)) if realized else float("nan"),
                "realized_logp_std": float(np.std(realized)) if realized else float("nan"),
                "flat_S_norm": float(torch.linalg.vector_norm(spec.flat_S).item()),
                "example_smiles": res.accepted[:5],
            }
        )
    # trajectory correlation: target vs realized mean
    tgt = np.asarray([r["target_logp"] for r in rungs], dtype=np.float64)
    real = np.asarray([r["realized_logp_mean"] for r in rungs], dtype=np.float64)
    mask = np.isfinite(real)
    if mask.sum() >= 3 and real[mask].std() > 1e-8:
        rho = float(np.corrcoef(tgt[mask], real[mask])[0, 1])
    else:
        rho = float("nan")
    return {"id": "G4", "rungs": rungs, "target_vs_realized_pearson": rho}


def run_g5_repeatability(
    fan: FunctionalAbstractionNetwork,
    model,
    tok,
    objective: str,
    *,
    n_batches: int,
    n_per_batch: int,
    temperature: float,
    device: torch.device,
    max_len: int,
    compose: bool = True,
) -> dict[str, Any]:
    obj = DesignObjective.from_string(objective)
    spec = fan.plan(obj, compose=compose)
    batch_means = []
    all_accepted: list[str] = []
    entropies = []
    for _ in range(n_batches):
        res = generate_from_spec(
            spec, model, tok, n=n_per_batch, temperature=temperature, device=device, max_len=max_len
        )
        lp = _logp_of(res.accepted)
        batch_means.append(float(np.mean(lp)) if lp else float("nan"))
        entropies.append(res.murcko_entropy)
        all_accepted.extend(res.accepted)
    means = np.asarray(batch_means, dtype=np.float64)
    means_ok = means[np.isfinite(means)]
    return {
        "id": "G5",
        "objective": objective,
        "spec_outcome": spec.outcome,
        "confidence": spec.confidence,
        "n_batches": n_batches,
        "behavior_mean_std_across_batches": float(means_ok.std()) if len(means_ok) else float("nan"),
        "behavior_means": batch_means,
        "mean_murcko_entropy": float(np.nanmean(entropies)) if entropies else float("nan"),
        "union_n_unique": len(set(all_accepted)),
        "union_murcko_entropy": murcko_entropy(all_accepted) if all_accepted else 0.0,
        "union_n_scaffolds": murcko_scaffold_count(all_accepted) if all_accepted else 0,
    }


def run_gen_eval(
    bank_path: Path,
    generator_ckpt: Path,
    p1_checkpoint: Path,
    out_dir: Path,
    *,
    n: int = 64,
    temperature: float = 1.0,
    device: str | None = None,
    seed: int = 42,
    g5_batches: int = 5,
    compose: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bank = SpecBank(bank_path)
    enc, _ = load_planner_for_embed(p1_checkpoint, device_t)
    fan = FunctionalAbstractionNetwork(bank, encoder=enc, device=device_t)
    model, tok, ckpt = load_generator(generator_ckpt, device_t)
    max_len = int(ckpt.get("max_len", 150))

    objectives = [
        "LogP=1",
        "LogP=2",
        "LogP=3",
        "LogP=4",
        "TPSA=40",
        "TPSA=80",
        "LogP=2,TPSA=50",
        "LogP=3,TPSA=60",
    ]
    kw = dict(n=n, temperature=temperature, device=device_t, max_len=max_len, compose=compose)
    g12 = run_g1_g2(fan, model, tok, objectives, **kw)
    g3 = run_g3(fan, model, tok, objectives, **kw)
    g4 = run_g4_logp_ladder(
        fan,
        model,
        tok,
        logp_targets=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        **kw,
    )
    g5 = run_g5_repeatability(
        fan,
        model,
        tok,
        "LogP=2.5",
        n_batches=g5_batches,
        n_per_batch=n,
        temperature=temperature,
        device=device_t,
        max_len=max_len,
        compose=compose,
    )

    summary = {
        "generator": str(generator_ckpt),
        "bank": str(bank_path),
        "n_per_objective": n,
        "temperature": temperature,
        "compose": compose,
        "tol_filter": THRESHOLDS.gen_behavior_tol,
        "G1_G2": g12,
        "G3": g3,
        "G4": g4,
        "G5": g5,
        "headline": {
            "g1_mean_match": float(np.nanmean([r["g1_match_rate"] for r in g12])),
            "g2_mean_entropy": float(np.nanmean([r["g2_murcko_entropy"] for r in g12])),
            "g3_conf_acceptance_rho": g3["confidence_vs_acceptance_pearson"],
            "g4_logp_rho": g4["target_vs_realized_pearson"],
            "g5_behavior_std": g5["behavior_mean_std_across_batches"],
            "g5_union_entropy": g5["union_murcko_entropy"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_dir / "gen_eval_summary.json")
    return summary
