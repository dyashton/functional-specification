"""G6 ablations: does Spec S carry behavior beyond descriptors / chance?

G6b — no retrain: fixed FAN Spec; scramble or randomize flat_S; filter off.
G6a — matched capacity: flat_S decoder vs property(y) decoder vs random S.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from functionalspec.data.descriptors import SURROGATE_ALWAYS, compute_surrogates
from functionalspec.eval.e2_run import load_planner_for_embed
from functionalspec.eval.harness import dump_json
from functionalspec.gen.fan import FunctionalAbstractionNetwork
from functionalspec.gen.generate import generate_from_spec, load_generator, _surrogates_matrix
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FlatSPayload, FunctionalSpecification
from functionalspec.gen.spec_bank import SpecBank
from functionalspec.metrics.diversity import murcko_entropy, murcko_scaffold_count
from functionalspec.metrics.thresholds import behavioral_variance
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.property_decoder import PropertyConditionedDecoder


AblationKind = Literal["true_S", "scramble_S", "random_S", "property_y"]


def _logp_stats(smiles: list[str]) -> dict[str, float]:
    vals = []
    for s in smiles:
        d = compute_surrogates(s)
        if d is not None and np.isfinite(d.get("LogP", float("nan"))):
            vals.append(float(d["LogP"]))
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "hit_pm05": 0.0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "hit_pm05": float("nan"),  # filled by caller with target
    }


def _hit_rate(smiles: list[str], target_logp: float, tol: float = 0.5) -> float:
    vals = []
    for s in smiles:
        d = compute_surrogates(s)
        if d is not None and np.isfinite(d.get("LogP", float("nan"))):
            vals.append(abs(float(d["LogP"]) - target_logp) <= tol)
    return float(np.mean(vals)) if vals else 0.0


def _v_beh_computable(smiles: list[str]) -> float:
    cols = list(SURROGATE_ALWAYS)
    kept, Y = _surrogates_matrix(smiles, cols)
    if len(kept) < 2:
        return float("nan")
    sd = np.where(Y.std(0) < 1e-6, 1.0, Y.std(0))
    return behavioral_variance((Y - Y.mean(0)) / sd)


def _mutate_spec(spec: FunctionalSpecification, kind: AblationKind, rng: np.random.Generator) -> FunctionalSpecification:
    """Return a Spec with modified flat_S (predicted_behavior unchanged for scoring)."""
    out = deepcopy(spec)
    s = spec.flat_S.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if kind == "true_S":
        return out
    if kind == "scramble_S":
        perm = rng.permutation(len(s))
        s2 = s[perm]
    elif kind == "random_S":
        # same L2 norm as true S — fair magnitude for FiLM
        nrm = float(np.linalg.norm(s)) + 1e-8
        s2 = rng.normal(size=len(s)).astype(np.float32)
        s2 = s2 / (np.linalg.norm(s2) + 1e-8) * nrm
    else:
        raise ValueError(f"G6b mutate does not support {kind}")
    out.payload = FlatSPayload(flat_S=torch.from_numpy(s2), token_indices=spec.payload.token_indices)
    return out


def _sample_metrics(
    smiles: list[str],
    target_logp: float,
) -> dict[str, Any]:
    lp = _logp_stats(smiles)
    return {
        "n_valid": lp["n"],
        "n_unique": len(set(smiles)),
        "logp_mean": lp["mean"],
        "logp_std": lp["std"],
        "hit_rate_pm05": _hit_rate(smiles, target_logp, 0.5),
        "murcko_entropy": murcko_entropy(smiles) if smiles else 0.0,
        "n_scaffolds": murcko_scaffold_count(smiles) if smiles else 0,
        "v_beh": _v_beh_computable(smiles),
    }


def run_g6b(
    bank_path: Path,
    generator_ckpt: Path,
    p1_checkpoint: Path,
    out_dir: Path,
    *,
    logp_targets: list[float] | None = None,
    n: int = 64,
    temperature: float = 1.0,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Decoder-uses-S check: true vs scramble vs random flat_S (filter off)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logp_targets = logp_targets or [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    bank = SpecBank(bank_path)
    enc, _ = load_planner_for_embed(p1_checkpoint, device_t)
    fan = FunctionalAbstractionNetwork(bank, encoder=enc, device=device_t)
    model, tok, ckpt = load_generator(generator_ckpt, device_t)
    max_len = int(ckpt.get("max_len", 150))

    kinds: list[AblationKind] = ["true_S", "scramble_S", "random_S"]
    rows: list[dict[str, Any]] = []
    for lp in logp_targets:
        obj = DesignObjective(target_y={"LogP": lp}, name=f"LogP={lp}")
        spec = fan.plan(obj)
        if spec.outcome == "unsupported":
            rows.append({"target_logp": lp, "outcome": "unsupported"})
            continue
        for kind in kinds:
            spec_k = _mutate_spec(spec, kind, rng)
            res = generate_from_spec(
                spec_k,
                model,
                tok,
                n=n,
                temperature=temperature,
                device=device_t,
                max_len=max_len,
                filter=False,
            )
            m = _sample_metrics(res.smiles, lp)
            rows.append(
                {
                    "target_logp": lp,
                    "kind": kind,
                    "outcome": spec.outcome,
                    "provenance": spec.provenance,
                    **m,
                }
            )

    # Headline: Pearson(target, realized_mean) per kind
    headline = {}
    for kind in kinds:
        sub = [r for r in rows if r.get("kind") == kind and np.isfinite(r.get("logp_mean", float("nan")))]
        if len(sub) >= 3:
            t = np.asarray([r["target_logp"] for r in sub], dtype=np.float64)
            m = np.asarray([r["logp_mean"] for r in sub], dtype=np.float64)
            rho = float(np.corrcoef(t, m)[0, 1]) if m.std() > 1e-8 else float("nan")
        else:
            rho = float("nan")
        hits = [r["hit_rate_pm05"] for r in sub] if sub else []
        headline[kind] = {
            "logp_pearson": rho,
            "mean_hit_rate_pm05": float(np.mean(hits)) if hits else float("nan"),
            "mean_murcko_entropy": float(np.nanmean([r["murcko_entropy"] for r in sub])) if sub else float("nan"),
            "mean_v_beh": float(np.nanmean([r["v_beh"] for r in sub])) if sub else float("nan"),
        }

    summary = {
        "id": "G6b",
        "generator": str(generator_ckpt),
        "n_per_rung": n,
        "filter": False,
        "headline": headline,
        "rows": rows,
        "pass_hint": (
            "true_S logp_pearson >> scramble_S and random_S "
            "(decoder uses Spec payload, not just sampling noise)"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_dir / "g6b_summary.json")
    return summary


def load_property_decoder(ckpt_path: Path, device: torch.device) -> tuple[PropertyConditionedDecoder, Any, dict]:
    from functionalspec.data.smiles_tokenizer import MoleculeTokenizer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok = MoleculeTokenizer.from_dict(ckpt["tokenizer"])
    model = PropertyConditionedDecoder(
        n_y=len(ckpt["surrogate_cols"]),
        cond_dim=int(ckpt["cond_dim"]),
        vocab_size=tok.vocab_size,
        hidden=int(ckpt.get("arm_hidden", 512)),
        num_layers=int(ckpt.get("arm_layers", 2)),
        cond_dropout=float(ckpt.get("cond_dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, tok, ckpt


@torch.no_grad()
def sample_from_cond(
    decoder: SmilesConditionedDecoder | PropertyConditionedDecoder,
    cond: torch.Tensor,
    tokenizer,
    *,
    n: int,
    max_len: int,
    temperature: float = 1.0,
    batch_size: int = 32,
) -> list[str]:
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    sampled: list[str] = []
    while len(sampled) < n:
        b = min(batch_size, n - len(sampled))
        c = cond.expand(b, -1)
        if isinstance(decoder, PropertyConditionedDecoder):
            rows = decoder.sample(c, max_len=max_len, temperature=temperature)
        else:
            rows = decoder.sample(c, max_len=max_len, temperature=temperature)
        for row in rows:
            smi = tokenizer.decode_to_smiles(row)
            if smi:
                sampled.append(smi)
    return sampled


def run_g6a(
    bank_path: Path,
    s_checkpoint: Path,
    y_checkpoint: Path,
    p1_checkpoint: Path,
    out_dir: Path,
    *,
    logp_targets: list[float] | None = None,
    n: int = 64,
    temperature: float = 1.0,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Matched S vs property-y vs random-S on LogP ladder (+ diversity)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logp_targets = logp_targets or [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    bank = SpecBank(bank_path)
    enc, _ = load_planner_for_embed(p1_checkpoint, device_t)
    fan = FunctionalAbstractionNetwork(bank, encoder=enc, device=device_t)
    s_model, s_tok, s_ckpt = load_generator(s_checkpoint, device_t)
    y_model, y_tok, y_ckpt = load_property_decoder(y_checkpoint, device_t)
    s_max = int(s_ckpt.get("max_len", 150))
    y_max = int(y_ckpt.get("max_len", 150))
    y_cols = list(y_ckpt["surrogate_cols"])
    y_mean = np.asarray(y_ckpt["y_mean"], dtype=np.float64)
    y_std = np.asarray(y_ckpt["y_std"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for lp in logp_targets:
        obj = DesignObjective(target_y={"LogP": lp}, name=f"LogP={lp}")
        spec = fan.plan(obj)
        if spec.outcome == "unsupported":
            rows.append({"target_logp": lp, "outcome": "unsupported"})
            continue

        # --- Model A: true S ---
        res_s = generate_from_spec(
            spec, s_model, s_tok, n=n, temperature=temperature, device=device_t, max_len=s_max, filter=False
        )
        rows.append({"target_logp": lp, "kind": "true_S", "outcome": spec.outcome, **_sample_metrics(res_s.smiles, lp)})

        # --- Model C: random S (same decoder) ---
        spec_r = _mutate_spec(spec, "random_S", rng)
        res_r = generate_from_spec(
            spec_r, s_model, s_tok, n=n, temperature=temperature, device=device_t, max_len=s_max, filter=False
        )
        rows.append({"target_logp": lp, "kind": "random_S", "outcome": spec.outcome, **_sample_metrics(res_r.smiles, lp)})

        # --- Model B: property vector (FAN-predicted y, z-scored) ---
        # Use Spec predicted_behavior aligned to y_cols; missing dims → 0 z
        y_raw = np.zeros(len(y_cols), dtype=np.float64)
        for i, c in enumerate(y_cols):
            if c in spec.surrogate_cols:
                j = list(spec.surrogate_cols).index(c)
                y_raw[i] = float(spec.predicted_behavior[j])
            elif c == "LogP":
                y_raw[i] = lp
        # Prefer objective LogP override for fair "condition on target"
        if "LogP" in y_cols:
            y_raw[y_cols.index("LogP")] = lp
        y_z = (y_raw - y_mean) / np.where(y_std < 1e-6, 1.0, y_std)
        # Zero out dims not in the design objective so B is "properties only" for LogP
        # (other dims = corpus mean → z=0). Keep full y for an alternate? Use objective-only:
        mask = np.zeros(len(y_cols), dtype=np.float64)
        mask[y_cols.index("LogP")] = 1.0
        y_z = y_z * mask
        cond = torch.from_numpy(y_z.astype(np.float32)).to(device_t)
        smiles_y = sample_from_cond(y_model, cond, y_tok, n=n, max_len=y_max, temperature=temperature)
        rows.append({"target_logp": lp, "kind": "property_y", "outcome": spec.outcome, **_sample_metrics(smiles_y, lp)})

    kinds = ["true_S", "property_y", "random_S"]
    headline = {}
    for kind in kinds:
        sub = [r for r in rows if r.get("kind") == kind and np.isfinite(r.get("logp_mean", float("nan")))]
        if len(sub) >= 3:
            t = np.asarray([r["target_logp"] for r in sub], dtype=np.float64)
            m = np.asarray([r["logp_mean"] for r in sub], dtype=np.float64)
            rho = float(np.corrcoef(t, m)[0, 1]) if m.std() > 1e-8 else float("nan")
        else:
            rho = float("nan")
        headline[kind] = {
            "logp_pearson": rho,
            "mean_hit_rate_pm05": float(np.mean([r["hit_rate_pm05"] for r in sub])) if sub else float("nan"),
            "mean_murcko_entropy": float(np.nanmean([r["murcko_entropy"] for r in sub])) if sub else float("nan"),
            "mean_n_scaffolds": float(np.nanmean([r["n_scaffolds"] for r in sub])) if sub else float("nan"),
            "mean_v_beh": float(np.nanmean([r["v_beh"] for r in sub])) if sub else float("nan"),
            "mean_logp_std": float(np.nanmean([r["logp_std"] for r in sub])) if sub else float("nan"),
        }

    summary = {
        "id": "G6a",
        "s_checkpoint": str(s_checkpoint),
        "y_checkpoint": str(y_checkpoint),
        "n_per_rung": n,
        "filter": False,
        "property_conditioning": "LogP z-score only (other dims zeroed)",
        "headline": headline,
        "rows": rows,
        "pass_hint": (
            "true_S should match or beat property_y on diversity at similar control; "
            "both should beat random_S on logp_pearson / hit_rate"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_dir / "g6a_summary.json")
    return summary
