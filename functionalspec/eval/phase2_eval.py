"""Phase II eval gates I1–I5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from functionalspec.env.dataset import ie_band, load_posed_complexes
from functionalspec.env.graph_co2 import co2_only_environment, environment_from_complex_xyz
from functionalspec.eval.harness import dump_json
from functionalspec.gen.fan import FunctionalAbstractionNetwork
from functionalspec.gen.fan_context import ContextFAN
from functionalspec.gen.generate import generate_from_spec, load_generator
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FlatSPayload, FunctionalSpecification
from functionalspec.gen.spec_bank import SpecBank
from functionalspec.metrics.diversity import murcko_entropy, murcko_scaffold_count


def _random_spec(bank: SpecBank, rng: np.random.Generator) -> FunctionalSpecification:
    i = int(rng.integers(0, len(bank)))
    return FunctionalSpecification(
        payload=FlatSPayload(flat_S=torch.from_numpy(bank.flat_S[i]), token_indices=bank.token_indices[i]),
        predicted_behavior=bank.Y_raw[i].copy(),
        confidence=0.1,
        outcome="uncertain",
        provenance=f"random_bank:{i}",
        surrogate_cols=tuple(bank.surrogate_cols),
        y_mean=bank.y_mean,
        y_std=bank.y_std,
    )


def _spec_strong_affinity(spec: FunctionalSpecification, bank: SpecBank, strong_idx: np.ndarray) -> float:
    """Max cosine sim of Spec flat_S to strong-host Specs in bank (proxy without Psi4)."""
    s = spec.flat_S.detach().cpu().numpy().reshape(-1).astype(np.float64)
    s = s / (np.linalg.norm(s) + 1e-8)
    Sn = bank.flat_S[strong_idx]
    Sn = Sn / (np.linalg.norm(Sn, axis=1, keepdims=True) + 1e-8)
    return float((Sn @ s).max()) if len(strong_idx) else float("nan")


def run_phase2_eval(
    fan_ckpt: Path,
    bank_path: Path,
    generator_ckpt: Path,
    co2_root: Path,
    out_dir: Path,
    *,
    n_gen: int = 32,
    n_strategies: int = 3,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bank = SpecBank(bank_path)
    fan = ContextFAN.load(fan_ckpt, bank, device=device_t)
    mvp_fan = FunctionalAbstractionNetwork(bank)
    model, tok, gckpt = load_generator(generator_ckpt, device_t)
    max_len = int(gckpt.get("max_len", 150))

    # Strong Specs: from posed IE labels matched into bank by SMILES when possible,
    # else top-R bank rows as proxy
    posed = load_posed_complexes(co2_root)
    strong_smiles = {p.smiles for p in posed if ie_band(p.ie_kcal_mol) == "strong"}
    strong_idx = np.array([i for i, s in enumerate(bank.smiles) if s in strong_smiles], dtype=int)
    if len(strong_idx) < 5:
        strong_idx = np.argsort(-bank.R)[:50]

    # Pick a held-out posed env for conditioning (or co2-only)
    test_envs = []
    for p in posed[:20]:
        try:
            test_envs.append(environment_from_complex_xyz(p.xyz_path))
        except Exception:
            continue
    if not test_envs:
        test_envs = [co2_only_environment()]

    env = test_envs[0]
    empty_obj = DesignObjective(name="empty")

    # --- I1: enrichment proxy ---
    def affinity_of_plan(plan_fn, label: str) -> dict[str, Any]:
        out = plan_fn()
        specs = out if isinstance(out, list) else [out]
        affs = [_spec_strong_affinity(sp, bank, strong_idx) for sp in specs]
        # generate from first Spec
        res = generate_from_spec(
            specs[0], model, tok, n=n_gen, temperature=1.0, device=device_t, max_len=max_len, filter=False
        )
        return {
            "label": label,
            "spec_affinity_mean": float(np.nanmean(affs)),
            "n_strategies": len(specs),
            "n_unique_mols": len(set(res.smiles)),
            "murcko_entropy": murcko_entropy(res.smiles) if res.smiles else 0.0,
            "n_scaffolds": murcko_scaffold_count(res.smiles) if res.smiles else 0,
            "outcome": specs[0].outcome,
            "provenance": specs[0].provenance,
        }

    i1_ctx = affinity_of_plan(
        lambda: fan.plan(empty_obj, environment=env, n_strategies=1, compose=True),
        "ctx_env",
    )
    i1_obj = affinity_of_plan(
        lambda: mvp_fan.plan(DesignObjective.from_string("LogP=2")),
        "obj_only_LogP2",
    )
    i1_rand = affinity_of_plan(lambda: _random_spec(bank, rng), "random_spec")

    # --- I2: diversity under one Spec ---
    i2 = {
        "murcko_entropy": i1_ctx["murcko_entropy"],
        "n_scaffolds": i1_ctx["n_scaffolds"],
        "n_unique": i1_ctx["n_unique_mols"],
    }

    # --- I3: multi-strategy distinctness ---
    strats = fan.plan(empty_obj, environment=env, n_strategies=n_strategies, compose=False)
    if not isinstance(strats, list):
        strats = [strats]
    flats = [sp.flat_S.detach().cpu().numpy().reshape(-1) for sp in strats]
    flats_n = [f / (np.linalg.norm(f) + 1e-8) for f in flats]
    pair_sims = []
    for i in range(len(flats_n)):
        for j in range(i + 1, len(flats_n)):
            pair_sims.append(float(flats_n[i] @ flats_n[j]))
    i3 = {
        "n_strategies": len(strats),
        "mean_pairwise_spec_cosine": float(np.mean(pair_sims)) if pair_sims else 1.0,
        "provenances": [sp.provenance for sp in strats],
    }

    # --- I4: ablations (reuse I1 baselines; do not resample random) ---
    i4 = {
        "env_affinity": i1_ctx["spec_affinity_mean"],
        "obj_affinity": i1_obj["spec_affinity_mean"],
        "random_affinity": i1_rand["spec_affinity_mean"],
        "env_beats_random": bool(i1_ctx["spec_affinity_mean"] > i1_rand["spec_affinity_mean"]),
        "env_vs_obj": float(i1_ctx["spec_affinity_mean"] - i1_obj["spec_affinity_mean"]),
        "note": "env-only vs obj-only vs random Spec; frozen gen (I5)",
    }

    # --- I5: frozen generator (checkpoint path recorded; no finetune in this run) ---
    i5 = {
        "generator_checkpoint": str(generator_ckpt),
        "fan_checkpoint": str(fan_ckpt),
        "arm_a_finetuned": False,
        "note": "Eval uses frozen runs/p2_selfies_cond weights",
    }

    summary = {
        "I1": {"ctx": i1_ctx, "obj_only": i1_obj, "random": i1_rand},
        "I2": i2,
        "I3": i3,
        "I4": i4,
        "I5": i5,
        "headline": {
            "ctx_affinity": i1_ctx["spec_affinity_mean"],
            "random_affinity": i1_rand["spec_affinity_mean"],
            "obj_affinity": i1_obj["spec_affinity_mean"],
            "diversity_H": i2["murcko_entropy"],
            "strategy_mean_cosine": i3["mean_pairwise_spec_cosine"],
            "env_beats_random": i4["env_beats_random"],
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_dir / "phase2_eval.json")
    return summary
