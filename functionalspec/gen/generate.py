"""Generate molecules from a FunctionalSpecification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from functionalspec.data.descriptors import SURROGATE_ALWAYS, compute_surrogates
from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.smiles_tokenizer import MoleculeTokenizer
from functionalspec.gen.spec import FunctionalSpecification
from functionalspec.metrics.diversity import murcko_entropy
from functionalspec.metrics.thresholds import THRESHOLDS, behavioral_variance
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.generator_b import LearnedMotifDecoder
from functionalspec.models.planner import FunctionalPlanner


@dataclass
class GenerationResult:
    smiles: list[str]
    accepted: list[str]
    n_sampled: int
    n_accepted: int
    acceptance_rate: float
    empirical_v_beh: float
    murcko_entropy: float
    outcome: str
    confidence: float
    predicted_behavior: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


def load_generator(
    checkpoint: Path,
    device: torch.device,
) -> tuple[FunctionalPlanner, MoleculeTokenizer, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    tok = MoleculeTokenizer.from_dict(ckpt["tokenizer"])
    surr_cols = ckpt["surrogate_cols"]
    n_slots = int(ckpt.get("num_slots", mcfg["num_slots"]))
    cb = int(ckpt.get("codebook_size", mcfg.get("primary_codebook", 128)))
    beta = float(ckpt.get("commitment_cost", mcfg["commitment_cost"]))
    no_vq = bool(ckpt.get("no_vq", False))
    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=n_slots,
        codebook_size=cb,
        n_surrogates=len(surr_cols),
        commitment_cost=beta,
        no_vq=no_vq,
        with_arm_a=False,
        with_arm_b=False,
    )
    cond_dim = n_slots * int(mcfg["hidden_dim"])
    if ckpt.get("arm_b", False):
        motif_payload = ckpt["motif_vocab"]
        model.arm_b = LearnedMotifDecoder(
            cond_dim=cond_dim,
            motif_vocab_size=len(motif_payload["tokens"]) + 2,
            atom_vocab_size=tok.vocab_size,
            motif_codebook=int(ckpt.get("motif_codebook", 256)),
            motif_dim=int(ckpt.get("motif_dim", 128)),
            max_motifs=int(ckpt.get("max_motifs", motif_payload["max_motifs"])),
            atom_hidden=int(ckpt.get("atom_hidden", 512)),
            atom_layers=int(ckpt.get("atom_layers", 2)),
            cond_dropout=float(ckpt.get("cond_dropout", 0.0)),
        )
    else:
        model.arm_a = SmilesConditionedDecoder(
            cond_dim=cond_dim,
            vocab_size=tok.vocab_size,
            hidden=int(ckpt.get("arm_hidden", 512)),
            num_layers=int(ckpt.get("arm_layers", 2)),
            cond_dropout=float(ckpt.get("cond_dropout", 0.0)),
        )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model, tok, ckpt


def _surrogates_matrix(smiles: list[str], cols: list[str]) -> tuple[list[str], np.ndarray]:
    kept, rows = [], []
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
        if ok:
            kept.append(s)
            rows.append(vals)
    if not rows:
        return [], np.zeros((0, len(cols)))
    return kept, np.asarray(rows, dtype=np.float64)


def generate_from_spec(
    spec: FunctionalSpecification,
    model: FunctionalPlanner,
    tokenizer: MoleculeTokenizer,
    *,
    n: int = 100,
    batch_size: int = 32,
    temperature: float = 1.0,
    tol: float | None = None,
    max_len: int = 150,
    device: torch.device | None = None,
    filter: bool = True,
) -> GenerationResult:
    if spec.outcome == "unsupported":
        return GenerationResult(
            smiles=[],
            accepted=[],
            n_sampled=0,
            n_accepted=0,
            acceptance_rate=0.0,
            empirical_v_beh=float("nan"),
            murcko_entropy=0.0,
            outcome=spec.outcome,
            confidence=spec.confidence,
            predicted_behavior={c: float(v) for c, v in zip(spec.surrogate_cols, spec.predicted_behavior)},
            meta={"reason": "unsupported_spec"},
        )

    device = device or next(model.parameters()).device
    tol = THRESHOLDS.gen_behavior_tol if tol is None else tol
    if model.arm_a is None and model.arm_b is None:
        raise RuntimeError("Checkpoint contains neither Arm A nor Arm B")
    flat = spec.flat_S.to(device)
    if flat.dim() == 1:
        flat = flat.unsqueeze(0)

    sampled: list[str] = []
    motif_ids: list[list[int]] = []
    with torch.no_grad():
        while len(sampled) < n:
            b = min(batch_size, n - len(sampled))
            cond = flat.expand(b, -1)
            if model.arm_b is not None:
                rows, predicted_ids = model.arm_b.sample(
                    cond, max_len=max_len, temperature=temperature
                )
                motif_ids.extend(predicted_ids.cpu().tolist())
            else:
                assert model.arm_a is not None
                rows = model.arm_a.sample(cond, max_len=max_len, temperature=temperature)
            for row in rows:
                smi = tokenizer.decode_to_smiles(row)
                if smi:
                    sampled.append(smi)

    cols_all = list(spec.surrogate_cols) if spec.surrogate_cols else list(SURROGATE_ALWAYS)
    # Only descriptors we can recompute on generated molecules
    computable = set(SURROGATE_ALWAYS) | {"HallKierAlpha"}
    idx = [i for i, c in enumerate(cols_all) if c in computable]
    cols = [cols_all[i] for i in idx] or list(SURROGATE_ALWAYS)
    y_pred_full = np.asarray(spec.predicted_behavior, dtype=np.float64)
    y_pred = y_pred_full[idx] if len(idx) == len(cols) and len(y_pred_full) >= max(idx or [0]) + 1 else y_pred_full[: len(cols)]

    kept, Y = _surrogates_matrix(sampled, cols)
    accepted = kept
    if filter and len(kept) > 0 and len(y_pred) == len(cols):
        # Behavior ball is in z-space (same units as THRESHOLDS.behavior_match_tol)
        if spec.y_mean is not None and spec.y_std is not None and len(idx) == len(cols):
            mu = np.asarray(spec.y_mean, dtype=np.float64)[idx]
            sd = np.asarray(spec.y_std, dtype=np.float64)[idx]
        elif spec.y_mean is not None and spec.y_std is not None:
            mu = np.asarray(spec.y_mean, dtype=np.float64)[: len(cols)]
            sd = np.asarray(spec.y_std, dtype=np.float64)[: len(cols)]
        else:
            mu, sd = Y.mean(0), Y.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Yz = (Y - mu) / sd
        yz = (y_pred - mu) / sd
        d = np.linalg.norm(Yz - yz[None, :], axis=1) / np.sqrt(len(cols))
        mask = d <= tol
        accepted = [kept[i] for i in range(len(kept)) if mask[i]]
        Y_acc = Y[mask] if accepted else np.zeros((0, len(cols)))
    else:
        Y_acc = Y

    if len(Y_acc) >= 2:
        # z within accepted set
        v = behavioral_variance((Y_acc - Y_acc.mean(0)) / np.where(Y_acc.std(0) < 1e-6, 1.0, Y_acc.std(0)))
    else:
        v = float("nan")
    h = murcko_entropy(accepted) if accepted else 0.0

    return GenerationResult(
        smiles=sampled,
        accepted=accepted,
        n_sampled=len(sampled),
        n_accepted=len(accepted),
        acceptance_rate=len(accepted) / max(len(sampled), 1),
        empirical_v_beh=v,
        murcko_entropy=h,
        outcome=spec.outcome,
        confidence=spec.confidence,
        predicted_behavior={c: float(v) for c, v in zip(cols, y_pred)},
        meta={
            "tol": tol,
            "temperature": temperature,
            "filter_cols": cols,
            "arm": "b" if model.arm_b is not None else "a",
            "motif_ids": motif_ids[: len(sampled)] if motif_ids else None,
            "motif_vocab_size": (
                model.arm_b.motif_vocab_size if model.arm_b is not None else None
            ),
        },
    )
