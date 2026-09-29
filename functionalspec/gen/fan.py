"""Functional Abstraction Network (FAN) — MVP: DesignObjective → Spec."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from functionalspec.data.featurize import smiles_to_graph
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FlatSPayload, FunctionalSpecification
from functionalspec.gen.spec_bank import SpecBank
from functionalspec.models.planner import FunctionalPlanner


Outcome = Literal["exists", "uncertain", "unsupported"]


def _confidence(R: float, density: float, dist: float, R_ref: float, dens_ref: float) -> tuple[float, Outcome]:
    """Multi-factor confidence in [0,1]; map to ternary outcome."""
    # Normalize roughly
    r_term = float(np.clip(R / (R_ref + 1e-6), 0, 2) / 2)
    d_term = float(np.clip(density / (dens_ref + 1e-6), 0, 2) / 2)
    dist_term = float(np.exp(-dist))  # 0 distance → 1
    conf = 0.45 * r_term + 0.25 * d_term + 0.30 * dist_term
    conf = float(np.clip(conf, 0, 1))
    if dist > 1.5 and R < 0.5 * R_ref:
        return conf, "unsupported"
    if conf >= 0.55 and dist <= 0.75:
        return conf, "exists"
    if conf < 0.35 or dist > 1.2:
        return conf, "uncertain" if dist <= 2.0 else "unsupported"
    return conf, "uncertain"


class FunctionalAbstractionNetwork:
    """FAN_MVP: retrieve → rank → optionally compose on-manifold Specs."""

    def __init__(self, bank: SpecBank, lam: float = 1.0, encoder: FunctionalPlanner | None = None, device: torch.device | None = None):
        self.bank = bank
        self.lam = lam
        self.encoder = encoder
        self.device = device or torch.device("cpu")
        self.R_ref = float(np.median(bank.R))
        self.dens_ref = float(np.median(bank.density))

    def _pack(
        self,
        *,
        flat: np.ndarray | torch.Tensor,
        y_pred: np.ndarray,
        confidence: float,
        outcome: Outcome,
        provenance: str,
        R: float = float("nan"),
        dist: float = float("nan"),
        token_indices: np.ndarray | None = None,
    ) -> FunctionalSpecification:
        if isinstance(flat, np.ndarray):
            flat_t = torch.from_numpy(np.asarray(flat, dtype=np.float32))
        else:
            flat_t = flat
        return FunctionalSpecification(
            payload=FlatSPayload(flat_S=flat_t, token_indices=token_indices),
            predicted_behavior=np.asarray(y_pred, dtype=np.float64),
            confidence=confidence,
            outcome=outcome,
            provenance=provenance,
            specificity_R_internal=R,
            behavior_distance=dist,
            surrogate_cols=tuple(self.bank.surrogate_cols),
            y_mean=np.asarray(self.bank.y_mean, dtype=np.float64),
            y_std=np.asarray(self.bank.y_std, dtype=np.float64),
        )

    def plan(
        self,
        objective: DesignObjective,
        *,
        environment=None,
        top_k: int = 50,
        compose: bool = True,
        unsupported_dist: float = 2.0,
        n_strategies: int = 1,
    ) -> FunctionalSpecification:
        """MVP: objective-only. Pass environment to ContextFAN for Phase II."""
        if environment is not None:
            raise TypeError(
                "FunctionalAbstractionNetwork.plan ignores environment; "
                "use ContextFAN.plan(objective, environment=...) for Phase II"
            )
        if objective.seed_smiles:
            return self.plan_from_seed(objective.seed_smiles)

        y_tgt = objective.vector(self.bank.surrogate_cols, self.bank.y_mean, self.bank.y_std)
        if y_tgt is None:
            return self._pack(
                flat=np.zeros(self.bank.flat_S.shape[1], dtype=np.float32),
                y_pred=np.zeros(len(self.bank.surrogate_cols)),
                confidence=0.0,
                outcome="unsupported",
                provenance="empty_objective",
            )

        w = objective.dim_weights(self.bank.surrogate_cols)
        # weighted z-distance
        diff = (self.bank.Y_z - y_tgt[None, :]) * w[None, :]
        dist = np.sqrt((diff**2).sum(axis=1) / max(float(w.sum()), 1.0))
        score = -dist + self.lam * (self.bank.R / (self.R_ref + 1e-6))
        order = np.argsort(-score)
        k = min(top_k, len(order))
        top = order[:k]
        best = int(top[0])
        best_dist = float(dist[best])

        if best_dist > unsupported_dist and float(self.bank.R[best]) < 0.3 * self.R_ref:
            return self._pack(
                flat=self.bank.flat_S[best],
                y_pred=self.bank.Y_raw[best].copy(),
                confidence=0.05,
                outcome="unsupported",
                provenance=f"bank:{best}",
                R=float(self.bank.R[best]),
                dist=best_dist,
                token_indices=self.bank.token_indices[best],
            )

        if compose and k > 1:
            # score-weighted convex combination of top-K flat_S
            sc = score[top]
            sc = sc - sc.max()
            wt = np.exp(sc)
            wt = wt / wt.sum()
            flat = (self.bank.flat_S[top] * wt[:, None]).sum(axis=0)
            y_pred = (self.bank.Y_raw[top] * wt[:, None]).sum(axis=0)
            R_mix = float((self.bank.R[top] * wt).sum())
            dens = float((self.bank.density[top] * wt).sum())
            # disagreement among top-K behaviors
            disagree = float(np.mean(np.var(self.bank.Y_z[top], axis=0)))
            dist_eff = best_dist + 0.25 * disagree
            conf, outcome = _confidence(R_mix, dens, dist_eff, self.R_ref, self.dens_ref)
            provenance = f"compose:{k}"
            tok = self.bank.token_indices[best]
        else:
            flat = self.bank.flat_S[best]
            y_pred = self.bank.Y_raw[best].copy()
            R_mix = float(self.bank.R[best])
            dens = float(self.bank.density[best])
            conf, outcome = _confidence(R_mix, dens, best_dist, self.R_ref, self.dens_ref)
            provenance = f"bank:{best}"
            tok = self.bank.token_indices[best]

        return self._pack(
            flat=flat,
            y_pred=y_pred,
            confidence=conf,
            outcome=outcome,
            provenance=provenance,
            R=R_mix,
            dist=best_dist,
            token_indices=tok,
        )

    def plan_from_seed(self, smiles: str) -> FunctionalSpecification:
        if self.encoder is None:
            # fall back to nearest bank SMILES exact or fail
            if smiles in self.bank.smiles:
                i = self.bank.smiles.index(smiles)
            else:
                return self._pack(
                    flat=np.zeros(self.bank.flat_S.shape[1], dtype=np.float32),
                    y_pred=np.zeros(len(self.bank.surrogate_cols)),
                    confidence=0.0,
                    outcome="unsupported",
                    provenance="seed_no_encoder",
                )
            conf, outcome = _confidence(
                float(self.bank.R[i]), float(self.bank.density[i]), 0.0, self.R_ref, self.dens_ref
            )
            return self._pack(
                flat=self.bank.flat_S[i],
                y_pred=self.bank.Y_raw[i].copy(),
                confidence=conf,
                outcome=outcome,
                provenance=f"seed_bank:{i}",
                R=float(self.bank.R[i]),
                dist=0.0,
                token_indices=self.bank.token_indices[i],
            )

        g = smiles_to_graph(smiles)
        if g is None:
            return self._pack(
                flat=np.zeros(self.bank.flat_S.shape[1], dtype=np.float32),
                y_pred=np.zeros(len(self.bank.surrogate_cols)),
                confidence=0.0,
                outcome="unsupported",
                provenance="seed_invalid",
            )
        self.encoder.eval()
        with torch.no_grad():
            x = g["x"].to(self.device)
            ei = g["edge_index"].to(self.device)
            ea = g["edge_attr"].to(self.device)
            b = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            out = self.encoder.forward_graphs(x, ei, ea, b)
            flat = out["flat_S"][0].cpu()
            tok = out["indices"][0].cpu().numpy()
        # nearest bank for R/density/behavior readout
        Sn = self.bank.flat_S / (np.linalg.norm(self.bank.flat_S, axis=1, keepdims=True) + 1e-8)
        q = flat.numpy()
        q = q / (np.linalg.norm(q) + 1e-8)
        j = int(np.argmax(Sn @ q))
        conf, outcome = _confidence(
            float(self.bank.R[j]), float(self.bank.density[j]), 0.1, self.R_ref, self.dens_ref
        )
        return self._pack(
            flat=flat,
            y_pred=self.bank.Y_raw[j].copy(),
            confidence=conf,
            outcome=outcome,
            provenance=f"seed_encode~bank:{j}",
            R=float(self.bank.R[j]),
            dist=0.1,
            token_indices=tok,
        )
