"""Phase II FAN: plan(InteractionEnvironment, DesignObjective) → Spec(s).

EnvEncoder produces InteractionRepresentation (affordances).
FAN ranks on-manifold Specs — does not dump environment into Spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functionalspec.env.encoder import EnvironmentEncoder
from functionalspec.env.types import InteractionEnvironment, InteractionRepresentation
from functionalspec.gen.fan import FunctionalAbstractionNetwork, _confidence
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FunctionalSpecification
from functionalspec.gen.spec_bank import SpecBank


class IRToQuery(nn.Module):
    """Map IR (+ optional objective vec) → query in flat_S space. No env dump into Spec."""

    def __init__(self, ir_dim: int, spec_dim: int, obj_dim: int = 0, hidden: int = 512):
        super().__init__()
        in_dim = ir_dim + max(obj_dim, 0)
        self.obj_dim = obj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, spec_dim),
        )

    def forward(self, ir: torch.Tensor, obj: torch.Tensor | None = None) -> torch.Tensor:
        if self.obj_dim > 0:
            if obj is None:
                obj = torch.zeros(ir.size(0), self.obj_dim, device=ir.device, dtype=ir.dtype)
            x = torch.cat([ir, obj], dim=-1)
        else:
            x = ir
        q = self.net(x)
        return F.normalize(q, dim=-1)


class ContextFAN(FunctionalAbstractionNetwork):
    """FAN(env, objective) with frozen Spec bank + trainable IR→query."""

    def __init__(
        self,
        bank: SpecBank,
        env_encoder: EnvironmentEncoder,
        ir_to_query: IRToQuery,
        *,
        lam: float = 1.0,
        device: torch.device | None = None,
        strong_mask: np.ndarray | None = None,
    ):
        super().__init__(bank, lam=lam, encoder=None, device=device)
        self.env_encoder = env_encoder.to(self.device)
        self.ir_to_query = ir_to_query.to(self.device)
        self.strong_mask = strong_mask  # optional bool mask over bank rows
        # Precompute normalized Spec bank
        S = bank.flat_S.astype(np.float32)
        self._Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)

    def encode_env(self, env: InteractionEnvironment) -> InteractionRepresentation:
        self.env_encoder.eval()
        with torch.no_grad():
            env_d = InteractionEnvironment(
                env_type=env.env_type,
                x=env.x.to(self.device),
                edge_index=env.edge_index.to(self.device),
                edge_attr=env.edge_attr.to(self.device),
                provenance=env.provenance,
                meta=env.meta,
            )
            return self.env_encoder.encode(env_d)

    def _objective_vec(self, objective: DesignObjective | None) -> np.ndarray | None:
        if objective is None or not objective.target_y:
            return None
        return objective.vector(self.bank.surrogate_cols, self.bank.y_mean, self.bank.y_std)

    def plan(
        self,
        objective: DesignObjective | None = None,
        *,
        environment: InteractionEnvironment | None = None,
        top_k: int = 50,
        compose: bool = True,
        unsupported_dist: float = 2.0,
        n_strategies: int = 1,
    ) -> FunctionalSpecification | list[FunctionalSpecification]:
        """Plan Spec(s) from (env, objective).

        MVP compat: environment=None → objective-only bank retrieve (parent behavior).
        Phase II: environment set → IR query + bank rank; objective optional soft prior.
        """
        if environment is None:
            if objective is None:
                objective = DesignObjective(name="empty")
            # seed / objective-only path
            if objective.seed_smiles or objective.target_y:
                return super().plan(
                    objective, top_k=top_k, compose=compose and n_strategies == 1, unsupported_dist=unsupported_dist
                )
            return self._pack(
                flat=np.zeros(self.bank.flat_S.shape[1], dtype=np.float32),
                y_pred=np.zeros(len(self.bank.surrogate_cols)),
                confidence=0.0,
                outcome="unsupported",
                provenance="empty_context",
            )

        ir = self.encode_env(environment)
        obj_np = self._objective_vec(objective)
        obj_dim = self.ir_to_query.obj_dim
        with torch.no_grad():
            ir_t = ir.embedding.detach().float().view(1, -1)
            if obj_dim > 0:
                if obj_np is None:
                    obj_t = torch.zeros(1, obj_dim, device=self.device)
                else:
                    # align / pad objective to obj_dim
                    v = np.zeros(obj_dim, dtype=np.float32)
                    n = min(obj_dim, len(obj_np))
                    v[:n] = obj_np[:n].astype(np.float32)
                    obj_t = torch.from_numpy(v).view(1, -1).to(self.device)
            else:
                obj_t = None
            self.ir_to_query.eval()
            q = self.ir_to_query(ir_t, obj_t).cpu().numpy()[0]

        sims = self._Sn @ q
        score = sims + self.lam * (self.bank.R / (self.R_ref + 1e-6)) * 0.05
        if obj_np is not None:
            w = objective.dim_weights(self.bank.surrogate_cols) if objective else np.ones(len(self.bank.surrogate_cols))
            diff = (self.bank.Y_z - obj_np[None, :]) * w[None, :]
            dist = np.sqrt((diff**2).sum(axis=1) / max(float(w.sum()), 1.0))
            score = score - 0.5 * dist

        order = np.argsort(-score)
        k = min(top_k, len(order))
        top = order[:k]

        strategies: list[FunctionalSpecification] = []
        # Greedy diverse strategies: pick tops with Spec cosine < 0.85 to already chosen
        chosen: list[int] = []
        for idx in top:
            if not chosen:
                chosen.append(int(idx))
            else:
                sims_c = [float(self._Sn[idx] @ self._Sn[c]) for c in chosen]
                if max(sims_c) < 0.85:
                    chosen.append(int(idx))
            if len(chosen) >= max(n_strategies, 1):
                break
        if not chosen:
            chosen = [int(top[0])]

        for i, bi in enumerate(chosen):
            R = float(self.bank.R[bi])
            dens = float(self.bank.density[bi])
            # distance proxy: 1 - cosine sim to query
            dist = float(1.0 - sims[bi])
            conf, outcome = _confidence(R, dens, dist, self.R_ref, self.dens_ref)
            strategies.append(
                self._pack(
                    flat=self.bank.flat_S[bi],
                    y_pred=self.bank.Y_raw[bi].copy(),
                    confidence=conf,
                    outcome=outcome,
                    provenance=f"ctx:{environment.provenance}|bank:{bi}|strat:{i}",
                    R=R,
                    dist=dist,
                    token_indices=self.bank.token_indices[bi],
                )
            )

        if n_strategies <= 1:
            if compose and k > 1:
                # optional compose among top for single Spec
                sc = score[top]
                sc = sc - sc.max()
                wt = np.exp(sc)
                wt = wt / wt.sum()
                flat = (self.bank.flat_S[top] * wt[:, None]).sum(axis=0)
                y_pred = (self.bank.Y_raw[top] * wt[:, None]).sum(axis=0)
                R_mix = float((self.bank.R[top] * wt).sum())
                dens = float((self.bank.density[top] * wt).sum())
                dist_eff = float(1.0 - sims[chosen[0]])
                conf, outcome = _confidence(R_mix, dens, dist_eff, self.R_ref, self.dens_ref)
                return self._pack(
                    flat=flat,
                    y_pred=y_pred,
                    confidence=conf,
                    outcome=outcome,
                    provenance=f"ctx:{environment.provenance}|compose:{k}",
                    R=R_mix,
                    dist=dist_eff,
                    token_indices=self.bank.token_indices[chosen[0]],
                )
            return strategies[0]
        return strategies

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "env_encoder": self.env_encoder.state_dict(),
                "ir_to_query": self.ir_to_query.state_dict(),
                "ir_dim": self.env_encoder.out_dim,
                "spec_dim": int(self.bank.flat_S.shape[1]),
                "obj_dim": self.ir_to_query.obj_dim,
                "lam": self.lam,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        bank: SpecBank,
        device: torch.device | None = None,
    ) -> "ContextFAN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        env_enc = EnvironmentEncoder(out_dim=int(ckpt["ir_dim"]))
        env_enc.load_state_dict(ckpt["env_encoder"])
        proj = IRToQuery(
            ir_dim=int(ckpt["ir_dim"]),
            spec_dim=int(ckpt["spec_dim"]),
            obj_dim=int(ckpt.get("obj_dim", 0)),
        )
        proj.load_state_dict(ckpt["ir_to_query"])
        return cls(bank, env_enc, proj, lam=float(ckpt.get("lam", 1.0)), device=device)
