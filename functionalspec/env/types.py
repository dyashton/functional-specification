"""Phase II environment / planning types.

Three clean concepts (do not mix):
  InteractionEnvironment — external physics (atom/interaction graph)
  DesignObjective        — user wants (see gen.objective)
  FunctionalSpecification — molecule-centric requirements (see gen.spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InteractionEnvironment:
    """External physical context the molecule will encounter.

    Must not carry DesignObjective fields (LogP targets, synth, toxicity, …).
    """

    env_type: str  # "co2" | "pocket" | …
    x: torch.Tensor  # (N, node_dim)
    edge_index: torch.Tensor  # (2, E)
    edge_attr: torch.Tensor  # (E, edge_dim)
    provenance: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))


@dataclass
class InteractionRepresentation:
    """Encoder output: what the world affords (not a Spec)."""

    embedding: torch.Tensor  # (D,) or (B, D)
    env_type: str = ""
    provenance: str = ""
