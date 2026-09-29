"""Environment Encoder: InteractionEnvironment → InteractionRepresentation.

Descriptive only — must not ingest DesignObjective fields or emit Spec.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.env.types import InteractionEnvironment, InteractionRepresentation
from functionalspec.models.encoder import GINELayer


class EnvironmentEncoder(nn.Module):
    """GINE over env graph → pooled affordance embedding (IR)."""

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        out_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, 16)
        self.layers = nn.ModuleList([GINELayer(hidden_dim, 16) for _ in range(num_layers)])
        self.out = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_emb(x)
        ea = self.edge_emb(edge_attr) if edge_attr.numel() else edge_attr
        for layer in self.layers:
            if edge_index.numel() == 0:
                break
            h = h + layer(h, edge_index, ea)
        pooled = h.mean(dim=0)
        return self.out(pooled)

    def encode(self, env: InteractionEnvironment) -> InteractionRepresentation:
        emb = self.forward_graph(env.x, env.edge_index, env.edge_attr)
        return InteractionRepresentation(
            embedding=emb,
            env_type=env.env_type,
            provenance=env.provenance,
        )

    def encode_batch(self, envs: list[InteractionEnvironment]) -> torch.Tensor:
        """Stack IR embeddings (B, D)."""
        return torch.stack([self.encode(e).embedding for e in envs], dim=0)
