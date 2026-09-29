"""Minimal GINE-style molecule encoder → token sequence."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINELayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = MLP(edge_dim, dim, dim)
        self.node_mlp = MLP(dim, dim, dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # x: (n, d), edge_index: (2, e), edge_attr: (e, de)
        src, dst = edge_index
        msg = x[src] + self.edge_mlp(edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        return self.node_mlp((1 + self.eps) * x + agg)


class MoleculeEncoder(nn.Module):
    """Encode batched single-graph tensors (pad externally for multi-graph)."""

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        out_dim: int | None = None,
    ):
        super().__init__()
        out_dim = out_dim or hidden_dim
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, 16)
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, 16) for _ in range(num_layers)]
        )
        self.out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        for layer in self.layers:
            h = h + layer(h, edge_index, e)
        return self.out(h)

    def pooled(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.forward(x, edge_index, edge_attr)
        if batch is None:
            return h.mean(dim=0, keepdim=True)
        # scatter mean per graph
        bsize = int(batch.max().item()) + 1
        out = h.new_zeros((bsize, h.size(-1)))
        counts = h.new_zeros((bsize, 1))
        out.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype))
        return out / counts.clamp_min(1.0)
