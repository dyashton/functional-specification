"""Property-conditional FiLM decoder (capacity-matched to Spec Arm A via y→cond_dim proj)."""

from __future__ import annotations

import torch
import torch.nn as nn

from functionalspec.models.generator_a import SmilesConditionedDecoder


class PropertyConditionedDecoder(nn.Module):
    """z-scored surrogate vector → project to Spec cond_dim → same FiLM GRU as Arm A."""

    def __init__(
        self,
        n_y: int,
        cond_dim: int,
        vocab_size: int = 128,
        hidden: int = 512,
        num_layers: int = 2,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_y = n_y
        self.cond_dim = cond_dim
        self.y_proj = nn.Sequential(
            nn.Linear(n_y, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.dec = SmilesConditionedDecoder(
            cond_dim=cond_dim,
            vocab_size=vocab_size,
            hidden=hidden,
            num_layers=num_layers,
            cond_dropout=cond_dropout,
        )

    def _to_cond(self, y_z: torch.Tensor) -> torch.Tensor:
        return self.y_proj(y_z)

    def forward(self, y_z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.dec(self._to_cond(y_z), tokens)

    @torch.no_grad()
    def sample(self, y_z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dec.sample(self._to_cond(y_z), **kwargs)
