"""Multi-task surrogate functional heads on S."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateHeads(nn.Module):
    def __init__(self, in_dim: int, n_targets: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, flat_S: torch.Tensor) -> torch.Tensor:
        return self.net(flat_S)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target)
        # mask: (batch, n_targets) bool/float
        err = (pred - target).pow(2)
        return (err * mask).sum() / mask.sum().clamp_min(1.0)
