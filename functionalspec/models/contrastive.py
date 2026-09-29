"""Multi-view contrastive loss with −ECFP fingerprint similarity in positive weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def multi_view_positive_weight(
    sim_desc: torch.Tensor,
    sim_qm: torch.Tensor | None,
    sim_bind: torch.Tensor | None,
    sim_interaction: torch.Tensor | None,
    sim_ecfp: torch.Tensor,
    w_desc: float = 1.0,
    w_qm: float = 0.5,
    w_bind: float = 1.0,
    w_interaction: float = 0.5,
    w_ecfp: float = 1.0,
) -> torch.Tensor:
    """Higher = better positive pair. ECFP similarity is subtracted."""
    s = w_desc * sim_desc - w_ecfp * sim_ecfp
    if sim_qm is not None:
        s = s + w_qm * sim_qm
    if sim_bind is not None:
        s = s + w_bind * sim_bind
    if sim_interaction is not None:
        s = s + w_interaction * sim_interaction
    return s


def info_nce(
    z: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    z: (batch, proj_dim) L2-normalized
    positive_mask: (batch, batch) bool, True for positives (exclude diagonal)
    """
    logits = z @ z.t() / temperature
    logits = logits - torch.eye(z.size(0), device=z.device) * 1e9
    # for each anchor, log-softmax over all others; maximize mass on positives
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = positive_mask.float()
    pos = pos / pos.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss = -(pos * log_prob).sum(dim=1)
    # only anchors that have ≥1 positive
    has_pos = positive_mask.any(dim=1)
    if not has_pos.any():
        return z.new_zeros(())
    return loss[has_pos].mean()
