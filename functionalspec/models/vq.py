"""Slot attention + VQ codebook for Functional Specification S."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (..., dim)
        flat = z.reshape(-1, self.dim)
        # distances to codebook
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        z_q = self.embedding(idx).view_as(z)
        # losses
        commitment = F.mse_loss(z, z_q.detach())
        codebook = F.mse_loss(z_q, z.detach())
        loss = codebook + self.commitment_cost * commitment
        # straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, idx.view(*z.shape[:-1]), loss


class SlotVQ(nn.Module):
    """K slots from pooled molecule features → VQ → discrete S.

    If ``no_vq`` is True, skip quantization (continuous-S ablation): flat_S is
    pre-VQ slots, vq_loss=0, indices are zeros.
    """

    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 12,
        codebook_size: int = 128,
        commitment_cost: float = 0.25,
        no_vq: bool = False,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.no_vq = no_vq
        self.codebook_size = max(int(codebook_size), 1)
        self.slot_proj = nn.Linear(dim, num_slots * dim)
        self.vq = VectorQuantizer(self.codebook_size, dim, commitment_cost)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        # pooled: (batch, dim)
        b = pooled.shape[0]
        slots = self.slot_proj(pooled).view(b, self.num_slots, self.dim)
        if self.no_vq:
            zeros = torch.zeros(b, self.num_slots, dtype=torch.long, device=slots.device)
            return {
                "S": slots,
                "indices": zeros,
                "vq_loss": slots.new_zeros(()),
                "slots_pre_vq": slots,
                "flat_S": slots.reshape(b, -1),
            }
        z_q, indices, vq_loss = self.vq(slots)
        return {
            "S": z_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "slots_pre_vq": slots,
            "flat_S": z_q.reshape(b, -1),
        }
