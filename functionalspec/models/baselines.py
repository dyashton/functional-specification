"""Matched-capacity baselines for E2/E3/E9."""

from __future__ import annotations

import torch
import torch.nn as nn

from functionalspec.models.vq import VectorQuantizer
from functionalspec.metrics.thresholds import behavior_match_ok


class ReconVQBottleneck(nn.Module):
    """Structure-oriented VQ bottleneck (reconstruction training — control only)."""

    def __init__(self, in_dim: int, codebook_size: int = 128, dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.vq = VectorQuantizer(codebook_size, dim)
        self.dec = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, in_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.enc(x)
        z_q, idx, vq_loss = self.vq(z)
        recon = self.dec(z_q)
        return {"z": z_q, "indices": idx, "vq_loss": vq_loss, "recon": recon}


class ContinuousFunctionLatent(nn.Module):
    """Continuous z with surrogate heads — no discrete tokens."""

    def __init__(self, in_dim: int, latent_dim: int = 256, n_targets: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.head = nn.Linear(latent_dim, n_targets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.enc(x)
        return {"z": z, "pred": self.head(z)}


def filter_matched_behavior(
    candidate_surrogates: torch.Tensor,
    target_mean: torch.Tensor,
    tol: float = 0.15,
    max_keep: int | None = None,
) -> torch.Tensor:
    """Return boolean mask of candidates whose surrogate vector is near target_mean.

    For set-level E3 matching, prefer resampling until set-mean matches; this helper
    supports per-molecule filtering as a first approximation.
    """
    d = candidate_surrogates.size(-1)
    dist = (candidate_surrogates - target_mean).pow(2).sum(-1).sqrt() / (d**0.5)
    mask = dist <= tol
    if max_keep is not None and mask.sum() > max_keep:
        # keep closest
        idx = torch.argsort(dist)[:max_keep]
        out = torch.zeros_like(mask)
        out[idx] = True
        return out
    return mask


def set_mean_matched(
    surrogates: torch.Tensor,
    target_mean: torch.Tensor,
    tol: float = 0.15,
) -> bool:
    mean = surrogates.mean(dim=0)
    return behavior_match_ok(mean.detach().cpu().numpy(), target_mean.detach().cpu().numpy(), tol=tol)
