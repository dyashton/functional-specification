"""Functional Specification interface (payload-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class FlatSPayload:
    """Today's Spec payload: continuous flat_S from P1 slot VQ."""

    flat_S: torch.Tensor  # (cond_dim,) or (1, cond_dim)
    token_indices: np.ndarray | None = None  # optional (K,) slot codebook ids


@dataclass
class FunctionalSpecification:
    """Public conditioning object for the Generator.

    `payload` is today's implementation (flat_S); Phase II may swap payload type
    without changing Generator callers that go through generate_from_spec.
    """

    payload: FlatSPayload
    predicted_behavior: np.ndarray  # raw surrogate space (not z-scored), aligned to bank cols
    confidence: float
    outcome: Literal["exists", "uncertain", "unsupported"]
    provenance: str
    specificity_R_internal: float = float("nan")
    behavior_distance: float = float("nan")
    surrogate_cols: tuple[str, ...] = ()
    y_mean: np.ndarray | None = None  # for z-scored behavior ball filter
    y_std: np.ndarray | None = None

    @property
    def flat_S(self) -> torch.Tensor:
        s = self.payload.flat_S
        return s if s.dim() == 1 else s.reshape(-1)
