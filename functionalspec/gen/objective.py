"""Design objectives for FAN_MVP (extensible stand-in for environments)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DesignObjective:
    """What the molecule must accomplish — MVP: descriptors / lead / multi-obj.

    Phase II: replace or extend with environment-conditioned objectives.
    """

    target_y: dict[str, float] = field(default_factory=dict)
    seed_smiles: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    name: str = ""

    def vector(self, cols: list[str], y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray | None:
        """Return z-scored target aligned to cols, or None if no descriptor targets."""
        if not self.target_y:
            return None
        y = np.zeros(len(cols), dtype=np.float64)
        mask = np.zeros(len(cols), dtype=bool)
        for i, c in enumerate(cols):
            if c in self.target_y:
                y[i] = (float(self.target_y[c]) - y_mean[i]) / y_std[i]
                mask[i] = True
        if not mask.any():
            return None
        return y

    def dim_weights(self, cols: list[str]) -> np.ndarray:
        w = np.ones(len(cols), dtype=np.float64)
        for i, c in enumerate(cols):
            if c in self.weights:
                w[i] = float(self.weights[c])
            elif self.target_y and c not in self.target_y:
                w[i] = 0.0
        if w.sum() <= 0:
            return np.ones(len(cols), dtype=np.float64)
        return w

    @classmethod
    def from_string(cls, s: str) -> "DesignObjective":
        """Parse 'LogP=3,TPSA=50' or 'seed:CCO'."""
        s = s.strip()
        if s.lower().startswith("seed:"):
            return cls(seed_smiles=s.split(":", 1)[1].strip(), name="seed")
        target: dict[str, float] = {}
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Bad objective fragment {part!r}; use Key=value")
            k, v = part.split("=", 1)
            target[k.strip()] = float(v.strip())
        return cls(target_y=target, name=s)
