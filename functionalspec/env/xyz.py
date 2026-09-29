"""Minimal XYZ I/O for host+CO₂ complexes (last 3 atoms = CO₂)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Xyz:
    symbols: list[str]
    coords: np.ndarray  # (N, 3)


def read_xyz(path: Path) -> Xyz:
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0].split()[0])
    syms, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        syms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Xyz(symbols=syms, coords=np.asarray(coords, dtype=np.float64))


def split_host_co2(xyz: Xyz) -> tuple[Xyz, Xyz]:
    """Convention from co2ie: last three atoms are C, O, O."""
    if len(xyz.symbols) < 4:
        raise ValueError("complex too small to split host/CO2")
    host = Xyz(symbols=xyz.symbols[:-3], coords=xyz.coords[:-3])
    co2 = Xyz(symbols=xyz.symbols[-3:], coords=xyz.coords[-3:])
    return host, co2


# Covalent distance cutoffs (Å) by element pair (symmetric lookup via frozenset)
_COV = {
    frozenset({"H", "H"}): 1.0,
    frozenset({"H", "C"}): 1.2,
    frozenset({"H", "N"}): 1.15,
    frozenset({"H", "O"}): 1.15,
    frozenset({"C", "C"}): 1.7,
    frozenset({"C", "N"}): 1.65,
    frozenset({"C", "O"}): 1.55,
    frozenset({"N", "N"}): 1.55,
    frozenset({"N", "O"}): 1.5,
    frozenset({"O", "O"}): 1.5,
}


def covalent_cutoff(a: str, b: str) -> float:
    return _COV.get(frozenset({a, b}), 1.8)
