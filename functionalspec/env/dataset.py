"""Load CO₂ posed complexes from pipeline energies.csv + complexes/*.xyz."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from functionalspec.env.graph_co2 import environment_from_complex_xyz
from functionalspec.env.types import InteractionEnvironment


@dataclass
class PosedComplex:
    mol_id: str
    smiles: str
    pose_id: str
    ie_kcal_mol: float
    xyz_path: Path
    run: str


def load_posed_complexes(
    co2_root: Path,
    run_names: tuple[str, ...] = ("full100", "small10", "test_n25", "pose_batch40"),
    *,
    best_per_smiles: bool = True,
) -> list[PosedComplex]:
    """Index posed host+CO₂ complexes with Psi4 IE labels."""
    root = Path(co2_root)
    rows: list[PosedComplex] = []
    for run in run_names:
        energies = root / "runs" / run / "energies.csv"
        if not energies.is_file():
            continue
        with energies.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                ie_s = (r.get("ie_kcal_mol") or "").strip()
                if not ie_s:
                    continue
                try:
                    ie = float(ie_s)
                except ValueError:
                    continue
                mol_id = r["mol_id"]
                pose_id = r["pose_id"]
                xyz = root / "runs" / run / "mols" / mol_id / "complexes" / f"{pose_id}.xyz"
                if not xyz.is_file():
                    continue
                rows.append(
                    PosedComplex(
                        mol_id=mol_id,
                        smiles=r.get("smiles") or "",
                        pose_id=pose_id,
                        ie_kcal_mol=ie,
                        xyz_path=xyz,
                        run=run,
                    )
                )
    if not best_per_smiles:
        return rows
    # Keep most negative IE per canonical smiles string
    best: dict[str, PosedComplex] = {}
    for p in rows:
        key = p.smiles or p.mol_id
        if key not in best or p.ie_kcal_mol < best[key].ie_kcal_mol:
            best[key] = p
    return list(best.values())


def environment_for_posed(p: PosedComplex) -> InteractionEnvironment:
    return environment_from_complex_xyz(p.xyz_path)


def ie_band(ie: float) -> str:
    if ie <= -6.0:
        return "strong"
    if ie >= -3.0:
        return "weak"
    return "mid"
