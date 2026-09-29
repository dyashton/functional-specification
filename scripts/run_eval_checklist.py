#!/usr/bin/env python3
"""Print eval checklist and run offline metric self-checks (no trained weights required)."""

from __future__ import annotations

import numpy as np

from functionalspec.eval.harness import e0_report, e3_report, e6_token_delta, e8_path_report
from functionalspec.metrics.thresholds import diversity_behavior_ratio, structural_diversity


CHECKLIST = """
FunctionalSpec eval checklist
=============================
[ ] prepare_corpus_a / prepare_corpus_b
[ ] E0 validity on generated SMILES
[ ] E2 surrogate vs structure probes (+ Recon-VQ margin)
[ ] E3 1000-sample R with MATCHED-behavior baseline
[ ] E4 Arm A vs Arm B (optional)
[ ] E5 controllability
[ ] E6 token intervention
[ ] E7 token atlas
[ ] E8 S interpolation
[ ] E9 frozen low-n transfer
[ ] codebook sweep 32..512
"""


def self_check() -> None:
    # Synthetic: high structure diversity, low behavior variance → high R
    rng = np.random.default_rng(0)
    smiles = [f"C{'C' * (i % 5)}O" for i in range(50)]  # mostly invalid diversity stand-in
    # Use random surrogates with tiny variance for "ours"
    Y = rng.normal(size=(50, 4)) * 0.05 + np.array([1.0, 2.0, 3.0, 4.0])
    Yb = rng.normal(size=(50, 4)) * 0.05 + np.array([1.0, 2.0, 3.0, 4.0])
    # Force structural diversity metric path via e3_report (RDKit may invalidate many)
    r = e3_report(["CCO", "CCCO", "c1ccccc1", "CCN", "CC(=O)O"] * 10, Y[:50], ["CCO"] * 50, Yb)
    assert "ours" in r
    assert structural_diversity(0.2) == 0.8
    assert diversity_behavior_ratio(0.8, 0.1) > 1.0
    d = e6_token_delta(Y, Y + 1.0)
    assert "max_abs_delta" in d
    e8 = e8_path_report([1.0] * 10, np.linspace(0, 1, 10 * 3).reshape(10, 3))
    assert e8["status"] in {"pass", "fail"}
    e0 = e0_report(["CCO", "notasmiles", "c1ccccc1"])
    assert 0.0 <= e0["validity"] <= 1.0
    print("self_check: ok")


def main() -> None:
    print(CHECKLIST)
    self_check()


if __name__ == "__main__":
    main()
