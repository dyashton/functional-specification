"""Unit tests for splits, metrics, and contrastive weighting."""

from __future__ import annotations

import numpy as np
import torch

from functionalspec.data.splits import scaffold_split
from functionalspec.metrics.thresholds import (
    behavioral_variance,
    diversity_behavior_ratio,
    e3_pass,
    structural_diversity,
)
from functionalspec.models.contrastive import multi_view_positive_weight
from functionalspec.models.vq import SlotVQ


def test_scaffold_split_sizes():
    smiles = ["CCO", "CCCO", "c1ccccc1", "c1ccccc1O", "CCN", "CCCN", "O=C=O", "C"]
    # duplicate scaffolds
    smiles = smiles + smiles
    sp = scaffold_split(smiles, 0.5, 0.25, 0.25, seed=0)
    n = len(smiles)
    assert set(sp["train"]) | set(sp["val"]) | set(sp["test"]) == set(range(n))
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == n


def test_diversity_ratio():
    assert structural_diversity(0.25) == 0.75
    assert abs(diversity_behavior_ratio(0.75, 0.25) - 3.0) < 1e-4
    Y = np.ones((10, 3))
    assert behavioral_variance(Y) == 0.0


def test_e3_pass_logic():
    ok, _ = e3_pass(n_scaffolds=40, v_beh=0.2, r=3.0, r_base=2.0, n_scaf_base=20)
    assert ok
    ok2, reasons = e3_pass(n_scaffolds=3, v_beh=0.2, r=3.0, r_base=2.0, n_scaf_base=20)
    assert not ok2
    assert reasons


def test_contrastive_subtracts_ecfp():
    one = torch.ones(4)
    zero = torch.zeros(4)
    w_close_chem = multi_view_positive_weight(one, None, None, None, sim_ecfp=one)
    w_far_chem = multi_view_positive_weight(one, None, None, None, sim_ecfp=zero)
    assert torch.all(w_far_chem > w_close_chem)


def test_slot_vq_shapes():
    m = SlotVQ(dim=32, num_slots=4, codebook_size=16)
    pooled = torch.randn(2, 32)
    out = m(pooled)
    assert out["S"].shape == (2, 4, 32)
    assert out["indices"].shape == (2, 4)
    assert out["flat_S"].shape == (2, 128)
