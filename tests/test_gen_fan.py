"""Smoke tests for DesignObjective → FAN → Spec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from functionalspec.gen.fan import FunctionalAbstractionNetwork, _confidence
from functionalspec.gen.objective import DesignObjective
from functionalspec.gen.spec import FlatSPayload, FunctionalSpecification


def test_design_objective_parse():
    o = DesignObjective.from_string("LogP=3,TPSA=50")
    assert o.target_y["LogP"] == 3.0
    assert o.target_y["TPSA"] == 50.0
    s = DesignObjective.from_string("seed:CCO")
    assert s.seed_smiles == "CCO"


def test_confidence_ternary():
    conf, out = _confidence(R=1.0, density=0.5, dist=0.1, R_ref=1.0, dens_ref=0.5)
    assert out == "exists"
    assert conf > 0.5
    _, out2 = _confidence(R=0.1, density=0.1, dist=3.0, R_ref=1.0, dens_ref=0.5)
    assert out2 == "unsupported"


def test_functional_specification_flat_S():
    payload = FlatSPayload(flat_S=torch.randn(48))
    spec = FunctionalSpecification(
        payload=payload,
        predicted_behavior=np.zeros(3),
        confidence=0.7,
        outcome="exists",
        provenance="unit",
    )
    assert spec.flat_S.shape == (48,)


@pytest.mark.skipif(
    not Path("runs/gen/spec_bank.npz").exists(),
    reason="spec bank not built",
)
def test_fan_plan_logp():
    from functionalspec.gen.spec_bank import SpecBank

    bank = SpecBank(Path("runs/gen/spec_bank.npz"))
    fan = FunctionalAbstractionNetwork(bank)
    spec = fan.plan(DesignObjective.from_string("LogP=2"))
    assert spec.outcome in ("exists", "uncertain", "unsupported")
    assert spec.flat_S.numel() == bank.flat_S.shape[1]
    far = fan.plan(DesignObjective.from_string("LogP=99"))
    assert far.outcome == "unsupported"
