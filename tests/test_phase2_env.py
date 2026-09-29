"""Phase II env + ContextFAN smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from functionalspec.env.encoder import EnvironmentEncoder
from functionalspec.env.graph_co2 import co2_only_environment
from functionalspec.env.types import InteractionEnvironment
from functionalspec.gen.objective import DesignObjective


def test_co2_only_environment_shapes():
    env = co2_only_environment()
    assert env.env_type == "co2"
    assert env.num_nodes == 3
    assert env.x.dim() == 2
    assert env.edge_index.size(0) == 2


def test_env_encoder_forward():
    env = co2_only_environment()
    enc = EnvironmentEncoder(out_dim=64, hidden_dim=64, num_layers=2)
    ir = enc.encode(env)
    assert ir.embedding.shape == (64,)
    assert torch.isfinite(ir.embedding).all()


def test_design_objective_not_on_environment():
    env = co2_only_environment()
    assert "LogP" not in env.meta
    obj = DesignObjective.from_string("LogP=3")
    assert "LogP" in obj.target_y


@pytest.mark.skipif(not Path("runs/gen/spec_bank.npz").exists(), reason="no bank")
def test_context_fan_plan_co2_only_without_ckpt():
    """Bank-only path: ContextFAN needs trained weights — skip if no ckpt."""
    if not Path("runs/phase2_co2/fan_context.pt").exists():
        pytest.skip("phase2 ckpt missing")
    from functionalspec.gen.fan_context import ContextFAN
    from functionalspec.gen.spec_bank import SpecBank

    bank = SpecBank(Path("runs/gen/spec_bank.npz"))
    fan = ContextFAN.load(Path("runs/phase2_co2/fan_context.pt"), bank)
    env = co2_only_environment()
    spec = fan.plan(DesignObjective(), environment=env, n_strategies=1)
    assert spec.outcome in ("exists", "uncertain", "unsupported")
    assert spec.flat_S.numel() == bank.flat_S.shape[1]
