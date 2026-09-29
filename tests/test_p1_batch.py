"""Smoke tests for graph featurization and P1 batching."""

from __future__ import annotations

import torch

from functionalspec.data.featurize import smiles_to_graph
from functionalspec.data.graph_dataset import collate_graphs
from functionalspec.models.planner import FunctionalPlanner


def test_smiles_to_graph_and_batch_encode():
    g1 = smiles_to_graph("CCO")
    g2 = smiles_to_graph("c1ccccc1")
    assert g1 is not None and g2 is not None
    batch = collate_graphs(
        [
            {
                "smiles": "CCO",
                "x": g1["x"],
                "edge_index": g1["edge_index"],
                "edge_attr": g1["edge_attr"],
                "y": torch.zeros(3),
                "fp": torch.zeros(32),
            },
            {
                "smiles": "c1ccccc1",
                "x": g2["x"],
                "edge_index": g2["edge_index"],
                "edge_attr": g2["edge_attr"],
                "y": torch.zeros(3),
                "fp": torch.zeros(32),
            },
        ]
    )
    model = FunctionalPlanner(
        node_dim=64,
        edge_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_slots=4,
        codebook_size=16,
        n_surrogates=3,
        with_arm_a=False,
        with_arm_b=False,
    )
    out = model.forward_graphs(batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch"])
    assert out["flat_S"].shape[0] == 2
    assert out["surrogate_pred"].shape == (2, 3)
