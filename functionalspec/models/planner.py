"""Planner bundle: encoder pooled features → SlotVQ → surrogate + contrastive APIs."""

from __future__ import annotations

import torch
import torch.nn as nn

from functionalspec.models.contrastive import ContrastiveProjector
from functionalspec.models.encoder import MoleculeEncoder
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.generator_b import LearnedMotifDecoder
from functionalspec.models.surrogate_heads import SurrogateHeads
from functionalspec.models.vq import SlotVQ


class FunctionalPlanner(nn.Module):
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_slots: int = 12,
        codebook_size: int = 128,
        n_surrogates: int = 8,
        commitment_cost: float = 0.25,
        no_vq: bool = False,
        with_arm_a: bool = True,
        with_arm_b: bool = False,
        arm_b_vocab_size: int | None = None,
        arm_b_atom_vocab_size: int | None = None,
        arm_b_motif_codebook: int = 256,
        arm_b_motif_dim: int = 128,
        arm_b_max_motifs: int = 8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.no_vq = no_vq
        self.encoder = MoleculeEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.slots = SlotVQ(
            hidden_dim, num_slots, codebook_size, commitment_cost, no_vq=no_vq
        )
        self.surrogate = SurrogateHeads(num_slots * hidden_dim, n_surrogates, hidden_dim)
        self.projector = ContrastiveProjector(num_slots * hidden_dim)
        self.arm_a = SmilesConditionedDecoder(num_slots * hidden_dim) if with_arm_a else None
        if with_arm_b:
            if arm_b_vocab_size is None or arm_b_atom_vocab_size is None:
                raise ValueError("Arm B requires motif and atom vocabulary sizes")
            self.arm_b = LearnedMotifDecoder(
                cond_dim=num_slots * hidden_dim,
                motif_vocab_size=arm_b_vocab_size,
                atom_vocab_size=arm_b_atom_vocab_size,
                motif_codebook=arm_b_motif_codebook,
                motif_dim=arm_b_motif_dim,
                max_motifs=arm_b_max_motifs,
            )
        else:
            self.arm_b = None

    def encode_pooled(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder.pooled(x, edge_index, edge_attr, batch=batch)

    def forward_from_pooled(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.slots(pooled)
        out["surrogate_pred"] = self.surrogate(out["flat_S"])
        out["z_proj"] = self.projector(out["flat_S"])
        return out

    def forward_graphs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self.encode_pooled(x, edge_index, edge_attr, batch=batch)
        return self.forward_from_pooled(pooled)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.slots.parameters():
            p.requires_grad = False
