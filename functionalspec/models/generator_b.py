"""Arm B: S → learned motif codes → motif-conditioned atom expansion."""

from __future__ import annotations

import torch
import torch.nn as nn

from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.vq import VectorQuantizer


class LearnedMotifDecoder(nn.Module):
    """Predict a fixed-length sequence of learned motif codebook IDs from S.

    Motif vocabulary is data-driven (VQ over fragment embeddings), not BRICS.
    Atom expansion is left to an external expander hook.
    """

    def __init__(
        self,
        cond_dim: int,
        motif_vocab_size: int,
        atom_vocab_size: int,
        motif_codebook: int = 256,
        motif_dim: int = 128,
        max_motifs: int = 8,
        hidden: int = 256,
        atom_hidden: int = 512,
        atom_layers: int = 2,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.motif_vocab_size = motif_vocab_size
        self.max_motifs = max_motifs
        self.motif_dim = motif_dim
        self.cond = nn.Linear(cond_dim, hidden)
        self.query = nn.Parameter(torch.randn(max_motifs, hidden) * 0.02)
        self.to_motif = nn.Linear(hidden, motif_dim)
        self.vq = VectorQuantizer(motif_codebook, motif_dim)
        self.fragment_head = nn.Linear(motif_dim, motif_vocab_size)
        self.atom_decoder = SmilesConditionedDecoder(
            cond_dim=motif_dim * max_motifs,
            vocab_size=atom_vocab_size,
            hidden=atom_hidden,
            num_layers=atom_layers,
            cond_dropout=cond_dropout,
        )

    def encode_motifs(self, flat_S: torch.Tensor) -> dict[str, torch.Tensor]:
        b = flat_S.size(0)
        cond = self.cond(flat_S).unsqueeze(1)  # (B,1,H)
        q = self.query.unsqueeze(0).expand(b, -1, -1) + cond
        pre = self.to_motif(q)
        z_q, indices, vq_loss = self.vq(pre)
        fragment_logits = self.fragment_head(z_q)
        return {
            "motif_S": z_q,
            "motif_indices": indices,
            "fragment_logits": fragment_logits,
            "predicted_fragment_ids": fragment_logits.argmax(dim=-1),
            "motif_vq_loss": vq_loss,
            "context": z_q.reshape(b, -1),
        }

    def forward(
        self,
        flat_S: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encode_motifs(flat_S)
        if tokens is not None:
            out["atom_logits"] = self.atom_decoder(out["context"], tokens)
        return out

    @torch.no_grad()
    def sample(
        self,
        flat_S: torch.Tensor,
        *,
        max_len: int = 150,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encode_motifs(flat_S)
        tokens = self.atom_decoder.sample(
            out["context"],
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            temperature=temperature,
        )
        return tokens, out["predicted_fragment_ids"]
