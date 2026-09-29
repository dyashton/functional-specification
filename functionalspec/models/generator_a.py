"""Arm A: Spec payload (flat_S) → SELFIES/SMILES decoder with per-step conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmilesConditionedDecoder(nn.Module):
    """GRU LM with FiLM conditioning from flat_S at every step (not only h0).

    Optional cond dropout (CFG-style) during training; at sample time use full cond
    or scale via cond_scale.
    """

    def __init__(
        self,
        cond_dim: int,
        vocab_size: int = 128,
        hidden: int = 512,
        num_layers: int = 2,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.cond_dropout = cond_dropout
        self.cond = nn.Linear(cond_dim, hidden)
        self.film = nn.Sequential(nn.Linear(cond_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden * 2))
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def _condition(self, flat_S: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns h0 (L,B,H), gamma (B,H), beta (B,H)."""
        if training and self.cond_dropout > 0:
            drop = torch.rand(flat_S.size(0), device=flat_S.device) < self.cond_dropout
            flat_S = flat_S.clone()
            flat_S[drop] = 0.0
        h0 = self.cond(flat_S).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        gb = self.film(flat_S)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        return h0, gamma, beta

    def _film_emb(self, emb: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # emb: (B, T, H) or (B, 1, H); gamma/beta: (B, H)
        return emb * gamma.unsqueeze(1) + beta.unsqueeze(1)

    def forward(
        self,
        flat_S: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        h0, gamma, beta = self._condition(flat_S, training=self.training)
        emb = self._film_emb(self.embed(tokens), gamma, beta)
        out, _ = self.rnn(emb, h0)
        return self.out(out)

    @torch.no_grad()
    def sample(
        self,
        flat_S: torch.Tensor,
        max_len: int = 120,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        temperature: float = 1.0,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        b = flat_S.size(0)
        # CFG-lite: blend toward unconditional (zeros) if cond_scale != 1
        if abs(cond_scale - 1.0) > 1e-6:
            flat_S = cond_scale * flat_S
        h0, gamma, beta = self._condition(flat_S, training=False)
        h = h0
        tok = torch.full((b, 1), bos_id, dtype=torch.long, device=flat_S.device)
        finished = torch.zeros(b, dtype=torch.bool, device=flat_S.device)
        outs = []
        for _ in range(max_len):
            emb = self._film_emb(self.embed(tok[:, -1:]), gamma, beta)
            out, h = self.rnn(emb, h)
            logits = self.out(out[:, -1]) / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, pad_id), nxt)
            outs.append(nxt)
            finished = finished | (nxt.squeeze(1) == eos_id)
            tok = torch.cat([tok, nxt], dim=1)
            if bool(finished.all()):
                break
        return torch.cat(outs, dim=1) if outs else tok
