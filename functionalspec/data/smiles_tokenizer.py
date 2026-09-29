"""Tokenizers for Arm A: character SMILES or SELFIES symbols."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

import selfies as sf
import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

Representation = Literal["smiles", "selfies"]


class MoleculeTokenizer:
    """Encode molecules as token ids; decode back to SMILES for eval."""

    def __init__(self, tokens: list[str] | None = None, representation: Representation = "selfies"):
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.representation: Representation = representation
        self.itos = specials + (tokens or [])
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: Iterable[str],
        representation: Representation = "selfies",
        min_count: int = 1,
    ) -> "MoleculeTokenizer":
        counts: Counter[str] = Counter()
        for smi in smiles_list:
            pieces = cls._tokenize_molecule(smi, representation)
            if pieces is None:
                continue
            counts.update(pieces)
        tokens = sorted([t for t, n in counts.items() if n >= min_count])
        return cls(tokens, representation=representation)

    # Back-compat
    from_smiles = from_smiles_list

    @staticmethod
    def _tokenize_molecule(smiles: str, representation: Representation) -> list[str] | None:
        if representation == "smiles":
            return list(smiles)
        try:
            se = sf.encoder(smiles)
        except Exception:
            return None
        if se is None:
            return None
        return list(sf.split_selfies(se))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str, max_len: int = 120) -> torch.Tensor | None:
        pieces = self._tokenize_molecule(smiles, self.representation)
        if pieces is None:
            return None
        ids = [BOS_ID] + [self.stoi.get(t, UNK_ID) for t in pieces] + [EOS_ID]
        ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def decode_to_smiles(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks: list[str] = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID, UNK_ID):
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok.startswith("<"):
                    continue
                toks.append(tok)
        if self.representation == "smiles":
            return "".join(toks)
        selfies_str = "".join(toks)
        try:
            return sf.decoder(selfies_str) or ""
        except Exception:
            return ""

    # Back-compat alias used by older sample scripts
    def decode(self, ids: list[int] | torch.Tensor) -> str:
        return self.decode_to_smiles(ids)

    def to_dict(self) -> dict:
        return {"itos": self.itos, "representation": self.representation}

    @classmethod
    def from_dict(cls, d: dict) -> "MoleculeTokenizer":
        specials = {"<pad>", "<bos>", "<eos>", "<unk>"}
        tokens = [t for t in d["itos"] if t not in specials]
        return cls(tokens, representation=d.get("representation", "smiles"))


# Back-compat name
SmilesTokenizer = MoleculeTokenizer


def pad_batch(seqs: list[torch.Tensor], pad_id: int = PAD_ID) -> torch.Tensor:
    t = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), t), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out
