"""BRICS-derived fragment targets for the learned-motif Arm B baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from rdkit.Chem import BRICS

from functionalspec.data.descriptors import mol_from_smiles


PAD_MOTIF = 0
UNK_MOTIF = 1


def extract_brics_fragments(smiles: str) -> list[str]:
    """Return deterministic BRICS fragment strings for a valid molecule."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    try:
        return sorted(str(fragment) for fragment in BRICS.BRICSDecompose(mol))
    except Exception:
        return []


@dataclass
class MotifVocabulary:
    """Fixed BRICS candidate vocabulary used to supervise learned motif slots."""

    tokens: list[str]
    max_motifs: int = 8

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str],
        *,
        max_size: int = 256,
        max_motifs: int = 8,
    ) -> "MotifVocabulary":
        counts: Counter[str] = Counter()
        for smi in smiles:
            counts.update(extract_brics_fragments(smi))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return cls(
            tokens=[fragment for fragment, _ in ordered[: max(max_size - 2, 0)]],
            max_motifs=max_motifs,
        )

    @property
    def size(self) -> int:
        return len(self.tokens) + 2

    def encode(self, smiles: str) -> list[int]:
        counts = Counter(extract_brics_fragments(smiles))
        ids = [
            self.tokens.index(fragment) + 2 if fragment in self.tokens else UNK_MOTIF
            for fragment, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        ids = ids[: self.max_motifs]
        return ids + [PAD_MOTIF] * (self.max_motifs - len(ids))

    def to_dict(self) -> dict:
        return {"tokens": self.tokens, "max_motifs": self.max_motifs}

    @classmethod
    def from_dict(cls, payload: dict) -> "MotifVocabulary":
        return cls(tokens=list(payload["tokens"]), max_motifs=int(payload["max_motifs"]))
