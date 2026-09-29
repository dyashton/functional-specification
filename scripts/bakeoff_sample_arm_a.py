#!/usr/bin/env python3
"""Sample Arm A at given temperatures for the SMILES vs SELFIES bake-off.

Writes one CSV per (checkpoint, temperature, seed):
  smiles,source_smiles,arm,temperature,seed
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.data.smiles_tokenizer import MoleculeTokenizer
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.planner import FunctionalPlanner
from functionalspec.train.p2 import encode_flat_S


class LegacyArmA(nn.Module):
    """Pre-FiLM Arm A (h0-only cond) matching runs/p2 and runs/p2_selfies checkpoints."""

    def __init__(self, cond_dim: int, vocab_size: int = 128, hidden: int = 512, num_layers: int = 2):
        super().__init__()
        self.cond = nn.Linear(cond_dim, hidden)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def sample(
        self,
        flat_S: torch.Tensor,
        max_len: int = 120,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        b = flat_S.size(0)
        h = self.cond(flat_S).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        tok = torch.full((b, 1), bos_id, dtype=torch.long, device=flat_S.device)
        finished = torch.zeros(b, dtype=torch.bool, device=flat_S.device)
        outs = []
        for _ in range(max_len):
            emb = self.embed(tok[:, -1:])
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


def load_p2(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    tok = MoleculeTokenizer.from_dict(ckpt["tokenizer"])
    if "representation" in ckpt and hasattr(tok, "representation"):
        tok.representation = ckpt["representation"]  # type: ignore[assignment]
    surr_cols = ckpt["surrogate_cols"]
    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=int(mcfg["num_slots"]),
        codebook_size=int(ckpt["codebook_size"]),
        n_surrogates=len(surr_cols),
        commitment_cost=float(mcfg["commitment_cost"]),
        with_arm_a=False,
        with_arm_b=False,
    )
    cond_dim = int(mcfg["num_slots"]) * int(mcfg["hidden_dim"])
    has_film = any(k.startswith("arm_a.film") for k in ckpt["model"])
    arm_cls = SmilesConditionedDecoder if has_film else LegacyArmA
    kwargs: dict = {
        "cond_dim": cond_dim,
        "vocab_size": tok.vocab_size,
        "hidden": int(ckpt.get("arm_hidden", 512)),
        "num_layers": int(ckpt.get("arm_layers", 2)),
    }
    if has_film:
        kwargs["cond_dropout"] = float(ckpt.get("cond_dropout", 0.1))
    model.arm_a = arm_cls(**kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, tok, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm-name", type=str, required=True, help="Label e.g. arma_smiles / arma_selfies")
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-mols", type=int, default=64)
    p.add_argument("--samples-per-s", type=int, default=4)
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, ckpt = load_p2(args.checkpoint, device)
    df = load_split_csv(args.corpus_a / f"{args.split}.csv").head(args.n_mols)
    ds = MoleculeGraphDataset(df, ckpt["surrogate_cols"], ckpt["y_mean"], ckpt["y_std"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_graphs)
    max_len = int(ckpt.get("max_len", 150 if tok.representation == "selfies" else 120))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assert model.arm_a is not None
    for seed in args.seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        # Encode S once per seed (deterministic given seed + data order).
        flats: list[torch.Tensor] = []
        sources: list[str] = []
        with torch.no_grad():
            for batch in loader:
                flats.append(encode_flat_S(model, batch, device))
                sources.extend(batch["smiles"])
        flat_all = torch.cat(flats, dim=0)

        for temp in args.temperatures:
            rows: list[dict] = []
            with torch.no_grad():
                for _ in range(args.samples_per_s):
                    sampled = model.arm_a.sample(
                        flat_all, max_len=max_len, temperature=float(temp)
                    )
                    for i, row in enumerate(sampled):
                        rows.append(
                            {
                                "smiles": tok.decode_to_smiles(row),
                                "source_smiles": sources[i],
                                "arm": args.arm_name,
                                "temperature": float(temp),
                                "seed": int(seed),
                            }
                        )
            out = args.out_dir / f"{args.arm_name}_T{temp}_seed{seed}.csv"
            with out.open("w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["smiles", "source_smiles", "arm", "temperature", "seed"]
                )
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} → {out} (repr={tok.representation})", flush=True)


if __name__ == "__main__":
    main()
