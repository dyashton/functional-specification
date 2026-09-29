#!/usr/bin/env python3
"""Sample molecules from frozen S using a P2 checkpoint (E0 / E3 helper)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import MoleculeGraphDataset, collate_graphs, load_split_csv
from functionalspec.data.smiles_tokenizer import MoleculeTokenizer
from functionalspec.eval.harness import dump_json, e0_report
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.planner import FunctionalPlanner
from functionalspec.train.p2 import encode_flat_S


def load_p2(ckpt_path: Path, device: torch.device) -> tuple[FunctionalPlanner, MoleculeTokenizer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    tok = MoleculeTokenizer.from_dict(ckpt["tokenizer"])
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
    model.arm_a = SmilesConditionedDecoder(
        cond_dim=cond_dim,
        vocab_size=tok.vocab_size,
        hidden=int(ckpt.get("arm_hidden", 512)),
        num_layers=int(ckpt.get("arm_layers", 2)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, tok, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/p2_selfies/p2_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-mols", type=int, default=256, help="Number of source molecules to encode")
    p.add_argument("--samples-per-s", type=int, default=4, help="Samples per frozen S")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("runs/p2_selfies/samples.csv"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, ckpt = load_p2(args.checkpoint, device)
    df = load_split_csv(args.corpus_a / f"{args.split}.csv").head(args.n_mols)
    ds = MoleculeGraphDataset(
        df,
        ckpt["surrogate_cols"],
        ckpt["y_mean"],
        ckpt["y_std"],
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_graphs)

    max_len = int(ckpt.get("max_len", 150 if tok.representation == "selfies" else 120))
    rows = []
    assert model.arm_a is not None
    with torch.no_grad():
        for batch in loader:
            flat_S = encode_flat_S(model, batch, device)
            src = batch["smiles"]
            for _ in range(args.samples_per_s):
                sampled = model.arm_a.sample(flat_S, max_len=max_len, temperature=args.temperature)
                for i, row in enumerate(sampled):
                    rows.append({"source_smiles": src[i], "smiles": tok.decode_to_smiles(row)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_smiles", "smiles"])
        w.writeheader()
        w.writerows(rows)

    report = e0_report([r["smiles"] for r in rows if r["smiles"]])
    dump_json(report, args.out.with_suffix(".e0.json"))
    print(f"representation={tok.representation}")
    print(f"wrote {len(rows)} samples → {args.out}")
    print(report)


if __name__ == "__main__":
    main()
