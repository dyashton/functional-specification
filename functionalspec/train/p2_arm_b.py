"""P2 training for Arm B: Spec → learned motifs → SELFIES molecule."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from functionalspec.config import load_config
from functionalspec.data.featurize import EDGE_DIM, NODE_DIM
from functionalspec.data.graph_dataset import (
    MoleculeGraphDataset,
    available_surrogates,
    collate_graphs,
    compute_y_stats,
    load_split_csv,
)
from functionalspec.data.motifs import MotifVocabulary, PAD_MOTIF
from functionalspec.data.smiles_tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    MoleculeTokenizer,
    Representation,
    pad_batch,
)
from functionalspec.eval.harness import e0_report
from functionalspec.models.generator_b import LearnedMotifDecoder
from functionalspec.models.planner import FunctionalPlanner


def _collate(batch: list[dict], tokenizer: MoleculeTokenizer, max_len: int) -> dict:
    out = collate_graphs(batch)
    seqs = []
    for smi in out["smiles"]:
        encoded = tokenizer.encode(smi, max_len=max_len)
        seqs.append(encoded if encoded is not None else torch.tensor([BOS_ID, EOS_ID]))
    out["tokens"] = pad_batch(seqs, pad_id=PAD_ID)
    return out


def _load_frozen_p1(
    checkpoint: Path,
    device: torch.device,
    n_surrogates: int,
) -> tuple[FunctionalPlanner, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or load_config()
    mcfg = cfg["model"]
    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=int(ckpt.get("num_slots", mcfg["num_slots"])),
        codebook_size=int(ckpt.get("codebook_size", mcfg["primary_codebook"])),
        n_surrogates=n_surrogates,
        commitment_cost=float(ckpt.get("commitment_cost", mcfg["commitment_cost"])),
        no_vq=bool(ckpt.get("no_vq", False)),
        with_arm_a=False,
        with_arm_b=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected P1 keys: {unexpected}")
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, ckpt


@torch.no_grad()
def _eval(
    model: FunctionalPlanner,
    loader: DataLoader,
    tokenizer: MoleculeTokenizer,
    device: torch.device,
    max_len: int,
    motif_loss_weight: float,
    vq_loss_weight: float,
) -> dict[str, float]:
    assert model.arm_b is not None
    model.eval()
    atom_total = motif_total = motif_correct = motif_count = valid_smiles = n_smiles = 0.0
    n_batches = 0
    for batch in loader:
        flat_s = _encode(model, batch, device)
        tokens = batch["tokens"].to(device)
        targets = batch["motif_targets"].to(device)
        out = model.arm_b(flat_s, tokens[:, :-1])
        atom_loss = F.cross_entropy(
            out["atom_logits"].reshape(-1, out["atom_logits"].size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
        )
        motif_loss = F.cross_entropy(
            out["fragment_logits"].reshape(-1, out["fragment_logits"].size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_MOTIF,
        )
        motif_mask = targets != PAD_MOTIF
        motif_correct += float(
            ((out["fragment_logits"].argmax(dim=-1) == targets) & motif_mask).sum()
        )
        motif_count += float(motif_mask.sum())
        atom_total += float(atom_loss)
        motif_total += float(motif_loss)
        if n_batches < 4:
            sampled, _ = model.arm_b.sample(
                flat_s, max_len=max_len, temperature=1.0
            )
            smiles = [tokenizer.decode_to_smiles(row) for row in sampled]
            report = e0_report([s for s in smiles if s])
            valid_smiles += report["validity"] * len(smiles)
            n_smiles += len(smiles)
        n_batches += 1
    return {
        "atom_loss": atom_total / max(n_batches, 1),
        "motif_loss": motif_total / max(n_batches, 1),
        "motif_accuracy": motif_correct / max(motif_count, 1.0),
        "sample_validity": valid_smiles / max(n_smiles, 1),
        "objective": atom_total / max(n_batches, 1)
        + motif_loss_weight * motif_total / max(n_batches, 1)
        + vq_loss_weight * 0.0,
    }


@torch.no_grad()
def _encode(model: FunctionalPlanner, batch: dict, device: torch.device) -> torch.Tensor:
    model.eval()
    return model.forward_graphs(
        batch["x"].to(device),
        batch["edge_index"].to(device),
        batch["edge_attr"].to(device),
        batch["batch"].to(device),
    )["flat_S"]


def train_arm_b(
    p1_checkpoint: Path,
    corpus_a_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    max_len: int | None = None,
    representation: Representation = "selfies",
    max_motifs: int = 8,
    motif_vocab_size: int = 256,
    motif_codebook: int = 256,
    motif_dim: int = 128,
    motif_loss_weight: float = 1.0,
    vq_loss_weight: float = 0.25,
    atom_hidden: int = 512,
    atom_layers: int = 2,
    cond_dropout: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_len = max_len or (150 if representation == "selfies" else 120)
    train_df = load_split_csv(corpus_a_dir / "train.csv")
    val_df = load_split_csv(corpus_a_dir / "val.csv")
    probe = torch.load(p1_checkpoint, map_location="cpu", weights_only=False)
    surr_cols = [c for c in probe["surrogate_cols"] if c in train_df.columns]
    y_mean, y_std = compute_y_stats(train_df, surr_cols)
    vocab = MotifVocabulary.from_smiles(
        train_df["SMILES"].astype(str),
        max_size=motif_vocab_size,
        max_motifs=max_motifs,
    )
    train_ds = MoleculeGraphDataset(train_df, surr_cols, y_mean, y_std, motif_vocab=vocab)
    val_ds = MoleculeGraphDataset(val_df, surr_cols, y_mean, y_std, motif_vocab=vocab)
    tokenizer = MoleculeTokenizer.from_smiles_list(train_ds.smiles, representation=representation)
    planner, p1_ckpt = _load_frozen_p1(p1_checkpoint, device_t, len(surr_cols))
    mcfg = (p1_ckpt.get("config") or load_config())["model"]
    cond_dim = int(p1_ckpt.get("num_slots", mcfg["num_slots"])) * int(mcfg["hidden_dim"])
    planner.arm_b = LearnedMotifDecoder(
        cond_dim=cond_dim,
        motif_vocab_size=vocab.size,
        atom_vocab_size=tokenizer.vocab_size,
        motif_codebook=motif_codebook,
        motif_dim=motif_dim,
        max_motifs=max_motifs,
        atom_hidden=atom_hidden,
        atom_layers=atom_layers,
        cond_dropout=cond_dropout,
    ).to(device_t)
    for parameter in planner.arm_b.parameters():
        parameter.requires_grad = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer, max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(batch, tokenizer, max_len),
    )
    optimizer = torch.optim.AdamW(planner.arm_b.parameters(), lr=lr, weight_decay=1e-4)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_valid = -1.0
    history = []
    best_path = out_dir / "arm_b_best.pt"
    for epoch in range(1, epochs + 1):
        planner.train()
        planner.encoder.eval()
        planner.slots.eval()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Arm B epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            flat_s = _encode(planner, batch, device_t)
            planner.arm_b.train()
            tokens = batch["tokens"].to(device_t)
            targets = batch["motif_targets"].to(device_t)
            out = planner.arm_b(flat_s, tokens[:, :-1])
            atom_loss = F.cross_entropy(
                out["atom_logits"].reshape(-1, out["atom_logits"].size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            motif_loss = F.cross_entropy(
                out["fragment_logits"].reshape(-1, out["fragment_logits"].size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_MOTIF,
            )
            loss = atom_loss + motif_loss_weight * motif_loss + vq_loss_weight * out["motif_vq_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.arm_b.parameters(), 5.0)
            optimizer.step()
            running += float(loss)
            pbar.set_postfix(loss=float(loss))
        val = _eval(
            planner, val_loader, tokenizer, device_t, max_len, motif_loss_weight, vq_loss_weight
        )
        row = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"epoch {epoch}: val_atom={val['atom_loss']:.4f} motif={val['motif_loss']:.4f} validity={val['sample_validity']:.3f}")
        score = val["sample_validity"] + 0.1 * val["motif_accuracy"]
        if score >= best_valid:
            best_valid = score
            torch.save(
                {
                    "model": planner.state_dict(),
                    "config": p1_ckpt.get("config") or load_config(),
                    "tokenizer": tokenizer.to_dict(),
                    "motif_vocab": vocab.to_dict(),
                    "surrogate_cols": surr_cols,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "num_slots": planner.num_slots,
                    "codebook_size": planner.codebook_size,
                    "commitment_cost": float(p1_ckpt.get("commitment_cost", 0.25)),
                    "no_vq": planner.no_vq,
                    "representation": representation,
                    "max_len": max_len,
                    "arm_b": True,
                    "motif_codebook": motif_codebook,
                    "motif_dim": motif_dim,
                    "max_motifs": max_motifs,
                    "atom_hidden": atom_hidden,
                    "atom_layers": atom_layers,
                    "cond_dropout": cond_dropout,
                },
                best_path,
            )
    return {"best_checkpoint": str(best_path), "history": history, "motif_vocab_size": vocab.size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-a", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--representation", choices=["selfies", "smiles"], default="selfies")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_arm_b(
        p1_checkpoint=args.p1_checkpoint,
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        representation=args.representation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
