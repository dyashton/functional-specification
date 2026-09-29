"""P2 training: freeze P1 encoder/VQ, train Arm A (S → SMILES)."""

from __future__ import annotations

import json
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
from functionalspec.data.smiles_tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    MoleculeTokenizer,
    Representation,
    pad_batch,
)
from functionalspec.eval.harness import e0_report
from functionalspec.models.generator_a import SmilesConditionedDecoder
from functionalspec.models.planner import FunctionalPlanner


def load_p1_planner(
    ckpt_path: Path,
    device: torch.device,
    vocab_size: int,
    arm_hidden: int = 512,
    arm_layers: int = 2,
    cond_dropout: float = 0.1,
) -> tuple[FunctionalPlanner, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or load_config()
    surr_cols = ckpt["surrogate_cols"]
    mcfg = cfg["model"]
    cb = int(ckpt.get("codebook_size", mcfg["primary_codebook"]))
    n_slots = int(ckpt.get("num_slots", mcfg["num_slots"]))
    beta = float(ckpt.get("commitment_cost", mcfg["commitment_cost"]))
    no_vq = bool(ckpt.get("no_vq", False))

    model = FunctionalPlanner(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_slots=n_slots,
        codebook_size=cb,
        n_surrogates=len(surr_cols),
        commitment_cost=beta,
        no_vq=no_vq,
        with_arm_a=False,
        with_arm_b=False,
    )
    # Load P1 weights (no arm_a in checkpoint)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading P1: {unexpected}")

    cond_dim = n_slots * int(mcfg["hidden_dim"])
    model.arm_a = SmilesConditionedDecoder(
        cond_dim=cond_dim,
        vocab_size=vocab_size,
        hidden=arm_hidden,
        num_layers=arm_layers,
        cond_dropout=cond_dropout,
    )
    model.to(device)

    # Freeze everything except Arm A
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("arm_a.")
    return model, ckpt


def collate_p2(batch: list[dict], tokenizer: MoleculeTokenizer, max_len: int = 150) -> dict:
    base = collate_graphs(batch)
    seqs = []
    for s in base["smiles"]:
        enc = tokenizer.encode(s, max_len=max_len)
        if enc is None:
            enc = torch.tensor([BOS_ID, EOS_ID], dtype=torch.long)
        seqs.append(enc)
    base["tokens"] = pad_batch(seqs, pad_id=PAD_ID)
    return base


@torch.no_grad()
def encode_flat_S(model: FunctionalPlanner, batch: dict, device: torch.device) -> torch.Tensor:
    model.eval()
    x = batch["x"].to(device)
    ei = batch["edge_index"].to(device)
    ea = batch["edge_attr"].to(device)
    b = batch["batch"].to(device)
    out = model.forward_graphs(x, ei, ea, b)
    return out["flat_S"]


def nll_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    tgt = tokens[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=PAD_ID,
    )


@torch.no_grad()
def eval_p2(
    model: FunctionalPlanner,
    loader: DataLoader,
    tokenizer: MoleculeTokenizer,
    device: torch.device,
    n_sample_batches: int = 4,
    temperature: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    n_tok = 0
    gen_smiles: list[str] = []

    for i, batch in enumerate(loader):
        flat_S = encode_flat_S(model, batch, device)
        tokens = batch["tokens"].to(device)
        assert model.arm_a is not None
        logits = model.arm_a(flat_S, tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )
        n = int((tokens[:, 1:] != PAD_ID).sum().item())
        total_nll += float(loss.item())
        n_tok += max(n, 1)

        if i < n_sample_batches:
            sampled = model.arm_a.sample(flat_S, max_len=tokenizer_max_len(tokenizer), temperature=temperature)
            for row in sampled:
                smi = tokenizer.decode_to_smiles(row)
                if smi:
                    gen_smiles.append(smi)

    e0 = e0_report(gen_smiles) if gen_smiles else {"validity": 0.0, "unique_rate": 0.0}
    ppl = float(np.exp(total_nll / max(n_tok, 1)))
    return {
        "nll_per_token": total_nll / max(n_tok, 1),
        "perplexity": ppl,
        "sample_validity": float(e0["validity"]),
        "sample_unique_rate": float(e0["unique_rate"]),
        "n_samples": float(len(gen_smiles)),
    }


def tokenizer_max_len(tokenizer: MoleculeTokenizer) -> int:
    return 150 if tokenizer.representation == "selfies" else 120


def train_p2(
    p1_checkpoint: Path,
    corpus_a_dir: Path,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    max_len: int | None = None,
    arm_hidden: int = 512,
    arm_layers: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    representation: Representation = "selfies",
    cond_dropout: float = 0.1,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_len = max_len or (150 if representation == "selfies" else 120)

    train_df = load_split_csv(corpus_a_dir / "train.csv")
    val_df = load_split_csv(corpus_a_dir / "val.csv")

    ckpt_probe = torch.load(p1_checkpoint, map_location="cpu", weights_only=False)
    surr_cols = ckpt_probe["surrogate_cols"]
    y_mean = np.asarray(ckpt_probe["y_mean"], dtype=np.float64)
    y_std = np.asarray(ckpt_probe["y_std"], dtype=np.float64)
    surr_cols = [c for c in surr_cols if c in train_df.columns]
    if len(surr_cols) != len(ckpt_probe["surrogate_cols"]):
        surr_cols = available_surrogates(train_df)
        y_mean, y_std = compute_y_stats(train_df, surr_cols)

    train_ds = MoleculeGraphDataset(train_df, surr_cols, y_mean, y_std)
    val_ds = MoleculeGraphDataset(val_df, surr_cols, y_mean, y_std)
    tokenizer = MoleculeTokenizer.from_smiles_list(train_ds.smiles, representation=representation)

    model, p1_ckpt = load_p1_planner(
        p1_checkpoint,
        device_t,
        vocab_size=tokenizer.vocab_size,
        arm_hidden=arm_hidden,
        arm_layers=arm_layers,
        cond_dropout=cond_dropout,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_p2(b, tokenizer, max_len=max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_p2(b, tokenizer, max_len=max_len),
    )

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_valid = -1.0
    best_path = out_dir / "p2_best.pt"

    print(f"P2 representation={representation} vocab_size={tokenizer.vocab_size} max_len={max_len} cond_dropout={cond_dropout}")

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()
        model.slots.eval()
        running = 0.0
        n_tok = 0
        pbar = tqdm(train_loader, desc=f"P2 epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            with torch.no_grad():
                flat_S = encode_flat_S(model, batch, device_t)
            tokens = batch["tokens"].to(device_t)
            assert model.arm_a is not None
            model.arm_a.train()
            logits = model.arm_a(flat_S, tokens[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.arm_a.parameters(), 5.0)
            opt.step()

            nt = int((tokens[:, 1:] != PAD_ID).sum().item())
            running += float(loss.item()) * nt
            n_tok += max(nt, 1)
            pbar.set_postfix(nll=float(loss.item()))

        val = eval_p2(model, val_loader, tokenizer, device_t)
        row = {
            "epoch": float(epoch),
            "train_nll": running / max(n_tok, 1),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch}: val_ppl={val['perplexity']:.2f} "
            f"validity={val['sample_validity']:.3f} unique={val['sample_unique_rate']:.3f}"
        )

        score = float(val["sample_validity"])
        if score >= best_valid:
            best_valid = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "representation": representation,
                    "surrogate_cols": surr_cols,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "p1_checkpoint": str(p1_checkpoint),
                    "codebook_size": p1_ckpt.get("codebook_size"),
                    "num_slots": p1_ckpt.get("num_slots"),
                    "commitment_cost": p1_ckpt.get("commitment_cost"),
                    "no_vq": p1_ckpt.get("no_vq", False),
                    "config": p1_ckpt.get("config"),
                    "epoch": epoch,
                    "val": val,
                    "arm_hidden": arm_hidden,
                    "arm_layers": arm_layers,
                    "max_len": max_len,
                    "cond_dropout": cond_dropout,
                    "conditioning": "film_per_step",
                },
                best_path,
            )

    meta = {
        "best_sample_validity": best_valid,
        "best_checkpoint": str(best_path),
        "representation": representation,
        "vocab_size": tokenizer.vocab_size,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "history": history,
    }
    (out_dir / "p2_history.json").write_text(json.dumps(meta, indent=2))
    return meta
