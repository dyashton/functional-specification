"""Train capacity-matched property→SELFIES decoder (G6a Model B)."""

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
from functionalspec.data.graph_dataset import (
    MoleculeGraphDataset,
    available_surrogates,
    compute_y_stats,
    load_split_csv,
)
from functionalspec.data.smiles_tokenizer import PAD_ID, MoleculeTokenizer, Representation
from functionalspec.eval.harness import e0_report
from functionalspec.models.property_decoder import PropertyConditionedDecoder
from functionalspec.train.p2 import collate_p2, tokenizer_max_len


@torch.no_grad()
def eval_property_p2(
    model: PropertyConditionedDecoder,
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
        y = batch["y"].to(device)
        tokens = batch["tokens"].to(device)
        logits = model(y, tokens[:, :-1])
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
            sampled = model.sample(y, max_len=tokenizer_max_len(tokenizer), temperature=temperature)
            for row in sampled:
                smi = tokenizer.decode_to_smiles(row)
                if smi:
                    gen_smiles.append(smi)
    e0 = e0_report(gen_smiles) if gen_smiles else {"validity": 0.0, "unique_rate": 0.0}
    return {
        "nll_per_token": total_nll / max(n_tok, 1),
        "perplexity": float(np.exp(total_nll / max(n_tok, 1))),
        "sample_validity": float(e0["validity"]),
        "sample_unique_rate": float(e0["unique_rate"]),
        "n_samples": float(len(gen_smiles)),
    }


def train_p2_property(
    p1_checkpoint: Path,
    corpus_a_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    max_len: int | None = None,
    arm_hidden: int = 512,
    arm_layers: int = 2,
    cond_dropout: float = 0.1,
    representation: Representation = "selfies",
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Matched recipe to P2 FiLM Arm A, but condition on z-scored surrogates."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_len = max_len or (150 if representation == "selfies" else 120)

    p1 = torch.load(p1_checkpoint, map_location="cpu", weights_only=False)
    cfg = p1.get("config") or load_config()
    mcfg = cfg["model"]
    n_slots = int(p1.get("num_slots", mcfg["num_slots"]))
    cond_dim = n_slots * int(mcfg["hidden_dim"])

    train_df = load_split_csv(corpus_a_dir / "train.csv")
    val_df = load_split_csv(corpus_a_dir / "val.csv")
    surr_cols = list(p1["surrogate_cols"])
    y_mean = np.asarray(p1["y_mean"], dtype=np.float64)
    y_std = np.asarray(p1["y_std"], dtype=np.float64)
    surr_cols = [c for c in surr_cols if c in train_df.columns]
    if len(surr_cols) != len(p1["surrogate_cols"]):
        surr_cols = available_surrogates(train_df)
        y_mean, y_std = compute_y_stats(train_df, surr_cols)

    train_ds = MoleculeGraphDataset(train_df, surr_cols, y_mean, y_std)
    val_ds = MoleculeGraphDataset(val_df, surr_cols, y_mean, y_std)
    tokenizer = MoleculeTokenizer.from_smiles_list(train_ds.smiles, representation=representation)

    model = PropertyConditionedDecoder(
        n_y=len(surr_cols),
        cond_dim=cond_dim,
        vocab_size=tokenizer.vocab_size,
        hidden=arm_hidden,
        num_layers=arm_layers,
        cond_dropout=cond_dropout,
    ).to(device_t)

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

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "p2_property_best.pt"
    best_valid = -1.0
    history: list[dict[str, float]] = []

    n_s = sum(p.numel() for p in model.parameters())
    print(
        f"P2-property n_y={len(surr_cols)} cond_dim={cond_dim} params={n_s} "
        f"vocab={tokenizer.vocab_size} cond_dropout={cond_dropout}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_tok = 0
        pbar = tqdm(train_loader, desc=f"P2-y epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            y = batch["y"].to(device_t)
            tokens = batch["tokens"].to(device_t)
            logits = model(y, tokens[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            nt = int((tokens[:, 1:] != PAD_ID).sum().item())
            running += float(loss.item()) * nt
            n_tok += max(nt, 1)
            pbar.set_postfix(nll=float(loss.item()))

        val = eval_property_p2(model, val_loader, tokenizer, device_t)
        history.append({"epoch": float(epoch), "train_nll": running / max(n_tok, 1), **{f"val_{k}": v for k, v in val.items()}})
        print(
            f"epoch {epoch}: val_ppl={val['perplexity']:.2f} "
            f"validity={val['sample_validity']:.3f} unique={val['sample_unique_rate']:.3f}"
        )
        if float(val["sample_validity"]) >= best_valid:
            best_valid = float(val["sample_validity"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "representation": representation,
                    "surrogate_cols": surr_cols,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "cond_dim": cond_dim,
                    "arm_hidden": arm_hidden,
                    "arm_layers": arm_layers,
                    "max_len": max_len,
                    "cond_dropout": cond_dropout,
                    "conditioning": "property_y_film",
                    "epoch": epoch,
                    "val": val,
                    "p1_checkpoint": str(p1_checkpoint),
                    "config": cfg,
                },
                best_path,
            )

    meta = {
        "best_sample_validity": best_valid,
        "best_checkpoint": str(best_path),
        "n_params": n_s,
        "cond_dim": cond_dim,
        "history": history,
    }
    (out_dir / "p2_property_history.json").write_text(json.dumps(meta, indent=2))
    return meta
