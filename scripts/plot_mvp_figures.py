#!/usr/bin/env python3
"""MVP paper figures: pipeline placeholder, E2, neighborhood cartoon, G5 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from functionalspec.data.descriptors import compute_surrogates


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.{{png,pdf}}")


def _mol_img(smiles: str, size=(220, 180)):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc is not None and sc.GetNumAtoms() >= 3:
            m = sc
    except Exception:
        pass
    return Draw.MolToImage(m, size=size)


def plot_pipeline_placeholder(out: Path) -> None:
    """Simple boxes the author can replace with a designed schematic."""
    fig, ax = plt.subplots(figsize=(10.5, 2.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    boxes = [
        (0.3, "DesignObjective\n(user wants)"),
        (2.5, "FAN\n(bank retrieve)"),
        (4.7, "Functional\nSpecification S"),
        (6.9, "FiLM\nGenerator"),
        (9.1, "Molecules"),
    ]
    for x, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.85, 0.55),
                1.7,
                1.3,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="#f3f4f6",
                edgecolor="#9ca3af",
                lw=1.4,
                linestyle="--",
            )
        )
        ax.text(x, 1.2, label, ha="center", va="center", fontsize=9, color="#4b5563")
    for x0, x1 in [(1.15, 1.65), (3.35, 3.85), (5.55, 6.05), (7.75, 8.25)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.2),
                (x1, 1.2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#9ca3af",
                lw=1.2,
            )
        )
    ax.text(
        5.25,
        0.22,
        "PLACEHOLDER — replace with designed pipeline schematic",
        ha="center",
        va="center",
        fontsize=9,
        color="#b45309",
        style="italic",
    )
    ax.set_title("MVP stack (placeholder)", fontsize=11, pad=6)
    _save(fig, out)


def plot_e2(summary: Path, out: Path) -> None:
    d = json.loads(summary.read_text())
    labels = [
        "Surrogate\n(S)",
        "Structure\n(S)",
        "Structure\n(FP-PCA)",
        "Surrogate\n(FP)",
    ]
    vals = [
        d["surrogate_probe"]["mean_spearman"],
        d["structure_probe"]["mean_spearman"],
        d["fp_pca_structure_probe"]["mean_spearman"],
        d["fp_surrogate_probe"]["mean_spearman"],
    ]
    colors = ["#0f766e", "#0f766e", "#6b7280", "#6b7280"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.65, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Spearman / AUROC proxy")
    ax.set_title("E2: S is not a fingerprint bottleneck")
    gap = d["gap_spearman"]
    margin = d["structure_margin"]
    ax.annotate(
        f"behavior−structure gap ≈ {gap:.2f}\nstructure margin vs FP-PCA ≈ {margin:.2f}",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#e5e7eb"),
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out)


def _knn_idx(sim: np.ndarray, i: int, k: int) -> np.ndarray:
    s = sim[i].copy()
    s[i] = -np.inf
    return np.argpartition(-s, k)[:k]


def plot_neighborhood_cartoon(emb_path: Path, csv_path: Path, out: Path, k: int = 8) -> None:
    z = np.load(emb_path, allow_pickle=True)
    S = z["S"].astype(np.float64)
    fp = z["fp"].astype(np.float64)
    smiles = [str(s) for s in z["smiles"]]
    # cosine S
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    s_sim = Sn @ Sn.T
    # tanimoto-ish on binary fp (assume 0/1)
    fp_bin = (fp > 0).astype(np.float64)
    inter = fp_bin @ fp_bin.T
    card = fp_bin.sum(axis=1, keepdims=True)
    union = card + card.T - inter
    t_sim = inter / (union + 1e-8)

    import pandas as pd

    df = pd.read_csv(csv_path)
    # pick probe: high Δv_beh, decent H_S
    df = df.assign(delta_v=df.v_beh_ECFP - df.v_beh_S)
    row = df.sort_values(["delta_v", "H_murcko_S"], ascending=[False, False]).iloc[0]
    probe = str(row.probe_smiles)
    i = next((j for j, s in enumerate(smiles) if s == probe), 0)

    s_nn = _knn_idx(s_sim, i, k)
    e_nn = _knn_idx(t_sim, i, k)

    def unique_scaffolds(idxs: np.ndarray, n: int = 5) -> list[str]:
        seen: set[str] = set()
        out_s: list[str] = []
        for j in idxs:
            smi = smiles[int(j)]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smi
            except Exception:
                sc = smi
            if sc in seen:
                continue
            seen.add(sc)
            out_s.append(smi)
            if len(out_s) >= n:
                break
        return out_s

    s_mols = unique_scaffolds(s_nn, 5)
    e_mols = unique_scaffolds(e_nn, 5)

    fig = plt.figure(figsize=(11.5, 6.2))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.18, 1.0, 1.0], hspace=0.28)
    ax_t = fig.add_subplot(outer[0])
    ax_t.axis("off")
    ax_t.set_title("Neighborhood cartoon (real probe)", fontsize=12, pad=2)
    ax_t.text(
        0.5,
        0.35,
        f"$v_{{\\mathrm{{beh}}}}$ $S$={row.v_beh_S:.2f} vs ECFP={row.v_beh_ECFP:.2f}"
        f"   ·   Murcko $H$ $S$={row.H_murcko_S:.2f} vs ECFP={row.H_murcko_ECFP:.2f}",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )

    for r_i, (title, mols, color) in enumerate(
        [
            (r"$S$ neighbors (diverse scaffolds)", s_mols, "#0f766e"),
            ("ECFP neighbors (structure-similar)", e_mols, "#6b7280"),
        ]
    ):
        row_gs = outer[r_i + 1].subgridspec(1, 6, wspace=0.15)
        ax_l = fig.add_subplot(row_gs[0])
        ax_l.axis("off")
        ax_l.text(0.5, 0.5, title, ha="center", va="center", fontsize=9.5, color=color, wrap=True)
        for c, smi in enumerate(mols[:5]):
            ax = fig.add_subplot(row_gs[c + 1])
            ax.axis("off")
            img = _mol_img(smi, size=(240, 200))
            if img is not None:
                ax.imshow(img)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1.2)

    fig.text(
        0.01,
        0.01,
        "Murcko scaffolds of nearest neighbors; illustrative single probe",
        fontsize=7.5,
        color="#9ca3af",
    )
    _save(fig, out)


def plot_g5(
    gen_ckpt: Path,
    bank_path: Path,
    g5_json: Path,
    out: Path,
    *,
    n_draw: int = 10,
    n_gen: int = 48,
) -> None:
    import torch
    from functionalspec.gen.fan import FunctionalAbstractionNetwork
    from functionalspec.gen.generate import generate_from_spec, load_generator
    from functionalspec.gen.objective import DesignObjective
    from functionalspec.gen.spec_bank import SpecBank

    g5 = json.loads(g5_json.read_text())["G5"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = SpecBank(bank_path)
    fan = FunctionalAbstractionNetwork(bank)
    model, tok, ckpt = load_generator(gen_ckpt, device)
    max_len = int(ckpt.get("max_len", 150))
    spec = fan.plan(DesignObjective.from_string("LogP=2.5"))
    res = generate_from_spec(
        spec, model, tok, n=n_gen, temperature=1.0, device=device, max_len=max_len, filter=False
    )
    smiles = list(dict.fromkeys(res.smiles))  # unique, preserve order
    # pick diverse scaffolds
    by_scaf: dict[str, str] = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or "<acyclic>"
        except Exception:
            continue
        if sc not in by_scaf:
            by_scaf[sc] = s
        if len(by_scaf) >= n_draw:
            break
    pick = list(by_scaf.values())[:n_draw]
    logps = []
    for s in pick:
        d = compute_surrogates(s)
        logps.append(float(d["LogP"]) if d and np.isfinite(d.get("LogP", np.nan)) else float("nan"))

    means = g5["behavior_means"]
    fig = plt.figure(figsize=(12.0, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.45], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.plot(range(1, len(means) + 1), means, "o-", color="#0f766e", lw=2, markersize=8, markerfacecolor="white", markeredgewidth=1.8)
    ax.axhline(2.5, color="#e5e7eb", lw=1.2, zorder=0)
    ax.set_xticks(range(1, len(means) + 1))
    ax.set_xlabel("Independent batch")
    ax.set_ylabel("Mean LogP")
    ax.set_title(
        f"G5: one Spec (LogP=2.5) → stable behavior, diverse chemistry   "
        f"(batch-mean std={g5['behavior_mean_std_across_batches']:.3f}; "
        f"union scaffolds={g5['union_n_scaffolds']})",
        fontsize=11,
    )
    ax.set_ylim(min(means) - 0.4, max(means) + 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n = len(pick)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    mol_gs = gs[1].subgridspec(rows, cols, wspace=0.08, hspace=0.25)
    for i, (smi, lp) in enumerate(zip(pick, logps)):
        r, c = divmod(i, cols)
        axm = fig.add_subplot(mol_gs[r, c])
        axm.axis("off")
        img = _mol_img(smi, size=(260, 200))
        if img is not None:
            axm.imshow(img)
        lab = f"LogP={lp:.1f}" if np.isfinite(lp) else ""
        axm.set_title(lab, fontsize=8, color="#374151")
    fig.text(0.5, 0.01, "Murcko scaffolds from a single Spec sample batch (illustrative)", ha="center", fontsize=8, color="#9ca3af")
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("runs/gen/paper_figs"))
    p.add_argument("--e2", type=Path, default=Path("runs/p1/e2/e2_summary.json"))
    p.add_argument("--embeddings", type=Path, default=Path("runs/p1/spec_specificity/embeddings.npz"))
    p.add_argument("--soft-csv", type=Path, default=Path("runs/p1/spec_specificity/soft_neighborhoods_k32.csv"))
    p.add_argument("--gen-eval", type=Path, default=Path("runs/gen/eval/gen_eval_summary.json"))
    p.add_argument("--generator", type=Path, default=Path("runs/p2_selfies_cond/p2_best.pt"))
    p.add_argument("--bank", type=Path, default=Path("runs/gen/spec_bank.npz"))
    p.add_argument("--skip-g5", action="store_true")
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_pipeline_placeholder(out / "fig_pipeline_placeholder")
    plot_e2(args.e2, out / "fig_e2")
    plot_neighborhood_cartoon(args.embeddings, args.soft_csv, out / "fig_neighborhood_cartoon")
    if not args.skip_g5:
        plot_g5(args.generator, args.bank, args.gen_eval, out / "fig_g5_diversity")


if __name__ == "__main__":
    main()
