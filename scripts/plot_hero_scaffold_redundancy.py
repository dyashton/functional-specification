#!/usr/bin/env python3
"""Hero: scaffold-redundant function — many Murcko cores → one Spec S."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


TEAL = "#0f766e"
TEAL_DARK = "#115e59"
STONE = "#78716c"
INK = "#1c1917"
MUTED = "#57534e"
BG = "#fafaf9"
CARD = "#ffffff"


def _murcko(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) or None
    except Exception:
        return None


def _mol_img(smiles: str, size=(260, 200)):
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


def _knn(sim: np.ndarray, i: int, k: int) -> np.ndarray:
    s = sim[i].copy()
    s[i] = -np.inf
    return np.argpartition(-s, k)[:k]


def _unique(smiles: list[str], idxs: np.ndarray, n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for j in idxs:
        smi = smiles[int(j)]
        sc = _murcko(smi)
        if not sc or sc in seen:
            continue
        seen.add(sc)
        out.append(smi)
        if len(out) >= n:
            break
    return out


def _n_unique(smiles: list[str], idxs: np.ndarray) -> int:
    return len({sc for j in idxs if (sc := _murcko(smiles[int(j)]))})


def _pick_probe(df: pd.DataFrame, smiles: list[str], s_sim, t_sim, k: int) -> tuple[pd.Series, int]:
    """Prefer high behavior gap + more unique S scaffolds + lower mean ECFP diversity."""
    best = None
    best_score = -1e9
    for _, r in df.iterrows():
        probe = str(r.probe_smiles)
        try:
            i = smiles.index(probe)
        except ValueError:
            continue
        n_s = _n_unique(smiles, _knn(s_sim, i, k))
        n_e = _n_unique(smiles, _knn(t_sim, i, k))
        dv = float(r.v_beh_ECFP - r.v_beh_S)
        score = dv * 2.5 + (n_s - n_e) / 8.0 + n_s / 40.0 + float(r.H_murcko_S - r.H_murcko_ECFP)
        if score > best_score:
            best_score = score
            best = (r, i, n_s, n_e)
    assert best is not None
    r, i, n_s, n_e = best
    r = r.copy()
    r["n_scaf_S"] = n_s
    r["n_scaf_ECFP"] = n_e
    return r, i


def _card(ax, x, y, w, h, *, fc, ec, lw=1.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            transform=ax.transData,
            clip_on=False,
        )
    )


def plot_hero(
    emb_path: Path,
    soft_csv: Path,
    out: Path,
    *,
    gen_eval: Path | None,
    k: int,
    n_s: int,
    n_e: int,
) -> None:
    z = np.load(emb_path, allow_pickle=True)
    S = z["S"].astype(np.float64)
    fp = z["fp"].astype(np.float64)
    smiles = [str(s) for s in z["smiles"]]

    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    s_sim = Sn @ Sn.T
    fp_bin = (fp > 0).astype(np.float64)
    inter = fp_bin @ fp_bin.T
    card = fp_bin.sum(axis=1, keepdims=True)
    t_sim = inter / (card + card.T - inter + 1e-8)

    df = pd.read_csv(soft_csv)
    row, i = _pick_probe(df, smiles, s_sim, t_sim, k)
    s_nn = _knn(s_sim, i, k)
    e_nn = _knn(t_sim, i, k)
    s_mols = _unique(smiles, s_nn, n_s)
    e_mols = _unique(smiles, e_nn, n_e)
    n_scaf_s = int(row["n_scaf_S"])
    n_scaf_e = int(row["n_scaf_ECFP"])

    g5_scaffolds = None
    g5_std = None
    if gen_eval and gen_eval.exists():
        g5 = json.loads(gen_eval.read_text())["G5"]
        g5_scaffolds = g5.get("union_n_scaffolds")
        g5_std = g5.get("behavior_mean_std_across_batches")

    fig = plt.figure(figsize=(15.0, 9.2), facecolor=BG)
    # rows: title | fan | contrast | footer metrics
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.7, 3.6, 2.4],
        hspace=0.18,
        left=0.03,
        right=0.97,
        top=0.96,
        bottom=0.05,
    )

    # ── Title ──────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0])
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    ax_t.text(0.5, 0.70, "Scaffold-redundant function", ha="center", va="center", fontsize=24, fontweight="bold", color=INK)
    ax_t.text(
        0.5,
        0.22,
        "One Functional Specification $S$  ·  many Murcko scaffolds  ·  shared surrogate behavior",
        ha="center",
        va="center",
        fontsize=12,
        color=MUTED,
    )

    # ── Fan: many scaffolds → one Spec ─────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Left rail: what "redundancy" means
    _card(ax, 0.15, 5.9, 3.5, 2.0, fc="#ecfdf5", ec=TEAL, lw=1.6)
    ax.text(1.9, 7.45, "THE CLAIM", ha="center", fontsize=9, fontweight="bold", color=TEAL_DARK)
    ax.text(
        1.9,
        6.55,
        "Behavior is scaffold-\nredundant under $S$:\nchemically different cores\ncan realize the same Spec.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=INK,
    )

    # Headline numbers
    _card(ax, 0.15, 3.55, 3.5, 2.1, fc=CARD, ec="#d6d3d1", lw=1.2)
    ax.text(1.9, 5.25, "THIS NEIGHBORHOOD", ha="center", fontsize=8.5, fontweight="bold", color=STONE)
    ax.text(1.9, 4.55, f"{n_scaf_s}", ha="center", fontsize=28, fontweight="bold", color=TEAL)
    ax.text(1.9, 3.95, f"unique Murcko scaffolds\nin $S$ $k$={k} neighbors", ha="center", fontsize=8.5, color=MUTED)

    _card(ax, 0.15, 1.15, 3.5, 2.1, fc=CARD, ec="#d6d3d1", lw=1.2)
    if g5_scaffolds is not None:
        ax.text(1.9, 2.85, "GENERATION (G5)", ha="center", fontsize=8.5, fontweight="bold", color=STONE)
        ax.text(1.9, 2.15, f"{g5_scaffolds}", ha="center", fontsize=28, fontweight="bold", color=TEAL)
        ax.text(
            1.9,
            1.55,
            f"scaffolds from one Spec\n(batch LogP std={g5_std:.2f})",
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )
    else:
        ax.text(1.9, 2.2, f"$v_{{\\mathrm{{beh}}}}$ $S$={row.v_beh_S:.2f}\nvs ECFP={row.v_beh_ECFP:.2f}", ha="center", fontsize=12, color=INK)

    # Hub Spec (center-right of fan)
    hub_x, hub_y = 11.6, 4.1
    ax.add_patch(Circle((hub_x, hub_y), 1.15, facecolor=TEAL, edgecolor=TEAL_DARK, lw=2, zorder=5))
    ax.text(hub_x, hub_y + 0.28, "ONE", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=6)
    ax.text(hub_x, hub_y - 0.15, "SPEC  $S$", ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=6)
    ax.text(hub_x, hub_y - 0.55, "shared behavior", ha="center", va="center", fontsize=8, color="#ccfbf1", zorder=6)

    # Place S scaffolds in an arc around the hub (left side) — fan INTO the Spec
    # Positions: grid of molecule cards on the left-center, arrows to hub
    cols = 4
    rows = int(np.ceil(len(s_mols) / cols))
    x0, y0 = 4.0, 7.35
    dx, dy = 1.55, 2.05
    for idx, smi in enumerate(s_mols):
        r, c = divmod(idx, cols)
        # fill column-major-ish by rows
        r, c = divmod(idx, cols)
        cx = x0 + c * dx
        cy = y0 - r * dy
        # molecule card
        _card(ax, cx - 0.68, cy - 0.85, 1.36, 1.55, fc=CARD, ec=TEAL, lw=1.8)
        img = _mol_img(smi, size=(240, 180))
        if img is not None:
            # imshow in data coords via inset-like extent
            ax.imshow(img, extent=(cx - 0.60, cx + 0.60, cy - 0.72, cy + 0.55), aspect="auto", zorder=4, origin="upper")
        ax.text(cx - 0.55, cy + 0.58, f"{idx+1}", fontsize=7.5, fontweight="bold", color=TEAL, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=TEAL, lw=1.0))
        # arrow from card to hub
        ax.annotate(
            "",
            xy=(hub_x - 1.05, hub_y),
            xytext=(cx + 0.70, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#5eead4",
                lw=1.3,
                alpha=0.85,
                mutation_scale=10,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=3,
        )

    ax.text(
        7.0,
        0.35,
        f"{len(s_mols)} distinct Murcko cores  →  converge on the same Spec  (scaffold redundancy)",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEAL_DARK,
    )

    # ── Bottom contrast strip ───────────────────────────────────────────────
    gs2 = gs[2].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.08)
    ax_c = fig.add_subplot(gs2[0])
    ax_c.set_xlim(0, 12)
    ax_c.set_ylim(0, 4.2)
    ax_c.axis("off")

    _card(ax_c, 0.1, 0.15, 11.7, 3.9, fc=CARD, ec="#e7e5e4", lw=1.2)
    ax_c.text(0.45, 3.55, "Contrast: fingerprint neighbors for the same probe", ha="left", fontsize=11, fontweight="bold", color=INK)
    ax_c.text(
        0.45,
        3.05,
        f"ECFP $k$={k}: {n_scaf_e} unique scaffolds  ·  higher behavior variance "
        f"($v_{{\\mathrm{{beh}}}}$={row.v_beh_ECFP:.2f} vs $S$={row.v_beh_S:.2f})",
        ha="left",
        fontsize=9,
        color=MUTED,
    )

    for idx, smi in enumerate(e_mols[:6]):
        cx = 1.15 + idx * 1.85
        cy = 1.45
        _card(ax_c, cx - 0.75, cy - 0.95, 1.5, 1.7, fc="#fafaf9", ec=STONE, lw=1.4)
        img = _mol_img(smi, size=(220, 170))
        if img is not None:
            ax_c.imshow(img, extent=(cx - 0.65, cx + 0.65, cy - 0.80, cy + 0.55), aspect="auto", zorder=4, origin="upper")

    ax_c.text(6.0, 0.35, "Structure-similar search does not target scaffold-redundant function", ha="center", fontsize=8.5, color=STONE, style="italic")

    # Metric bars panel
    ax_m = fig.add_subplot(gs2[1])
    ax_m.set_facecolor(CARD)
    metrics = [
        ("Unique scaffolds\n(higher = more redundant)", n_scaf_s, n_scaf_e),
        ("Behavior variance\n(lower = tighter function)", float(row.v_beh_S), float(row.v_beh_ECFP)),
        ("Scaffold entropy $H$\n(higher = more diverse)", float(row.H_murcko_S), float(row.H_murcko_ECFP)),
    ]
    # three small horizontal comparisons
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 3.4)
    ax_m.axis("off")
    ax_m.text(0.5, 3.15, "Same probe, two geometries", ha="center", fontsize=10.5, fontweight="bold", color=INK)

    for mi, (label, vs, ve) in enumerate(metrics):
        y = 2.45 - mi * 1.0
        ax_m.text(0.05, y + 0.55, label, ha="left", va="center", fontsize=8, color=MUTED)
        # normalize bar widths within row
        m = max(vs, ve, 1e-6)
        ax_m.barh([y + 0.18], [0.55 * vs / m], height=0.22, left=0.05, color=TEAL, label="$S$" if mi == 0 else None)
        ax_m.barh([y - 0.08], [0.55 * ve / m], height=0.22, left=0.05, color=STONE, label="ECFP" if mi == 0 else None)
        ax_m.text(0.05 + 0.55 * vs / m + 0.02, y + 0.18, f"{vs:.2f}" if isinstance(vs, float) and vs < 10 else f"{vs:.0f}",
                  va="center", fontsize=8, color=TEAL_DARK, fontweight="bold")
        ax_m.text(0.05 + 0.55 * ve / m + 0.02, y - 0.08, f"{ve:.2f}" if isinstance(ve, float) and ve < 10 else f"{ve:.0f}",
                  va="center", fontsize=8, color=STONE, fontweight="bold")
    ax_m.legend(loc="lower right", frameon=False, fontsize=8)

    fig.text(
        0.5,
        0.012,
        "Real Corpus A probe  ·  Murcko scaffolds of nearest neighbors  ·  "
        "G5 count from Spec-conditioned generation (LogP=2.5)",
        ha="center",
        fontsize=8,
        color="#a8a29e",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}.{{png,pdf}}")
    print(f"S scaffolds={n_scaf_s} ECFP={n_scaf_e}  vS={row.v_beh_S:.3f} vE={row.v_beh_ECFP:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path, default=Path("runs/p1/spec_specificity/embeddings.npz"))
    p.add_argument("--soft-csv", type=Path, default=Path("runs/p1/spec_specificity/soft_neighborhoods_k64.csv"))
    p.add_argument("--gen-eval", type=Path, default=Path("runs/gen/eval/gen_eval_summary.json"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/fig_hero"))
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-s", type=int, default=8)
    p.add_argument("--n-e", type=int, default=6)
    args = p.parse_args()
    plot_hero(
        args.embeddings,
        args.soft_csv,
        args.out,
        gen_eval=args.gen_eval,
        k=args.k,
        n_s=args.n_s,
        n_e=args.n_e,
    )


if __name__ == "__main__":
    main()
