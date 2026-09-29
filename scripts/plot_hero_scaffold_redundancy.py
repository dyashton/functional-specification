#!/usr/bin/env python3
"""Hero: scaffold-redundant function — PIL-composed so molecules stay unstretched."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


TEAL = (15, 118, 110)
TEAL_DARK = (17, 94, 89)
STONE = (120, 113, 108)
INK = (28, 25, 23)
MUTED = (87, 83, 78)
BG = (250, 250, 249)
CARD = (255, 255, 255)
TEAL_SOFT = (236, 253, 245)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
        if bold
        else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _murcko(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) or None
    except Exception:
        return None


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


def _mol_tile(smiles: str, *, edge: tuple[int, int, int], idx: int, tile: int = 360, pad: int = 18) -> Image.Image:
    m = Chem.MolFromSmiles(smiles)
    inner = tile - 2 * pad
    if m is None:
        mol_im = Image.new("RGB", (inner, inner), CARD)
    else:
        mol_im = Draw.MolToImage(m, size=(inner, inner))
    canvas = Image.new("RGB", (tile, tile), CARD)
    canvas.paste(mol_im, (pad, pad))
    draw = ImageDraw.Draw(canvas)
    for t in range(3):
        draw.rectangle([t, t, tile - 1 - t, tile - 1 - t], outline=edge)
    r = 15
    cx, cy = 24, 24
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CARD, outline=edge, width=2)
    font = _font(15, bold=True)
    label = str(idx)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2 - 1), label, fill=edge, font=font)
    return canvas


def _grid(mols: list[str], *, edge: tuple[int, int, int], cols: int, tile: int, gap: int = 14) -> Image.Image:
    rows = int(np.ceil(len(mols) / cols))
    w = cols * tile + (cols - 1) * gap
    h = rows * tile + (rows - 1) * gap
    grid = Image.new("RGB", (w, h), BG)
    for i, smi in enumerate(mols):
        r, c = divmod(i, cols)
        grid.paste(_mol_tile(smi, edge=edge, idx=i + 1, tile=tile), (c * (tile + gap), r * (tile + gap)))
    return grid


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, *, fill, outline, width=2, radius=16):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _card(draw, xy, title: str, body: str, *, fill, outline, title_fill, body_fill=INK):
    _rounded_rect(draw, xy, fill=fill, outline=outline, width=2, radius=14)
    x0, y0, x1, y1 = xy
    cx = (x0 + x1) / 2
    draw.text((cx, y0 + 18), title, fill=title_fill, font=_font(13, bold=True), anchor="mt")
    # body lines
    font = _font(12)
    lines = body.split("\n")
    y = y0 + 48
    for line in lines:
        draw.text((cx, y), line, fill=body_fill, font=font, anchor="mt")
        y += 16


def _metrics_png(n_scaf_s, n_scaf_e, v_s, v_e, h_s, h_e, size=(420, 360)) -> Image.Image:
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(0.5, 3.25, "Same probe,\ntwo geometries", ha="center", fontsize=11, fontweight="bold", color="#1c1917")
    metrics = [
        ("Unique scaffolds\n(higher = more redundant)", n_scaf_s, n_scaf_e),
        ("Behavior variance\n(lower = tighter)", v_s, v_e),
        ("Scaffold entropy H\n(higher = more diverse)", h_s, h_e),
    ]
    for mi, (label, vs, ve) in enumerate(metrics):
        y = 2.55 - mi * 0.95
        ax.text(0.06, y + 0.48, label, ha="left", va="center", fontsize=8, color="#57534e")
        mmax = max(vs, ve, 1e-6)
        ax.barh([y + 0.15], [0.58 * vs / mmax], height=0.18, left=0.06, color="#0f766e", label="S" if mi == 0 else None)
        ax.barh([y - 0.08], [0.58 * ve / mmax], height=0.18, left=0.06, color="#78716c", label="ECFP" if mi == 0 else None)
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) and v < 10 else f"{int(v)}"
        ax.text(0.06 + 0.58 * vs / mmax + 0.02, y + 0.15, fmt(vs), va="center", fontsize=8, color="#115e59", fontweight="bold")
        ax.text(0.06 + 0.58 * ve / mmax + 0.02, y - 0.08, fmt(ve), va="center", fontsize=8, color="#78716c", fontweight="bold")
    ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
    cardn = fp_bin.sum(axis=1, keepdims=True)
    t_sim = inter / (cardn + cardn.T - inter + 1e-8)

    df = pd.read_csv(soft_csv)
    row, i = _pick_probe(df, smiles, s_sim, t_sim, k)
    s_mols = _unique(smiles, _knn(s_sim, i, k), n_s)
    e_mols = _unique(smiles, _knn(t_sim, i, k), n_e)
    n_scaf_s = int(row["n_scaf_S"])
    n_scaf_e = int(row["n_scaf_ECFP"])

    g5_scaffolds = g5_std = None
    if gen_eval and gen_eval.exists():
        g5 = json.loads(gen_eval.read_text())["G5"]
        g5_scaffolds = g5.get("union_n_scaffolds")
        g5_std = g5.get("behavior_mean_std_across_batches")

    s_grid = _grid(s_mols[:8], edge=TEAL, cols=4, tile=360, gap=16)
    e_grid = _grid(e_mols[:4], edge=STONE, cols=4, tile=300, gap=16)
    metrics = _metrics_png(
        n_scaf_s,
        n_scaf_e,
        float(row.v_beh_S),
        float(row.v_beh_ECFP),
        float(row.H_murcko_S),
        float(row.H_murcko_ECFP),
    )

    margin = 40
    gap = 20
    content_w = s_grid.size[0]
    # bottom row: e_grid + metrics
    metrics = metrics.resize(
        (max(280, content_w - e_grid.size[0] - gap), e_grid.size[1]),
        Image.Resampling.BILINEAR,
    )
    # keep metrics height match e_grid without stretching width wrongly — fit height
    mh = e_grid.size[1]
    mw = int(metrics.size[0] * (mh / metrics.size[1]))
    metrics = metrics.resize((mw, mh), Image.Resampling.BILINEAR)
    bottom_w = e_grid.size[0] + gap + metrics.size[0]
    content_w = max(content_w, bottom_w)

    # Header heights
    title_h = 70
    card_h = 120
    label_h = 36
    footer_h = 36
    total_h = (
        margin
        + title_h
        + gap
        + card_h
        + gap
        + label_h
        + s_grid.size[1]
        + gap
        + label_h
        + 28
        + e_grid.size[1]
        + margin
        + footer_h
    )
    total_w = content_w + 2 * margin

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    y = margin
    draw.text((total_w / 2, y + 8), "Scaffold-redundant function", fill=INK, font=_font(28, bold=True), anchor="mt")
    draw.text(
        (total_w / 2, y + 42),
        "One Functional Specification S  ·  many distinct full molecules  ·  shared surrogate behavior",
        fill=MUTED,
        font=_font(14),
        anchor="mt",
    )
    y += title_h + gap

    # 4 cards
    card_w = (content_w - 3 * 12) // 4
    x = margin
    cards = [
        ("THE CLAIM", "Behavior is scaffold-\nredundant under S:\nchemically different\nmolecules realize\nthe same Spec.", TEAL_SOFT, TEAL, TEAL_DARK),
        ("THIS NEIGHBORHOOD", f"{n_scaf_s}\nunique Murcko in\nS k={k} neighbors", CARD, (214, 211, 209), STONE),
        (
            "GENERATION (G5)",
            (
                f"{g5_scaffolds}\nscaffolds from one Spec\n(batch LogP std={g5_std:.2f})"
                if g5_scaffolds is not None
                else f"v_beh S={row.v_beh_S:.2f}\nvs ECFP={row.v_beh_ECFP:.2f}"
            ),
            CARD,
            (214, 211, 209),
            STONE,
        ),
    ]
    for title, body, fill, outline, tc in cards:
        _card(draw, (x, y, x + card_w, y + card_h), title, body, fill=fill, outline=outline, title_fill=tc)
        x += card_w + 12
    # Spec hub card
    _rounded_rect(draw, (x, y, x + card_w, y + card_h), fill=TEAL, outline=TEAL_DARK, width=2, radius=14)
    draw.text((x + card_w / 2, y + 48), "ONE SPEC  S", fill=CARD, font=_font(15, bold=True), anchor="mt")
    draw.text((x + card_w / 2, y + 78), "shared behavior", fill=(204, 251, 241), font=_font(12), anchor="mt")
    y += card_h + gap

    draw.text(
        (margin, y + 8),
        "S neighbors — full molecules (unique Murcko), all map to the same Spec",
        fill=TEAL_DARK,
        font=_font(15, bold=True),
        anchor="lt",
    )
    y += label_h
    canvas.paste(s_grid, (margin, y))
    y += s_grid.size[1] + gap

    draw.text(
        (margin, y + 4),
        "Contrast: ECFP neighbors for the same probe (full molecules)",
        fill=INK,
        font=_font(14, bold=True),
        anchor="lt",
    )
    draw.text(
        (margin, y + 24),
        f"k={k}: {n_scaf_e} unique scaffolds  ·  higher v_beh ({row.v_beh_ECFP:.2f} vs S={row.v_beh_S:.2f})  ·  structure search ≠ scaffold-redundant function",
        fill=MUTED,
        font=_font(11),
        anchor="lt",
    )
    y += label_h + 28
    canvas.paste(e_grid, (margin, y))
    canvas.paste(metrics, (margin + e_grid.size[0] + gap, y))
    y += e_grid.size[1] + 16

    draw.text(
        (total_w / 2, y + 4),
        "Full molecules (unique by Murcko)  ·  real Corpus A probe  ·  G5 from Spec-conditioned generation (LogP=2.5)",
        fill=(168, 162, 158),
        font=_font(11),
        anchor="mt",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out.with_suffix(".png"))
    # PDF via matplotlib wrapper of the PNG (lossless-ish embed)
    fig = plt.figure(figsize=(total_w / 100, total_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(canvas), aspect="equal", interpolation="nearest")
    ax.axis("off")
    fig.savefig(out.with_suffix(".pdf"), dpi=100)
    plt.close(fig)

    print(f"wrote {out}.{{png,pdf}}  size={canvas.size}")
    print(f"S scaffolds={n_scaf_s} ECFP={n_scaf_e}  vS={row.v_beh_S:.3f} vE={row.v_beh_ECFP:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path, default=Path("runs/p1/spec_specificity/embeddings.npz"))
    p.add_argument("--soft-csv", type=Path, default=Path("runs/p1/spec_specificity/soft_neighborhoods_k64.csv"))
    p.add_argument("--gen-eval", type=Path, default=Path("runs/gen/eval/gen_eval_summary.json"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/fig_hero"))
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-s", type=int, default=8)
    p.add_argument("--n-e", type=int, default=4)
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
