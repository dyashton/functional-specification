#!/usr/bin/env python3
"""Plot MVP ablation suite results into runs/ablations/figures/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL = "#0f766e"
TEAL_LIGHT = "#5eead4"
GRAY = "#6b7280"
GRAY_LIGHT = "#d1d5db"
FAIL = "#b45309"
PASS = "#0f766e"
INK = "#111827"
MUTED = "#4b5563"


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.{{png,pdf}}")


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _load(root: Path) -> dict:
    return json.loads((root / "summary.json").read_text())


def _e2_json(root: Path, cell: str) -> dict | None:
    p = root / cell / "e2" / "e2_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def plot_codebook_sweep(summary: dict, root: Path, out: Path) -> None:
    top2 = summary.get("top2") or {}
    cands = top2.get("candidates") or []
    if not cands:
        return
    Cs = [c["C"] for c in cands]
    gaps = [c["gap"] for c in cands]
    dRs = [c["delta_R"] for c in cands]
    win_list = list(top2.get("top2") or [])
    winners = set(win_list)
    rank = {c: i + 1 for i, c in enumerate(win_list)}
    defaults = (summary.get("defaults") or {}).get("defaults") or {}
    def_c = int(defaults.get("codebook_size") or (win_list[0] if win_list else 512))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = [TEAL if c in winners else GRAY for c in Cs]

    ax = axes[0]
    ax.plot(Cs, gaps, color=GRAY_LIGHT, lw=1.5, zorder=1)
    ax.scatter(Cs, gaps, c=colors, s=70, zorder=2, edgecolors="white", linewidths=0.8)
    if def_c in Cs:
        ax.scatter([def_c], [gaps[Cs.index(def_c)]], s=160, c=TEAL, marker="*", zorder=3, edgecolors="white")
    ax.axhline(0.25, color=FAIL, ls="--", lw=1, alpha=0.8, label="E2 gap threshold")
    ax.set_xlabel("|C| codebook size")
    ax.set_ylabel("E2 behavior−structure gap")
    ax.set_title("Codebook size → E2 gap")
    ax.set_xticks(Cs)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    ax.plot(Cs, dRs, color=GRAY_LIGHT, lw=1.5, zorder=1)
    ax.scatter(Cs, dRs, c=colors, s=70, zorder=2, edgecolors="white", linewidths=0.8)
    if def_c in Cs:
        ax.scatter([def_c], [dRs[Cs.index(def_c)]], s=160, c=TEAL, marker="*", zorder=3, edgecolors="white",
                   label="full-model |C|")
    for c, y in zip(Cs, dRs):
        if c in rank:
            ax.annotate(
                f"top-{rank[c]}" + (" ★ default" if c == def_c else ""),
                (c, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7.5,
                color=TEAL,
            )
    ax.set_xlabel("|C| codebook size")
    ax.set_ylabel("Soft-neighborhood ΔR (k=64)")
    ax.set_title("Codebook size → specificity")
    ax.set_xticks(Cs)
    _style(ax)

    fig.suptitle(
        f"HPS: codebook size |C| (★ = full-model default C={def_c}; teal = top-2)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out)


def plot_necessity(summary: dict, root: Path, out: Path) -> None:
    # baseline + necessity + surrogates
    order = [
        ("codebook_C512", "VQ + contrast\n(C=512)"),
        ("necessity_novq", "no VQ"),
        ("necessity_nocontrast", "no contrastive"),
        ("necessity_wecfp0", "w_ecfp = 0"),
        ("surrogates_physchem", "physchem-only\nsurrogates"),
    ]
    labels, gaps, passes, structure = [], [], [], []
    for key, lab in order:
        v = summary["variants"].get(key) or {}
        e2 = _e2_json(root, key)
        labels.append(lab)
        gaps.append(v.get("e2_gap") if v.get("e2_gap") is not None else (e2 or {}).get("gap_spearman"))
        passes.append(bool(v.get("e2_pass")) if "e2_pass" in v else bool((e2 or {}).get("pass")))
        structure.append((e2 or {}).get("structure_probe", {}).get("mean_spearman"))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    x = np.arange(len(labels))
    colors = [PASS if p else FAIL for p in passes]
    bars = ax.bar(x, gaps, color=colors, width=0.7, edgecolor="white", linewidth=0.8)
    ax.axhline(0.25, color=INK, ls="--", lw=1, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("E2 gap (Spearman)")
    ax.set_title("Design necessity (vs full model)")
    ax.set_ylim(0, max(g for g in gaps if g is not None) * 1.25)
    for b, g, p in zip(bars, gaps, passes):
        if g is None:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            g + 0.008,
            f"{g:.2f}\n{'pass' if p else 'FAIL'}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )
    _style(ax)

    ax = axes[1]
    # paired surrogate vs structure for same cells
    sur_vals, str_vals = [], []
    for key, _ in order:
        e2 = _e2_json(root, key)
        if not e2:
            sur_vals.append(np.nan)
            str_vals.append(np.nan)
            continue
        sur_vals.append(e2["surrogate_probe"]["mean_spearman"])
        str_vals.append(e2["structure_probe"]["mean_spearman"])
    w = 0.36
    ax.bar(x - w / 2, sur_vals, width=w, color=TEAL, label="Surrogate (S)", edgecolor="white")
    ax.bar(x + w / 2, str_vals, width=w, color=GRAY, label="Structure (S)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Spearman")
    ax.set_title("Probe split (same cells)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)

    fig.tight_layout()
    _save(fig, out)


def plot_hps(summary: dict, root: Path, out: Path) -> None:
    """β and K sweeps with full-model defaults highlighted."""
    defaults = (summary.get("defaults") or {}).get("defaults") or {}
    def_beta = float(defaults.get("commitment_cost_beta", 0.25))
    def_k = int(defaults.get("num_slots", 12))
    full_cell = (summary.get("defaults") or {}).get("full_model_cell", "codebook_C512")

    # Discover beta_* / slots_K* from variants
    beta_rows: list[tuple[float, str, dict]] = []
    slot_rows: list[tuple[int, str, dict]] = []
    for name, v in summary["variants"].items():
        if "e2_gap" not in v:
            continue
        if name.startswith("beta_"):
            try:
                b = float(name.split("beta_", 1)[1])
            except ValueError:
                continue
            beta_rows.append((b, name, v))
        elif name.startswith("slots_K"):
            try:
                k = int(name.split("slots_K", 1)[1])
            except ValueError:
                continue
            slot_rows.append((k, name, v))
    # Ensure full-model default appears even if only codebook_C* exists
    if full_cell in summary["variants"] and "e2_gap" in summary["variants"][full_cell]:
        fv = summary["variants"][full_cell]
        if not any(abs(b - def_beta) < 1e-9 for b, _, _ in beta_rows):
            beta_rows.append((def_beta, full_cell, fv))
        if not any(k == def_k for k, _, _ in slot_rows):
            slot_rows.append((def_k, full_cell, fv))

    beta_rows.sort(key=lambda r: r[0])
    slot_rows.sort(key=lambda r: r[0])
    if not beta_rows and not slot_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    if beta_rows:
        xs = [b for b, _, _ in beta_rows]
        gaps = [v["e2_gap"] for _, _, v in beta_rows]
        dRs = [v["delta_R_k64"] for _, _, v in beta_rows]
        ax.plot(xs, gaps, "-", color=TEAL_LIGHT, lw=1.4, zorder=1)
        for x, g, name in zip(xs, gaps, [n for _, n, _ in beta_rows]):
            is_def = abs(x - def_beta) < 1e-9
            ax.scatter(
                [x],
                [g],
                s=140 if is_def else 55,
                c=TEAL if is_def else GRAY,
                marker="*" if is_def else "o",
                zorder=3,
                edgecolors="white",
                linewidths=0.6,
            )
            if is_def:
                ax.annotate(
                    f"default β={def_beta:g}\n(full model)",
                    (x, g),
                    textcoords="offset points",
                    xytext=(8, 10),
                    fontsize=7.5,
                    color=TEAL,
                )
        ax.axvline(def_beta, color=TEAL, ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("commitment β")
        ax.set_ylabel("E2 gap", color=TEAL)
        ax.tick_params(axis="y", labelcolor=TEAL)
        ax2 = ax.twinx()
        ax2.plot(xs, dRs, "s--", color=GRAY, lw=1.2, markersize=5, alpha=0.85)
        ax2.set_ylabel("ΔR (k=64)", color=GRAY)
        ax2.tick_params(axis="y", labelcolor=GRAY)
        ax.set_xticks(xs)
        ax.set_title("HPS: commitment β")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    else:
        ax.axis("off")

    ax = axes[1]
    if slot_rows:
        xs = [k for k, _, _ in slot_rows]
        gaps = [v["e2_gap"] for _, _, v in slot_rows]
        dRs = [v["delta_R_k64"] for _, _, v in slot_rows]
        colors = [TEAL if k == def_k else GRAY for k in xs]
        bars = ax.bar(
            range(len(xs)),
            gaps,
            color=colors,
            edgecolor="white",
            width=0.65,
            label="E2 gap",
        )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
            [f"K={k}" + (" ★" if k == def_k else "") for k in xs],
            fontsize=8,
        )
        ax.set_ylabel("E2 gap")
        ax.set_title(f"HPS: slots K  (★ default K={def_k})")
        ymax = max(gaps) * 1.35
        ax.set_ylim(0, ymax)
        for i, (b, g, d, k) in enumerate(zip(bars, gaps, dRs, xs)):
            tag = " default" if k == def_k else ""
            ax.text(
                b.get_x() + b.get_width() / 2,
                g + 0.01,
                f"{g:.2f}\nΔR={d:.1f}{tag}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=MUTED,
            )
        _style(ax)
    else:
        ax.axis("off")

    fig.suptitle(
        f"HPS vs full-model defaults (★ / star = {full_cell}: K={def_k}, β={def_beta:g})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out)


def plot_g6(summary: dict, out: Path) -> None:
    g6b = (summary["variants"].get("p2_C512_cd0.1") or {}).get("g6b")
    g6a = (summary["variants"].get("g6a_C512") or {}).get("g6a")
    if not g6b and not g6a:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    if g6b:
        # hit rate primary; annotate pearson
        names = ["true_S", "scramble_S", "random_S"]
        hits = [g6b[n]["mean_hit_rate_pm05"] for n in names]
        pears = [g6b[n]["logp_pearson"] for n in names]
        colors = [TEAL, GRAY, FAIL]
        ax = axes[0]
        bars = ax.bar(range(len(names)), hits, color=colors, edgecolor="white", width=0.65)
        ax.set_ylabel("Mean hit rate |ΔLogP| ≤ 0.5")
        ax.set_title("G6b: Spec conditioning (C=512, cd=0.1)")
        ax.set_ylim(0, max(hits) * 1.45)
        for b, h, p in zip(bars, hits, pears):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.01,
                f"hit {h:.2f}\nρ={p:+.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MUTED,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["true S", "scramble S", "random S"])
        _style(ax)
    else:
        axes[0].axis("off")

    if g6a:
        names = ["true_S", "property_y", "random_S"]
        hits = [g6a[n]["mean_hit_rate_pm05"] for n in names]
        pears = [g6a[n]["logp_pearson"] for n in names]
        colors = [TEAL, TEAL_LIGHT, FAIL]
        ax = axes[1]
        bars = ax.bar(range(len(names)), hits, color=colors, edgecolor="white", width=0.65)
        ax.set_ylabel("Mean hit rate |ΔLogP| ≤ 0.5")
        ax.set_title("G6a: Spec vs property Gen")
        ax.set_ylim(0, max(hits) * 1.45)
        for b, h, p in zip(bars, hits, pears):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.01,
                f"hit {h:.2f}\nρ={p:+.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MUTED,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["true S", "property y", "random S"])
        _style(ax)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    _save(fig, out)


def plot_compose_vs_retrieve(summary: dict, root: Path, out: Path) -> None:
    cells = [
        ("gen_C512_compose", "Compose\n(FAN)"),
        ("gen_C512_retrieve", "Retrieve\nonly"),
    ]
    headlines = []
    for key, lab in cells:
        p = root / key / "gen_eval_summary.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text()).get("headline") or {}
        headlines.append((lab, h, summary["variants"].get(key, {}).get("g4_rho")))

    if not headlines:
        return

    keys = [
        ("g4_logp_rho", "G4 LogP ρ"),
        ("g1_mean_match", "G1 match"),
        ("g2_mean_entropy", "G2 Murcko H"),
        ("g5_union_entropy", "G5 union H"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(11.5, 3.4))
    for ax, (k, title) in zip(axes, keys):
        labs = [h[0] for h in headlines]
        vals = [h[1].get(k, np.nan) for h in headlines]
        colors = [TEAL, GRAY]
        bars = ax.bar(labs, vals, color=colors[: len(vals)], edgecolor="white", width=0.6)
        ax.set_title(title, fontsize=10)
        ymax = max(v for v in vals if v == v)
        ymin = min(0, min(v for v in vals if v == v))
        ax.set_ylim(ymin, ymax * 1.25 if ymax > 0 else 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.02 * (ymax or 1)), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=MUTED)
        _style(ax)
    fig.suptitle(
        "Design: FAN compose vs retrieve-only (C=512)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, out)


# Design-decision cells: removing/changing these asks "was this choice necessary?"
_DESIGN_E2 = (
    "codebook_C512",  # full model reference
    "necessity_novq",
    "necessity_nocontrast",
    "necessity_wecfp0",
    "surrogates_physchem",
)
# Hyperparameter cells: same architecture, sweep knobs (robustness / selection)
_HPS_E2_PREFIXES = ("codebook_C", "beta_", "slots_K")


def _e2_rows(summary: dict, names: list[str]) -> list[tuple[str, str, float, float]]:
    out = []
    for name in names:
        v = summary["variants"].get(name)
        if not v or "e2_gap" not in v:
            continue
        out.append(
            (
                name,
                "pass" if v.get("e2_pass") else "FAIL",
                float(v["e2_gap"]),
                float(v.get("delta_R_k64") or 0.0),
            )
        )
    return out


def _barh_gap(ax, e2_rows: list[tuple[str, str, float, float]], title: str) -> None:
    y = np.arange(len(e2_rows))
    gaps = [r[2] for r in e2_rows]
    colors = [PASS if r[1] == "pass" else FAIL for r in e2_rows]
    ax.barh(y, gaps, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0.25, color=INK, ls="--", lw=1, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in e2_rows], fontsize=8)
    ax.set_xlabel("E2 behavior−structure gap")
    ax.set_title(title, fontsize=10)
    xmax = max(gaps) * 1.35 if gaps else 1.0
    for yi, g, d in zip(y, gaps, [r[3] for r in e2_rows]):
        ax.text(g + 0.004, yi, f"{g:.3f}  ·  ΔR={d:.1f}", va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, xmax)
    _style(ax)
    ax.invert_yaxis()


def plot_overview(summary: dict, out: Path) -> None:
    """Split scorecard: design necessity (left) vs hyperparameter sweep (right)."""
    design = _e2_rows(summary, list(_DESIGN_E2))
    hps_names = sorted(
        n
        for n, v in summary["variants"].items()
        if "e2_gap" in v and any(n.startswith(p) for p in _HPS_E2_PREFIXES)
    )
    hps = _e2_rows(summary, hps_names)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, max(3.8, 0.34 * max(len(design), len(hps)) + 1.4)),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    _barh_gap(
        axes[0],
        design,
        "A. Design decisions\n(fail = that choice was necessary)",
    )
    _barh_gap(
        axes[1],
        hps,
        "B. Hyperparameters\n(all should stay near full / pass E2)",
    )
    fig.suptitle(
        "Two ablation stories — do not mix: necessity (A) vs knob sweep (B)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("runs/ablations"),
        help="Ablation suite output directory",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Figures directory (default: <root>/figures)",
    )
    args = ap.parse_args()
    root = args.root
    out = args.out or (root / "figures")
    summary = _load(root)

    plot_overview(summary, out / "fig_ablation_e2_scorecard")
    plot_codebook_sweep(summary, root, out / "fig_ablation_codebook")
    plot_necessity(summary, root, out / "fig_ablation_necessity")
    plot_hps(summary, root, out / "fig_ablation_hps")
    plot_compose_vs_retrieve(summary, root, out / "fig_ablation_compose")
    plot_g6(summary, out / "fig_ablation_g6")
    print(f"figures → {out.resolve()}")


if __name__ == "__main__":
    main()
