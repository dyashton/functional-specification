#!/usr/bin/env python3
"""MVP Tier-1/2 ablation suite: codebook, design necessity, HPS (β/K), P2, gen.

Resumable phases. Artifacts under --out (default runs/ablations/).

Two stories (see runs/ablations/README.md):
  A. Design necessity — no VQ / no contrast / physchem / w_ecfp
  B. Hyperparameters — |C|, β, K (defaults = full model: C=top1, K=12, β=0.25)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Full-model defaults (must match configs/model commitment_cost / num_slots)
DEFAULT_BETA = 0.25
DEFAULT_SLOTS = 12
BETA_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SLOT_GRID = [4, 6, 8, 12, 16, 24]


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _uv(*args: str) -> list[str]:
    return ["uv", "run", "python", *args]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _delta_r_k64(spec_dir: Path) -> float:
    p = spec_dir / "specificity_summary.json"
    if not p.exists():
        return float("-inf")
    s = _load_json(p)
    row = s.get("soft_neighborhood_summaries", {}).get("64", {})
    return float(row.get("delta_R_mean", float("-inf")))


def _e2_pass(e2_dir: Path) -> bool:
    p = e2_dir / "e2_summary.json"
    if not p.exists():
        return False
    return bool(_load_json(p).get("pass", False))


def _e2_gap(e2_dir: Path) -> float:
    p = e2_dir / "e2_summary.json"
    if not p.exists():
        return float("-inf")
    return float(_load_json(p).get("gap_spearman", float("-inf")))


def eval_p1(
    ckpt: Path,
    cell: Path,
    *,
    corpus_a: Path,
    n_probes: int,
    recon_epochs: int,
    residualize: bool = False,
) -> None:
    e2_out = cell / "e2"
    spec_out = cell / "spec"
    if not (e2_out / "e2_summary.json").exists():
        _run(
            _uv(
                "scripts/run_e2.py",
                "--checkpoint",
                str(ckpt),
                "--corpus-a",
                str(corpus_a),
                "--out",
                str(e2_out),
                "--recon-epochs",
                str(recon_epochs),
            )
        )
    if not (spec_out / "specificity_summary.json").exists():
        cmd = _uv(
            "scripts/run_spec_specificity.py",
            "--checkpoint",
            str(ckpt),
            "--corpus-a",
            str(corpus_a),
            "--out",
            str(spec_out),
            "--k",
            "32",
            "64",
            "128",
            "--n-probes",
            str(n_probes),
        )
        _run(cmd)
    if residualize and not (spec_out / "specificity_summary_resid.json").exists():
        _run(
            _uv(
                "scripts/run_spec_specificity.py",
                "--checkpoint",
                str(ckpt),
                "--corpus-a",
                str(corpus_a),
                "--out",
                str(spec_out),
                "--k",
                "32",
                "64",
                "128",
                "--n-probes",
                str(n_probes),
                "--residualize-logp-tpsa-mw",
            )
        )


def train_p1_if_needed(cell: Path, p1_args: list[str], epochs: int, corpus_a: Path) -> Path:
    ckpt = cell / "p1" / "p1_best.pt"
    if ckpt.exists():
        print(f"skip train (exists): {ckpt}")
        return ckpt
    out = cell / "p1"
    _run(
        _uv(
            "scripts/train_p1.py",
            "--corpus-a",
            str(corpus_a),
            "--out",
            str(out),
            "--epochs",
            str(epochs),
            *p1_args,
        )
    )
    return ckpt


def _beta_cell_name(beta: float) -> str:
    t = f"{beta:.4f}".rstrip("0").rstrip(".")
    if "." not in t:
        t += ".0"
    return f"beta_{t}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _reuse_full_p1(args: argparse.Namespace, primary: int, cell: Path) -> Path | None:
    """Point cell at codebook_C{primary} (full-model defaults K=12, β=0.25)."""
    src = args.out / f"codebook_C{primary}" / "p1" / "p1_best.pt"
    if not src.exists():
        return None
    link = cell / "p1" / "p1_best.pt"
    _link_or_copy(src, link)
    return link


def pick_top2(out: Path, sizes: list[int]) -> list[int]:
    rows = []
    for c in sizes:
        cell = out / f"codebook_C{c}"
        rows.append(
            {
                "C": c,
                "e2_pass": _e2_pass(cell / "e2"),
                "gap": _e2_gap(cell / "e2"),
                "delta_R": _delta_r_k64(cell / "spec"),
            }
        )
    passed = [r for r in rows if r["e2_pass"]]
    pool = passed if len(passed) >= 2 else sorted(rows, key=lambda r: (r["gap"], r["delta_R"]), reverse=True)
    ranked = sorted(pool, key=lambda r: r["delta_R"], reverse=True)
    top = [int(r["C"]) for r in ranked[:2]]
    (out / "top2_codebooks.json").write_text(json.dumps({"candidates": rows, "top2": top}, indent=2))
    print("top2 codebooks:", top)
    return top


def _write_defaults(out: Path, primary: int) -> None:
    meta = {
        "full_model_cell": f"codebook_C{primary}",
        "defaults": {
            "codebook_size": primary,
            "num_slots": DEFAULT_SLOTS,
            "commitment_cost_beta": DEFAULT_BETA,
            "contrastive": True,
            "surrogates": "full",
        },
        "hps_grids": {"beta": BETA_GRID, "num_slots": SLOT_GRID},
        "note": (
            f"Default HPS cells slots_K{DEFAULT_SLOTS} and {_beta_cell_name(DEFAULT_BETA)} "
            f"reuse {primary} full-model checkpoint (symlink)."
        ),
    }
    (out / "defaults.json").write_text(json.dumps(meta, indent=2))


def phase_codebook(args: argparse.Namespace) -> list[int]:
    sizes = [32, 64, 128, 256, 512]
    for c in sizes:
        cell = args.out / f"codebook_C{c}"
        ckpt = train_p1_if_needed(
            cell,
            ["--codebook-size", str(c)],
            args.epochs,
            args.corpus_a,
        )
        eval_p1(
            ckpt,
            cell,
            corpus_a=args.corpus_a,
            n_probes=args.n_probes,
            recon_epochs=args.recon_epochs,
            residualize=(c == 128),
        )
    top = pick_top2(args.out, sizes)
    _write_defaults(args.out, top[0] if top else 512)
    return top


def phase_necessity(args: argparse.Namespace, top2: list[int]) -> None:
    """Story A — design necessity only (not β/K HPS)."""
    primary = top2[0] if top2 else 512
    cells: list[tuple[str, list[str]]] = [
        ("necessity_novq", ["--no-vq"]),
        ("necessity_nocontrast", ["--contrastive-weight", "0", "--codebook-size", str(primary)]),
        ("necessity_wecfp0", ["--w-ecfp", "0", "--codebook-size", str(primary)]),
        (
            "surrogates_physchem",
            [
                "--config",
                str(ROOT / "configs/ablations/physchem_only.yaml"),
                "--codebook-size",
                str(primary),
            ],
        ),
    ]
    for name, p1_args in cells:
        cell = args.out / name
        ckpt = train_p1_if_needed(cell, p1_args, args.epochs, args.corpus_a)
        eval_p1(
            ckpt,
            cell,
            corpus_a=args.corpus_a,
            n_probes=args.n_probes,
            recon_epochs=args.recon_epochs,
        )


def phase_hps(args: argparse.Namespace, top2: list[int]) -> None:
    """Story B — β and K sweeps at full-model |C| (defaults marked via reuse)."""
    primary = top2[0] if top2 else 512
    _write_defaults(args.out, primary)

    jobs: list[tuple[str, list[str], bool]] = []
    for beta in BETA_GRID:
        name = _beta_cell_name(beta)
        is_default = abs(beta - DEFAULT_BETA) < 1e-9
        jobs.append(
            (
                name,
                ["--commitment-cost", str(beta), "--codebook-size", str(primary), "--num-slots", str(DEFAULT_SLOTS)],
                is_default,
            )
        )
    for k in SLOT_GRID:
        name = f"slots_K{k}"
        is_default = k == DEFAULT_SLOTS
        jobs.append(
            (
                name,
                ["--num-slots", str(k), "--codebook-size", str(primary), "--commitment-cost", str(DEFAULT_BETA)],
                is_default,
            )
        )

    for name, p1_args, is_default in jobs:
        cell = args.out / name
        if is_default:
            link = _reuse_full_p1(args, primary, cell)
            if link is not None:
                eval_p1(
                    link,
                    cell,
                    corpus_a=args.corpus_a,
                    n_probes=args.n_probes,
                    recon_epochs=args.recon_epochs,
                )
                continue
        ckpt = train_p1_if_needed(cell, p1_args, args.epochs, args.corpus_a)
        eval_p1(
            ckpt,
            cell,
            corpus_a=args.corpus_a,
            n_probes=args.n_probes,
            recon_epochs=args.recon_epochs,
        )


def phase_p2(args: argparse.Namespace, top2: list[int]) -> None:
    if not top2:
        top2 = pick_top2(args.out, [32, 64, 128, 256, 512])
    # top2 at cd=0.1; winner also cd in {0, 0.2}
    jobs: list[tuple[int, float]] = [(c, 0.1) for c in top2]
    jobs += [(top2[0], 0.0), (top2[0], 0.2)]
    seen = set()
    for c, cd in jobs:
        key = (c, cd)
        if key in seen:
            continue
        seen.add(key)
        cell = args.out / f"p2_C{c}_cd{cd}"
        p1 = args.out / f"codebook_C{c}" / "p1" / "p1_best.pt"
        p2_ckpt = cell / "p2_best.pt"
        if not p2_ckpt.exists():
            _run(
                _uv(
                    "scripts/train_p2.py",
                    "--p1-checkpoint",
                    str(p1),
                    "--corpus-a",
                    str(args.corpus_a),
                    "--out",
                    str(cell),
                    "--epochs",
                    str(args.epochs),
                    "--representation",
                    "selfies",
                    "--cond-dropout",
                    str(cd),
                )
            )
        bank = cell / "spec_bank.npz"
        if not bank.exists():
            _run(
                _uv(
                    "scripts/build_spec_bank.py",
                    "--p1-checkpoint",
                    str(p1),
                    "--corpus-a",
                    str(args.corpus_a),
                    "--out",
                    str(bank),
                )
            )


def phase_gen(args: argparse.Namespace, top2: list[int]) -> None:
    if not top2:
        top2 = pick_top2(args.out, [32, 64, 128, 256, 512])
    win = top2[0]
    for c in top2:
        cell = args.out / f"p2_C{c}_cd0.1"
        p1 = args.out / f"codebook_C{c}" / "p1" / "p1_best.pt"
        gen_out = cell / "gen"
        g6b_out = cell / "g6b"
        if not (gen_out / "gen_eval_summary.json").exists():
            _run(
                _uv(
                    "scripts/run_gen_eval.py",
                    "--generator",
                    str(cell / "p2_best.pt"),
                    "--p1-checkpoint",
                    str(p1),
                    "--bank",
                    str(cell / "spec_bank.npz"),
                    "--out",
                    str(gen_out),
                    "--n",
                    str(args.n_gen),
                )
            )
        if not (g6b_out / "g6b_summary.json").exists():
            _run(
                _uv(
                    "scripts/run_g6b.py",
                    "--generator",
                    str(cell / "p2_best.pt"),
                    "--p1-checkpoint",
                    str(p1),
                    "--bank",
                    str(cell / "spec_bank.npz"),
                    "--out",
                    str(g6b_out),
                    "--n",
                    str(args.n_gen),
                )
            )

    # compose vs retrieve on winner
    win_cell = args.out / f"p2_C{win}_cd0.1"
    p1 = args.out / f"codebook_C{win}" / "p1" / "p1_best.pt"
    for tag, compose_flag in (("compose", []), ("retrieve", ["--no-compose"])):
        out = args.out / f"gen_C{win}_{tag}"
        if (out / "gen_eval_summary.json").exists():
            continue
        _run(
            _uv(
                "scripts/run_gen_eval.py",
                "--generator",
                str(win_cell / "p2_best.pt"),
                "--p1-checkpoint",
                str(p1),
                "--bank",
                str(win_cell / "spec_bank.npz"),
                "--out",
                str(out),
                "--n",
                str(args.n_gen),
                *compose_flag,
            )
        )

    # G6a once for winner
    prop = args.out / f"p2_property_C{win}"
    if not (prop / "p2_property_best.pt").exists():
        _run(
            _uv(
                "scripts/train_p2_property.py",
                "--p1-checkpoint",
                str(p1),
                "--corpus-a",
                str(args.corpus_a),
                "--out",
                str(prop),
                "--epochs",
                str(args.epochs),
            )
        )
    g6a = args.out / f"g6a_C{win}"
    if not (g6a / "g6a_summary.json").exists():
        _run(
            _uv(
                "scripts/run_g6a.py",
                "--s-checkpoint",
                str(win_cell / "p2_best.pt"),
                "--y-checkpoint",
                str(prop / "p2_property_best.pt"),
                "--p1-checkpoint",
                str(p1),
                "--bank",
                str(win_cell / "spec_bank.npz"),
                "--out",
                str(g6a),
                "--n",
                str(args.n_gen),
            )
        )


def phase_summarize(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {"out": str(args.out), "variants": {}}
    top2_path = args.out / "top2_codebooks.json"
    if top2_path.exists():
        summary["top2"] = _load_json(top2_path)
    defaults_path = args.out / "defaults.json"
    if defaults_path.exists():
        summary["defaults"] = _load_json(defaults_path)
    else:
        primary = (summary.get("top2") or {}).get("top2", [512])[0]
        summary["defaults"] = {
            "full_model_cell": f"codebook_C{primary}",
            "defaults": {
                "codebook_size": primary,
                "num_slots": DEFAULT_SLOTS,
                "commitment_cost_beta": DEFAULT_BETA,
            },
        }

    for cell in sorted(args.out.iterdir()):
        if not cell.is_dir():
            continue
        entry: dict[str, Any] = {"path": str(cell)}
        e2 = cell / "e2" / "e2_summary.json"
        if e2.exists():
            d = _load_json(e2)
            entry["e2_pass"] = d.get("pass")
            entry["e2_gap"] = d.get("gap_spearman")
        spec = cell / "spec" / "specificity_summary.json"
        if spec.exists():
            d = _load_json(spec)
            entry["delta_R_k64"] = d.get("soft_neighborhood_summaries", {}).get("64", {}).get("delta_R_mean")
            entry["code_usage"] = d.get("code_usage") or d.get("discrete_code_occupancy")
        gen = cell / "gen" / "gen_eval_summary.json"
        if gen.exists():
            entry["g4_rho"] = _load_json(gen).get("headline", {}).get("g4_logp_rho")
        for g6b_name in ("g6b_summary.json", "summary.json"):
            g6b = cell / "g6b" / g6b_name
            if g6b.exists():
                entry["g6b"] = _load_json(g6b).get("headline")
                break
        summary["variants"][cell.name] = entry

    # compose/retrieve + g6a at top level
    for p in args.out.glob("gen_C*_*/gen_eval_summary.json"):
        summary["variants"][p.parent.name] = {
            "path": str(p.parent),
            "g4_rho": _load_json(p).get("headline", {}).get("g4_logp_rho"),
            "compose": _load_json(p).get("compose"),
        }
    for p in args.out.glob("g6a_C*/**/g6a_summary.json"):
        summary["variants"][p.parent.name] = {"path": str(p.parent), "g6a": _load_json(p).get("headline")}

    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return summary


def _resolve_top2(args: argparse.Namespace) -> list[int]:
    p = args.out / "top2_codebooks.json"
    if p.exists():
        return list(_load_json(p)["top2"])
    return pick_top2(args.out, [32, 64, 128, 256, 512])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["all", "codebook", "necessity", "hps", "p2", "gen", "summarize"],
        default="all",
    )
    p.add_argument("--out", type=Path, default=Path("runs/ablations"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--recon-epochs", type=int, default=15)
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--n-gen", type=int, default=64)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    top2: list[int] = []
    if args.phase in ("all", "codebook"):
        top2 = phase_codebook(args)
    if args.phase in ("all", "necessity"):
        top2 = top2 or _resolve_top2(args)
        phase_necessity(args, top2)
    if args.phase in ("all", "hps"):
        top2 = top2 or _resolve_top2(args)
        phase_hps(args, top2)
    if args.phase in ("all", "p2"):
        top2 = top2 or _resolve_top2(args)
        phase_p2(args, top2)
    if args.phase in ("all", "gen"):
        top2 = top2 or _resolve_top2(args)
        phase_gen(args, top2)
    if args.phase in ("all", "summarize"):
        phase_summarize(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"command failed with {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
