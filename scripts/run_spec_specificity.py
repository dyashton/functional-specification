#!/usr/bin/env python3
"""Encoder-side functional neighborhoods: specificity of S vs ECFP partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.spec_specificity import run_spec_specificity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/p1/p1_best.pt"))
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p1/spec_specificity"))
    p.add_argument("--k", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--n-probes", type=int, default=200)
    p.add_argument("--n-clusters", type=int, default=100)
    p.add_argument("--min-cluster-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--residualize-logp-tpsa-mw",
        action="store_true",
        help="OLS-residualize surrogates on LogP/TPSA/MW before v_beh",
    )
    args = p.parse_args()

    report = run_spec_specificity(
        checkpoint=args.checkpoint,
        corpus_a=args.corpus_a,
        out_dir=args.out,
        k_list=tuple(args.k),
        n_probes=args.n_probes,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        residualize_logp_tpsa_mw=args.residualize_logp_tpsa_mw,
    )
    print(
        "Discrete codes nearly unique:",
        report["discrete_code_occupancy"]["n_unique_codes"],
        "/",
        report["discrete_code_occupancy"]["n_mols"],
    )
    for k, s in report["soft_neighborhood_summaries"].items():
        print(
            f"k={k}: frac(S R > ECFP R)={s['frac_S_higher_R_than_ECFP']:.2f} "
            f"mean ΔR={s['delta_R_mean']:.2f} mean v_beh(S)={s['v_beh_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
