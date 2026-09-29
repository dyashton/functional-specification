#!/usr/bin/env python3
"""E2: surrogate vs structure probes on frozen S (+ Recon-VQ control)."""

from __future__ import annotations

import argparse
from pathlib import Path

from functionalspec.eval.e2_run import run_e2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/p1/p1_best.pt"),
        help="P1 or P2 checkpoint (encoder/VQ used; arm ignored)",
    )
    p.add_argument("--corpus-a", type=Path, default=Path("data/processed/corpus_a"))
    p.add_argument("--out", type=Path, default=Path("runs/p1/e2"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--fp-bits-probe", type=int, default=128)
    p.add_argument("--recon-epochs", type=int, default=30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    report = run_e2(
        checkpoint=args.checkpoint,
        corpus_a_dir=args.corpus_a,
        out_dir=args.out,
        split=args.split,
        batch_size=args.batch_size,
        fp_bits_probe=args.fp_bits_probe,
        recon_epochs=args.recon_epochs,
        device=args.device,
        seed=args.seed,
    )
    print(
        f"E2 pass={report['pass']} gap={report['gap_spearman']:.3f} "
        f"surr_spear={report['surrogate_probe']['mean_spearman']:.3f} "
        f"struct_spear={report['structure_probe']['mean_spearman']:.3f} "
        f"control_struct={report['structure_control_spearman']:.3f} "
        f"margin={report['structure_margin']:.3f}"
    )
    print(
        f"NN test: S-NN T={report['nn_test']['mean_tanimoto_S_NN']:.3f} "
        f"FP-NN T={report['nn_test']['mean_tanimoto_FP_NN']:.3f} "
        f"margin={report['nn_test']['margin']:.3f} nn_pass={report['nn_pass']}"
    )
    print(f"→ {args.out / 'e2_summary.json'}")


if __name__ == "__main__":
    main()
