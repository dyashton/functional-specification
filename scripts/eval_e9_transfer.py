#!/usr/bin/env python3
"""E9 frozen transfer low-n curves.

Provide feature matrices as .npy and labels as .npy; methods named via --method name:path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from functionalspec.eval.harness import dump_json, e9_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--y", type=Path, required=True)
    p.add_argument("--method", action="append", required=True, help="name:path.npy")
    p.add_argument("--task-type", default="regression", choices=["regression", "classification"])
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e9.json"))
    args = p.parse_args()

    y = np.load(args.y)
    feats = {}
    for spec in args.method:
        name, path = spec.split(":", 1)
        feats[name] = np.load(path)
    report = e9_report(feats, y, task_type=args.task_type)
    # json-serialize int keys
    report["ranks"] = {str(k): v for k, v in report["ranks"].items()}
    report["curves"] = {m: {str(k): v for k, v in curve.items()} for m, curve in report["curves"].items()}
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
