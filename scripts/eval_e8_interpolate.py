#!/usr/bin/env python3
"""E8 interpolation path report.

--validity-csv: one column validity with T rows
--surrogate-npy: (T, d)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from functionalspec.eval.harness import dump_json, e8_path_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validity-csv", type=Path, required=True)
    p.add_argument("--surrogate-npy", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e8.json"))
    args = p.parse_args()
    v = pd.read_csv(args.validity_csv).iloc[:, 0].astype(float).tolist()
    Y = np.load(args.surrogate_npy)
    report = e8_path_report(v, Y)
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
