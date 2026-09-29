#!/usr/bin/env python3
"""E6 token intervention: compare surrogate means with vs without a token."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from functionalspec.eval.harness import dump_json, e6_token_delta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--with-npy", type=Path, required=True, help="(n,d) surrogates with token")
    p.add_argument("--without-npy", type=Path, required=True, help="(n,d) surrogates without token")
    p.add_argument("--out", type=Path, default=Path("data/processed/eval/e6.json"))
    args = p.parse_args()
    report = e6_token_delta(np.load(args.with_npy), np.load(args.without_npy))
    dump_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
