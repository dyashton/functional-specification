# Eval Harness Checklist

Run from `FunctionalSpec/` after `uv sync` and data prep.

## Prerequisites

```bash
uv sync
uv run python scripts/prepare_corpus_b.py \
  --input ../CO2_IE_Dataset/data/compiled.csv \
  --out data/processed/corpus_b

uv run python scripts/prepare_corpus_a.py \
  --input ../Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv \
  --out data/processed/corpus_a
# optional smoke: --max-rows 2000
```

Place E9 held-out CSVs under `data/processed/e9/{task_id}.csv` when available (see DATA_SPEC).

---

## Experiment runners

| Exp | Script | What it does |
|-----|--------|----------------|
| E0 | `scripts/eval_e0_sanity.py` | Validity / unique rates on a SMILES list |
| E2 | `scripts/run_e2.py` | Embed checkpoint → surrogate vs ECFP-bit probes + Recon/FP-PCA control |
| E3 | `scripts/run_e3.py` | Freeze \(S^*\), sample Arm A, matched-behavior \(R\) / scaffolds |
| E6 | `scripts/eval_e6_token_intervention.py` | Token knockout Δsurrogate |
| E7 | `scripts/eval_e7_token_atlas.py` | Per-token surrogate histograms (CSV) |
| E8 | `scripts/eval_e8_interpolate.py` | Path validity / surrogate smoothness |
| E9 | `scripts/run_e9.py` | Frozen S vs ECFP low-n transfer (after `prepare_e9.py`) |
| G1–G5 | `scripts/run_gen_eval.py` | Spec→Generator: match, diversity, confidence, LogP ladder, repeatability |
| Sweep | `scripts/eval_codebook_sweep.py` | Aggregate stub metrics over \|C\| |
| All | `scripts/run_eval_checklist.py` | Prints checklist + runs offline metric self-checks |

Generation stack: `scripts/build_spec_bank.py` → `scripts/generate_from_spec.py` (FAN outcomes `exists|uncertain|unsupported`).

Library API: `functionalspec/eval/harness.py`, `functionalspec/gen/`.

---

## E3 matched-behavior protocol (mandatory)

1. Encode / pick \(S^*\); sample 1000 from Arm A (or B).
2. Compute set mean surrogate \(\bar y_{S^*}\).
3. Sample from Recon-VQ / structural baseline; filter or resample until
   \(\|\bar y_{S^*}-\bar y_{\mathrm{base}}\|_2/\sqrt{d}\le 0.15\).
4. Compare \(D_{\mathrm{struct}}\), \(N_{\mathrm{scaf}}\), \(R\) only under match.

Without step 3–4, do not claim E3 pass.

---

## Pass/fail source of truth

[`docs/METRICS.md`](METRICS.md) and `functionalspec/metrics/thresholds.py`.

---

## Training (not required for design-phase checklist)

P1→P2→P3 trainers are intentionally thin stubs; plug TransPharmer / full graph batches when scaling. The harness evaluates **arrays of SMILES + surrogates + optional S embeddings**, so it works on model dumps or oracle-labeled sets.
