# Functional Specification Bottleneck — MVP Design Doc

This document is the project-facing design summary. The research plan file is authoritative for narrative history; **do not edit the plan from this repo workflow**. Implementation lives under `FunctionalSpec/`.

## Three questions (architecture invariant)

The system answers **three different questions**, each with its own representation:

| # | Question | Representation |
|---|----------|----------------|
| 1 | What does the environment look like? | Environment / Environment Embedding |
| 2 | What capabilities are required? | Functional Specification \(S=\{s_1,\ldots,s_K\}\) |
| 3 | What chemistry realizes those capabilities? | Molecule (graph / SELFIES / SMILES) |

```text
Interaction Environment → Environment Encoder → Embedding
        → Functional Abstraction Network (FAN) → Functional Specification
        → Molecule Generator → Molecule(s)
```

- **Environment Encoder** — descriptive only (“what is this environment?”). Does not decide chemistry.
- **FAN** (canonical name; informal “planner” OK) — abstracts to the *minimal functional requirements* for success. Not “compress the environment” and not “predict descriptors” as the end goal.
- **Generator** — realizes Spec → chemistry; diversity under one Spec is expected.

**Spec interface ≠ payload.** Callers depend on `FunctionalSpecification` (`functionalspec/gen/spec.py`). Today’s payload is `FlatSPayload(flat_S)`; Phase II may change the payload without rewriting the Generator contract.

### MVP vs Phase II

| | MVP (now) | Phase II |
|---|-----------|----------|
| Upstream of FAN | `DesignObjective` (descriptors / affinity / lead) | Environment Embedding from pockets, CO₂, pores, … |
| FAN | Bank retrieve → rank → compose (on-manifold) | Trained env → Spec abstraction |
| Spec → Generator | Same `FunctionalSpecification` + Arm A SELFIES | Unchanged contract |

```text
DesignObjective ──MVP──► FAN_MVP ──► FunctionalSpecification ──► Generator
EnvironmentEmb ─PhaseII► FAN     ──► FunctionalSpecification ──► Generator
```

So MVP proves **Spec is a reusable conditioning language**; Phase II proves **FAN can infer Spec from environments**.

**MVP manuscript:** [`docs/papers/MVP_MANUSCRIPT.md`](docs/papers/MVP_MANUSCRIPT.md) (claims lock + Abstract–Conclusion). Phase II stays out of that paper.

## Publishable MVP claim

\(S\) is a **functional bottleneck**: more predictive of molecular **behavior** (surrogate views) than **structure**, and a single \(S\) admits many chemically distinct realizations of the same design problem (high structural diversity, low behavioral variance).

- **Abstraction → E3** (matched-behavior baseline mandatory)
- **Not-just-a-bottleneck → E2**
- **Spec→chemistry → G1–G5** (match, diversity, confidence, LogP ladder, repeatability)
- **Frozen transfer → E9** — **scope limit / negative result** under the strict protocol (1/5 tasks; solubility only). Do **not** claim universal frozen transfer.

Phase II adds the Environment Encoder and earns the name *Interaction Specification* (FAN swap + encoder only — Generator stays). See [docs/PHASE2.md](docs/PHASE2.md).

## Docs

| Doc | Content |
|-----|---------|
| [DATA_SPEC.md](docs/DATA_SPEC.md) | Corpus A/B, E9 tasks, contrastive weights |
| [METRICS.md](docs/METRICS.md) | Operational pass/fail |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Architecture, arms, baselines, codebook sweep |
| [EVAL_HARNESS.md](docs/EVAL_HARNESS.md) | Scripts and checklist |
| [BEHAVIORAL_GEOMETRY_PAPER.md](docs/BEHAVIORAL_GEOMETRY_PAPER.md) | Soft neighborhoods / \(v_{\mathrm{beh}}\) |

## Quickstart

```bash
cd FunctionalSpec
uv sync
uv run pytest -q
uv run python scripts/prepare_corpus_b.py \
  --input ../CO2_IE_Dataset/data/compiled.csv \
  --out data/processed/corpus_b
uv run python scripts/prepare_corpus_a.py \
  --input ../Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv \
  --out data/processed/corpus_a
uv run python scripts/run_eval_checklist.py
```

## P1 training (next after corpora)

```bash
uv run python scripts/train_p1.py \
  --corpus-a data/processed/corpus_a \
  --out runs/p1 \
  --epochs 30 \
  --batch-size 64
```

Watch `val_spearman` (want high) vs `struct_R2` (want low-ish vs a recon baseline). Checkpoint: `runs/p1/p1_best.pt`.

## P2 training (after P1) — SELFIES Arm A with FiLM conditioning

Arm A conditions on Spec payload (`flat_S`) with **per-step FiLM** + optional cond dropout (CFG-style). Prefer the conditioned run:

```bash
uv run python scripts/train_p2.py \
  --p1-checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p2_selfies_cond \
  --representation selfies \
  --epochs 15 \
  --batch-size 64 \
  --cond-dropout 0.1

uv run python scripts/sample_from_s.py \
  --checkpoint runs/p2_selfies_cond/p2_best.pt \
  --n-mols 64 \
  --samples-per-s 8 \
  --out runs/p2_selfies_cond/samples.csv
```

## Arm B comparison — learned motifs with atom expansion

Arm B uses BRICS fragments only as supervision candidates. A trainable VQ
codebook predicts motif slots from `flat_S`, and a motif-conditioned SELFIES
decoder expands those slots into molecules:

```bash
uv run python -m functionalspec.train.p2_arm_b \
  --p1-checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p2_arm_b \
  --representation selfies \
  --epochs 15 \
  --batch-size 64
```

Run the matched comparison after both checkpoints exist:

```bash
uv run python scripts/compare_arms.py \
  --arm-a runs/p2_selfies_cond/p2_best.pt \
  --arm-b runs/p2_arm_b/arm_b_best.pt \
  --bank runs/gen/spec_bank.npz \
  --p1-checkpoint runs/p1/p1_best.pt \
  --out runs/gen/arm_comparison \
  --n 64
```

## Spec bank + FAN + generate

```bash
uv run python scripts/build_spec_bank.py \
  --p1-checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/gen/spec_bank.npz

uv run python scripts/generate_from_spec.py \
  --objective "LogP=3,TPSA=50" \
  --bank runs/gen/spec_bank.npz \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --n 100 \
  --out runs/gen/demo
```

FAN outcomes: **exists | uncertain | unsupported**. Specificity \(R\) is **internal** to ranking/confidence (not a user-facing Spec field required by the Generator).

## E3 (after P2 SELFIES)

```bash
uv run python scripts/run_e3.py \
  --checkpoint runs/p2_selfies_cond/p2_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p2_selfies_cond/e3 \
  --n-specs 5 \
  --n-samples 1000
```

Writes `e3_summary.json` plus per-spec sample CSVs. Pass requires high scaffolds / \(R\) vs a matched-behavior corpus baseline. Re-check C1 (behavior explode) via `scripts/run_e3_quantify.py`.

## G1–G5 (Spec→Generator scientific eval)

```bash
uv run python scripts/run_gen_eval.py \
  --bank runs/gen/spec_bank.npz \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --out runs/gen/eval \
  --n 64
```

| ID | Question |
|----|----------|
| G1 | Match requested design-objective behavior? |
| G2 | Structural diversity among accepted molecules? |
| G3 | Does confidence predict generation reliability? |
| G4 | LogP ladder — Spec trajectory + realized LogP |
| G5 | Same Spec, many batches — stable behavior, diverse chemistry |

## G6 ablations (is \(S\) necessary?)

```bash
# G6b: scramble / random flat_S (no retrain) — decoder uses Spec?
uv run python scripts/run_g6b.py \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --out runs/gen/g6b --n 64

# G6a Model B: matched property→SELFIES FiLM decoder
uv run python scripts/train_p2_property.py \
  --p1-checkpoint runs/p1/p1_best.pt \
  --out runs/p2_property --epochs 15

# G6a: S vs property-y vs random-S
uv run python scripts/run_g6a.py \
  --s-checkpoint runs/p2_selfies_cond/p2_best.pt \
  --y-checkpoint runs/p2_property/p2_property_best.pt \
  --out runs/gen/g6a --n 64
```

| Kind | Question |
|------|----------|
| true \(S\) | Spec → chemistry |
| scramble / random \(S\) | Does the decoder ignore payload? |
| property \(y\) | Why not condition on descriptors directly? |

## E2 (publishability gate)

```bash
uv run python scripts/run_e2.py \
  --checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p1/e2 \
  --split val
```

Reports surrogate vs structure probe gap and Recon-VQ structure margin.

## E9 (held-out transfer)

```bash
# once: download MoleculeNet + build flexibility CSV
uv run python scripts/prepare_e9.py \
  --out data/processed/e9 \
  --corpus-a data/processed/corpus_a

uv run python scripts/run_e9.py \
  --checkpoint runs/p1/p1_best.pt \
  --e9-dir data/processed/e9 \
  --out runs/p1/e9
```

Pass: ≥3/5 tasks where frozen \(S\) wins majority of finite low-\(n\) ranks **and** full-data score ≥ ECFP − 0.02.

## Specification specificity (encoder-side neighborhoods)

Exact VQ codes are nearly unique on Corpus A — analyze soft \(k\)-NN balls in continuous \(S\) (and k-means partitions):

```bash
uv run python scripts/run_spec_specificity.py \
  --checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p1/spec_specificity \
  --k 32 64 128 \
  --n-probes 200
```

Writes CSVs for \(H_{\mathrm{Murcko}}\) vs \(v_{\mathrm{beh}}\) scatter and \(R=(H_m+H_b)/v_{\mathrm{beh}}\) histograms; ECFP neighborhoods of equal \(k\) as control.

Paper write-up of results + narrative: [docs/BEHAVIORAL_GEOMETRY_PAPER.md](docs/BEHAVIORAL_GEOMETRY_PAPER.md).

## Surrogate language (mandatory)

Physicochemical and QM descriptors are **imperfect proxies for molecular function**, used because mechanistic labels are unavailable.
