# Behavioral Geometry of Chemical Space

**Status:** living evidence note (updated 2026-09-16)  
**Manuscript extract:** [`papers/MVP_MANUSCRIPT.md`](papers/MVP_MANUSCRIPT.md) — submission-shaped MVP draft (claims lock + Abstract–Conclusion). Write from that file; keep numbers here as the source of truth.

**Artifacts:**
- Encoder geometry: `runs/p1/spec_specificity/` · `fig_behavioral_geometry.{png,pdf}`
- Spec→Generator: `runs/gen/eval/gen_eval_summary.json` (G1–G5)
- Ablations: `runs/gen/g6b/g6b_summary.json`, `runs/gen/g6a/g6a_summary.json`
- Ladder figure: `runs/gen/fig_logp_ladder.{png,pdf}` · `scripts/plot_logp_ladder.py`
- Extra paper figs: `runs/gen/paper_figs/` · `scripts/plot_mvp_figures.py` (pipeline placeholder, E2, neighborhood cartoon, G5)
- FiLM Arm A: `runs/p2_selfies_cond/`; property baseline: `runs/p2_property/`

**Related runs:** P1 `runs/p1/`, E2 `runs/p1/e2/`, E3 `runs/p2_selfies_cond/e3/` (FiLM) and earlier `runs/p2_selfies/e3/`, E9 `runs/p1/e9/`, Spec bank `runs/gen/spec_bank.npz`

---

## 1. One-sentence result

The learned latent \(S\) induces a **behavioral geometry** of chemical space (structure-diverse, behavior-coherent neighborhoods vs ECFP), and — as a **generative control interface** — maps design objectives to diverse chemistries with predictable property movement (LogP ladder \(\rho\approx 0.99\)), outperforming matched descriptor conditioning on diversity.

---

## 2. Stored quantitative results

### 2.1 Encoder-side soft neighborhoods (Part A — geometry)

Corpus A train+val embedded with frozen P1 (`p1_best.pt`). For each of 80 random probes, take \(k\) nearest neighbors in continuous flat-\(S\) (cosine) vs \(k\) nearest in ECFP (Tanimoto). Metrics: Murcko entropy \(H\), BRICS entropy, behavioral variance \(v_{\mathrm{beh}}\) (mean variance of z-scored surrogates).

| \(k\) | mean \(H\) (\(S\)) | mean \(H\) (ECFP) | mean \(v_{\mathrm{beh}}\) (\(S\)) | mean \(v_{\mathrm{beh}}\) (ECFP) | frac \(v_S < v_{\mathrm{ECFP}}\) | mean \(\Delta R\) |
|------|--------------------|-------------------|-----------------------------------|----------------------------------|-------------------------------|-------------------|
| 32 | 3.40 | 3.35 | **0.243** | 0.502 | **0.99** | +18.1 |
| 64 | 4.09 | 4.03 | **0.277** | 0.513 | **0.99** | +16.8 |

Specificity summary statistic (secondary):

\[
R = \frac{H_{\mathrm{Murcko}} + H_{\mathrm{BRICS}}}{v_{\mathrm{beh}} + \varepsilon}
\]

At \(k=64\): mean \(R_S \approx 34.0\) vs \(R_{\mathrm{ECFP}} \approx 17.2\) (≈99% of probes have \(R_S > R_{\mathrm{ECFP}}\)).

**Files:** `soft_neighborhoods_k{32,64}.csv`, `specificity_summary.json`, `embeddings.npz`

### 2.2 Exact discrete VQ codes (context, not the claim)

| | |
|--|--|
| Molecules | 7331 |
| Unique 12-slot codes | 7287 |
| Codes with size ≥ 2 | 42 |
| Max bucket size | 3 |

Exact code equality is nearly injective. Analysis uses **soft neighborhoods** in continuous \(S\) (and k-means partitions). This is expected for a high-capacity discrete code — analogous to unique sentences in a language corpus.

### 2.3 k-means partitions of \(S\) (supporting)

80 clusters (min size 16), same metrics vs equal-sized ECFP neighborhoods of cluster medoids: again ≈99% of clusters have higher \(R\) for \(S\), lower mean \(v_{\mathrm{beh}}\) (0.31 vs higher ECFP). See `kmeans_partitions.csv`.

### 2.4 Generative E3 (supporting / heterogeneous specificity)

**FiLM Arm A** (`runs/p2_selfies_cond/`, per-step FiLM + cond dropout 0.1): 5 frozen \(S^*\), 500 samples each. Pass rate **3/5**. Quantify worlds: `C2, C1, C2, OK, C2` (`runs/p2_selfies_cond/e3/e3_quantify.json`). One seed remains C1 (behavior explode); not motif-collapse overall.

Earlier non-FiLM autopsy (`runs/p2_selfies/e3/`) showed the same qualitative split (tight vs underspecified codes). Specificity is continuous across latent regions.

### 2.5 Earlier gates (context)

| Exp | Result | Role in paper |
|-----|--------|---------------|
| E2 | Pass (gap≈0.28, structure margin vs FP-PCA) | \(S\) is not a fingerprint bottleneck |
| E3 (FiLM) | Soft 3/5; worlds mostly C2/OK, one C1 | Neighborhoods are *generative*; conditioning not perfect |
| E9 (strict) | 1/5 (solubility only) | Do **not** claim universal frozen transfer |

---

### 2.6 Spec as generative interface — G4 / G5 (Part B — control)

Pipeline: `DesignObjective → FAN (bank retrieve/rank/compose) → FunctionalSpecification → FiLM Arm A`. Filter off for ladder statistics below unless noted; see `runs/gen/eval/gen_eval_summary.json`.

#### G4 — LogP ladder (headline generative result)

Target LogP \(\in \{-1,0,1,2,3,4,5\}\) → FAN Spec → sample → realized LogP of accepted molecules.

\[
\rho(\text{target LogP},\;\text{realized mean LogP}) = \mathbf{0.990}
\]

| Target LogP | Realized mean ± std | \(n\) accepted |
|-------------|---------------------|----------------|
| −1 | 0.20 ± 0.88 | 35 |
| 0 | 0.98 ± 0.90 | 38 |
| 1 | 1.61 ± 0.83 | 36 |
| 2 | 1.93 ± 0.72 | 28 |
| 3 | 2.48 ± 0.90 | 35 |
| 4 | 3.13 ± 0.98 | 37 |
| 5 | 4.18 ± 0.84 | 34 |

Monotonic and controllable; absolute calibration has offset (targets slightly undershoot at the high end). This is **behavior navigation through \(S\)**, not incidental property drift from motif clusters.

#### G5 — Repeatability (one Spec → many chemistries)

Fixed Spec for `LogP=2.5`, four independent batches (\(n=48\) each):

| Batch mean LogP |
|-----------------|
| 2.29, 2.50, 2.26, 2.15 |

- Behavior stability: **std of batch means = 0.123**
- Union chemistry: **136** unique molecules, **114** scaffolds, Murcko \(H = 4.44\)

Central thesis in one ratio: **low behavior variance across repeats, high structural diversity under one Spec.**

#### G1–G3 (supporting / open)

| ID | Result | Note |
|----|--------|------|
| G1 | mean match ≈ 0.31 | Objective hit rate among accepted; room to improve calibration |
| G2 | mean Murcko \(H\) ≈ 3.42 | Diversity among accepted sets |
| G3 | \(\rho(\mathrm{confidence},\;\mathrm{acceptance}) \approx -0.47\) | Encoder neighborhood \(R\) does **not** predict generation reliability — interesting, not MVP-blocking |

---

### 2.7 Ablations — is \(S\) necessary? G6b / G6a (Part C)

#### G6b — Decoder uses Spec payload (no retrain)

Hold FAN Spec fixed; mutate only `flat_S`; sample with filter off (`runs/gen/g6b/`).

| Condition | LogP \(\rho\) | hit ±0.5 | mean Murcko \(H\) |
|-----------|---------------|----------|-------------------|
| true \(S\) | **0.997** | **0.273** | **3.88** |
| scramble dims | −0.226 | 0.104 | 1.82 |
| random (same \(\|S\|_2\)) | −0.133 | 0.091 | 1.72 |

Control collapses when the payload is destroyed → the FiLM decoder is not ignoring \(S\).

#### G6a — Matched \(S\) vs property vector vs random \(S\)

Capacity-matched property FiLM decoder (`runs/p2_property/`, \(y\to\mathrm{cond\_dim}\) proj + same GRU recipe, 15 epochs). Eval: LogP ladder, filter off (`runs/gen/g6a/`). Property arm conditions on LogP z-score only (other dims zeroed).

| Model | LogP \(\rho\) | hit ±0.5 | mean Murcko \(H\) | mean scaffolds |
|-------|---------------|----------|-------------------|----------------|
| true \(S\) | **0.985** | **0.281** | **3.92** | **56.5** |
| property \(y\) | 0.906 | 0.190 | 1.95 | 25.8 |
| random \(S\) | −0.486 | 0.094 | 2.56 | 36.8 |

Descriptors give **some** control (“high-ish” \(\rho\)). \(S\) matches or beats control and roughly **doubles** structural diversity. That answers: *why a functional latent, not just condition on LogP?*

Property baseline train fit is weaker (val PPL ~8.8 vs ~2.7 for Spec Arm A) under the same recipe — consistent with \(S\) being a better conditioning language for chemistry.

---

## 3. Safe claim vs overclaim

### Say this

> The learned Functional Specification \(S\) organizes chemical space into structure-diverse, behavior-coherent neighborhoods relative to fingerprints, and serves as a **controllable latent interface**: design objectives map through \(S\) to diverse chemical realizations with predictable surrogate behavior. Matched descriptor conditioning achieves partial property control with substantially lower scaffold diversity; scrambling or randomizing \(S\) destroys control.

### Do not say

- “Universal molecular language”
- “Functional Spec replaces fingerprints on all tasks” (E9 strict 1/5)
- “\(S\) is compositionally editable via single-token interventions” (still open)
- “Encoder specificity \(R\) predicts generation reliability” (G3 fails — different concept)
- “Perfect LogP calibration” (ladder is monotonic with offset)
- Phase II claims (environment → Interaction Spec) — **separate paper**

### Softened / optional

- “Not a pure descriptor cluster” — G6a is the right ablation class; residualization of encoder \(v_{\mathrm{beh}}\) on LogP/TPSA/MW remains a useful encoder-side control for Part A.

---

## 4. How this becomes a paper

### Working title options

1. *Behavioral Geometry of Chemical Space: From Latent Neighborhoods to a Generative Design Interface*
2. *Functional Specifications: Controllable Behavioral Latents for Diverse Molecular Realization*
3. *When Structure Is Diverse and Behavior Is Shared: Soft Equivalence Classes as a Generative Interface*

### Narrative spine (three acts)

1. **Encoder learns a behavioral geometry**  
   Soft \(S\)-neighborhoods vs ECFP: \(H\) matched, \(v_{\mathrm{beh}}\) down. Figure = `fig_behavioral_geometry`.

2. **\(S\) is a generative control interface**  
   FAN + FiLM Arm A: G4 LogP ladder (\(\rho\approx 0.99\)); G5 one Spec → many scaffolds, stable behavior.

3. **Why \(S\), not descriptors alone**  
   G6b (payload necessary); G6a (\(S\) ≥ control, ≫ diversity vs matched property decoder). Honest E9 as scope limit on frozen transfer.

### Suggested structure

| Section | Content |
|---------|---------|
| Intro | Design intent vs realization; fingerprints organize by structure; need a *controllable behavioral interface* |
| Method | P1 slots+VQ; FAN_MVP (objective→Spec); FiLM Arm A; soft neighborhood + G4–G6 protocols |
| Result 1 | Behavioral geometry figure + stats |
| Result 2 | G4 ladder + G5 repeatability (+ E3 heterogeneity) |
| Result 3 | G6b / G6a ablations |
| Result 4 | E2 probes; honest E9 as negative / scope limit |
| Discussion | Interface ≠ universal embedding; encoder \(R\) ≠ decoder reliability (G3); Phase II env path deferred |
| Limitations | Surrogate proxies; Corpus A; residual C1 seeds; absolute calibration |

### Central figures

1. **`fig_behavioral_geometry`** (exists) — \(H\) vs \(v_{\mathrm{beh}}\), \(S\) vs ECFP  
2. **`fig_logp_ladder`** (exists) — target vs realized LogP for true \(S\) / property \(y\) / random / scramble; diversity bars  
   - `runs/gen/fig_logp_ladder.{png,pdf}` · regenerate: `uv run python scripts/plot_logp_ladder.py`  
3. Optional: G5 panel — batch LogP means + example scaffolds under one Spec

### Venue fit

Still natural as **representation + design interface** (not “another generative model paper”). Generation and ablations support the geometry/interface claim.

---

## 5. Minimal remaining experiments for a submission

Priority order (narrow; do not redesign):

1. ~~**G4/G6 comparison figure**~~ — done: `runs/gen/fig_logp_ladder.{png,pdf}`  
2. **MVP ablation suite (Tier 1–2)** — codebook `|C|`, continuous-S, contrastive/`w_ecfp`, slots, \(\beta\), surrogates, cond dropout, FAN compose, residualized \(v_{\mathrm{beh}}\) — see §6  
3. **Held-out chemistry** — neighborhood protocol on scaffold-split / external set  
4. Optional: C1 autopsy on the remaining E3 failure seed; token intervention for editability  

Do **not** block writing on G3 confidence repair, Phase II environment encoder, or Arm B.

---

## 6. How to regenerate

### MVP ablation suite (primary)

```bash
cd FunctionalSpec

# Smoke (wiring)
uv run python scripts/run_mvp_ablations.py \
  --phase all --epochs 3 --recon-epochs 3 --n-probes 40 --n-gen 16 \
  --out runs/ablations_smoke

# Full paper suite (overnight+)
uv run python scripts/run_mvp_ablations.py --phase all --out runs/ablations

# Phase-by-phase (resumable)
uv run python scripts/run_mvp_ablations.py --phase codebook --out runs/ablations
uv run python scripts/run_mvp_ablations.py --phase necessity --out runs/ablations
uv run python scripts/run_mvp_ablations.py --phase hps --out runs/ablations   # β ∈ {0.05…2}, K ∈ {4…24}
uv run python scripts/run_mvp_ablations.py --phase p2 --out runs/ablations
uv run python scripts/run_mvp_ablations.py --phase gen --out runs/ablations
uv run python scripts/run_mvp_ablations.py --phase summarize --out runs/ablations

# Codebook-only alias
uv run python scripts/eval_codebook_sweep.py --out runs/ablations --execute
```

Results: `runs/ablations/summary.json` (and per-cell `e2/`, `spec/`, `gen/`, …).

Figures: `uv run python scripts/plot_ablation_figures.py --root runs/ablations`
→ `runs/ablations/figures/`.

**Read the suite as two stories (do not mix):**

| Story | Question | Cells | Good outcome |
|-------|----------|-------|--------------|
| **A. Design necessity** | Was this architectural choice required? | no VQ, no contrastive, physchem-only surrogates, `w_ecfp=0`, FAN compose vs retrieve, G6b true/scramble/random | Ablated (or control) **fails** / collapses; full model **passes** |
| **B. Hyperparameters** | How sensitive are we to knobs? Which value to pick? | \|C\| ∈ {32…512}, β ∈ {0.05,0.1,0.25★,0.5,1,2}, K ∈ {4,6,8,12★,16,24}, cond-dropout | All (or most) **still pass**; pick best (★ = full-model default) |

- Story A is **justification** (“we needed VQ / contrast / rich surrogates / compose”).
- Story B is **selection + robustness** (“defaults aren’t brittle; \|C\|=512 is preferred”).
- A FAIL in A is a **positive** paper result. A FAIL in B would be worrying (brittle HPS).

Residualized Part A (also run for C128 inside codebook phase):

```bash
uv run python scripts/run_spec_specificity.py \
  --checkpoint runs/p1/p1_best.pt \
  --out runs/p1/spec_specificity \
  --k 32 64 128 --n-probes 200 \
  --residualize-logp-tpsa-mw
```

### Baseline regen (existing)

```bash
cd FunctionalSpec

# Part A — neighborhoods + figure
uv run python scripts/run_spec_specificity.py \
  --checkpoint runs/p1/p1_best.pt \
  --corpus-a data/processed/corpus_a \
  --out runs/p1/spec_specificity \
  --k 32 64 128 --n-probes 200

uv run python scripts/plot_behavioral_geometry.py \
  --dir runs/p1/spec_specificity --ks 32 64

# Part B — Spec bank, G1–G5
uv run python scripts/build_spec_bank.py \
  --p1-checkpoint runs/p1/p1_best.pt \
  --out runs/gen/spec_bank.npz

uv run python scripts/run_gen_eval.py \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --out runs/gen/eval --n 64

# Part C — ablations
uv run python scripts/run_g6b.py \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --out runs/gen/g6b --n 64

uv run python scripts/train_p2_property.py \
  --out runs/p2_property --epochs 15

uv run python scripts/run_g6a.py \
  --s-checkpoint runs/p2_selfies_cond/p2_best.pt \
  --y-checkpoint runs/p2_property/p2_property_best.pt \
  --out runs/gen/g6a --n 64
```

Interactive canvas (IDE): `behavioral-geometry.canvas.tsx`.

---

## 7. Takeaway for the project

The MVP did not prove a universal frozen “language of chemistry.” The defensible contribution is sharper and now supported on both encoder and generator sides:

> **A surrogate-trained discrete bottleneck induces a behavioral geometry of chemical space and a controllable Spec interface** — soft equivalence classes that are structure-diverse and behavior-coherent relative to fingerprints, and a generative path where one Spec yields many chemistries with predictable surrogate behavior, beyond matched descriptor conditioning.

Frozen transfer (E9) and environment-conditioned Interaction Specs remain future chapters. Writing can proceed on Parts A–C above.
