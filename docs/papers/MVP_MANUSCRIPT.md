# Behavioral Geometry of Chemical Space: From Latent Neighborhoods to a Generative Design Interface

**Working title (alt):** Functional Specifications: Controllable Behavioral Latents for Diverse Molecular Realization  
**Status:** MVP manuscript draft (extracted 2026-09-16) — evidence-complete for Parts A–C; optional probe CIs / leakage controls deferred  
**Living evidence note:** [`../BEHAVIORAL_GEOMETRY_PAPER.md`](../BEHAVIORAL_GEOMETRY_PAPER.md)  
**Not this paper:** Phase II environment → Interaction Spec ([`../PHASE2.md`](../PHASE2.md))

---

## Claims lock

### We claim

1. Soft neighborhoods in the learned Functional Specification \(S\) are **structure-diverse** and **behavior-coherent** relative to ECFP neighborhoods of matched size (Part A).
2. \(S\) is a **controllable generative interface**: design objectives map through on-manifold Specs to molecules whose surrogate behavior moves predictably (LogP ladder \(\rho \approx 0.99\); Part B).
3. One Spec yields **many chemistries** at stable behavior across repeats (Part B / G5).
4. The FiLM generator **depends on the Spec payload** (scramble / random \(S\) destroy control; Part C / G6b).
5. Relative to a **capacity-matched property-conditional** decoder, \(S\) matches or beats property control and roughly **doubles** scaffold diversity (Part C / G6a).
6. \(S\) is **not** merely a fingerprint bottleneck (E2 pass).

### We do not claim

- A universal molecular language or frozen transfer to all downstream tasks (E9 strict: 1/5).
- Perfect absolute LogP calibration (ladder is monotonic with offset).
- That encoder neighborhood specificity \(R\) predicts generation reliability (G3 fails).
- Compositional editability via single-token interventions.
- Environment-conditioned planning / Interaction Specs (Phase II).
- That Arm A is SOTA generative chemistry; generation supports the *interface* claim.

---

## Abstract

Fingerprints organize chemical space by structure. Design problems need a different object: a reusable conditioning interface that states *what a molecule should do*, then admits many structural realizations. We learn a discrete Functional Specification \(S\) — a slot + vector-quantized bottleneck trained for surrogate-behavior prediction without molecule reconstruction as the planner objective. Soft neighborhoods in \(S\) match fingerprint neighborhoods in structural diversity while substantially reducing surrogate-behavior variance. Conditioning a frozen FiLM SELFIES generator on Specs produced by a bank-based Functional Abstraction Network (FAN) yields controllable LogP navigation (\(\rho \approx 0.99\)) and high scaffold diversity under a fixed Spec. Ablations show the decoder uses the Spec payload, and that matched descriptor conditioning achieves weaker diversity at similar or worse control. We position \(S\) as a behavioral design interface, not a universal embedding.

---

## 1. Introduction

Molecular design separates **intent** (desired behavior) from **realization** (concrete chemistry). Standard descriptors and fingerprints excel at structural similarity search, but they are awkward as *generative programs*: conditioning only on LogP (or a small descriptor vector) under-specifies chemistry and often collapses diversity, while free optimization in continuous latents drifts off the training manifold.

We study a middle object — a **Functional Specification** \(S\) — intended to answer: *what functional requirements must this molecule satisfy?* rather than *which molecule should I make?* The MVP stack is:

```text
DesignObjective  →  FAN (on-manifold Spec bank)  →  FunctionalSpecification  →  Generator  →  Molecules
```

**Contribution.** We show that a surrogate-trained discrete bottleneck induces (i) a **behavioral geometry** of chemical space and (ii) a **controllable Spec interface** for diverse molecular realization, beyond matched descriptor conditioning. Environment-derived Specs are out of scope (Phase II).

---

## 2. Method (MVP)

### 2.1 Encoder and Functional Specification

Molecules are encoded with a GINE graph network into \(K=12\) learned slots, quantized with a VQ codebook (primary size 128). Surrogate heads predict multi-view physicochemical properties; a multi-view contrastive term encourages behavioral similarity without collapsing to ECFP geometry. Importantly, **molecule reconstruction is not** the planner training objective for \(S\).

Payload exposed to generation today: continuous `flat_S` (flattened slot embeddings) packaged as `FunctionalSpecification`. Discrete codes remain available for analysis; exact code equality is nearly injective on Corpus A, so geometry claims use **soft neighborhoods** in continuous \(S\).

### 2.2 FAN (MVP)

A Spec bank stores on-manifold `(flat_S, surrogate targets, density/R metadata)` from the training corpus. Given a `DesignObjective` (e.g. target LogP), FAN **retrieves → ranks → optionally composes** bank Specs. Outcomes are ternary: `exists` | `uncertain` | `unsupported`. Free off-manifold Adam in \(S\) is rejected in the MVP.

### 2.3 Generator (Arm A)

A SELFIES GRU is conditioned on `flat_S` via **per-step FiLM**, trained with light conditional dropout, encoder frozen (P2). The generator never sees raw design objectives — only Spec. Arm B (motif expansion) is deferred.

### 2.4 Evaluation protocols

| Part | Question | Protocol |
|------|----------|----------|
| A | Does \(S\) organize behavior vs structure? | Soft \(k\)-NN in \(S\) vs ECFP; Murcko/BRICS entropy \(H\), behavioral variance \(v_{\mathrm{beh}}\) |
| B | Is Spec a control interface? | FAN objectives → sample; G4 LogP ladder; G5 repeat batches |
| C | Is \(S\) necessary vs descriptors? | G6b payload scramble/random; G6a matched property FiLM |

Artifacts: `runs/p1/spec_specificity/`, `runs/gen/eval/`, `runs/gen/g6{a,b}/`, figures below.

---

## 3. Results

### 3.1 Behavioral geometry (Part A)

Corpus A embeddings under frozen P1. For 80 probes, \(k \in \{32,64\}\) neighbors in flat-\(S\) (cosine) vs ECFP (Tanimoto):

| \(k\) | mean \(H\) (\(S\)) | mean \(H\) (ECFP) | mean \(v_{\mathrm{beh}}\) (\(S\)) | mean \(v_{\mathrm{beh}}\) (ECFP) | frac \(v_S < v_{\mathrm{ECFP}}\) |
|------|--------------------|-------------------|-----------------------------------|----------------------------------|-------------------------------|
| 32 | 3.40 | 3.35 | **0.243** | 0.502 | **0.99** |
| 64 | 4.09 | 4.03 | **0.277** | 0.513 | **0.99** |

Structural diversity is matched; behavior is tighter under \(S\). Specificity \(R = (H_{\mathrm{Murcko}}+H_{\mathrm{BRICS}})/(v_{\mathrm{beh}}+\varepsilon)\) roughly doubles vs ECFP at \(k=64\). k-means partitions of \(S\) reproduce the same qualitative gap.

**Figure 1.** `runs/p1/spec_specificity/fig_behavioral_geometry.{png,pdf}`

**Supporting.** E2: \(S\) passes structure-vs-behavior probes vs FP-PCA (gap ≈ 0.28) — not a fingerprint bottleneck. Exact VQ codes are nearly unique (7287 / 7331); soft geometry is the right object.

### 3.2 Spec as generative control interface (Part B)

Pipeline: objective → FAN → FiLM Arm A (`runs/p2_selfies_cond/`).

**G4 — LogP ladder.** Targets \(\{-1,\ldots,5\}\) → realized mean LogP of accepted molecules:

| Target | −1 | 0 | 1 | 2 | 3 | 4 | 5 |
|--------|----|---|---|---|---|---|---|
| Realized mean | 0.20 | 0.98 | 1.61 | 1.93 | 2.48 | 3.13 | 4.18 |

\[
\rho(\text{target},\;\text{realized mean}) = 0.990
\]

Monotonic control with absolute offset (high targets undershoot). This is behavior navigation through \(S\), not incidental motif drift.

**G5 — One Spec, many chemistries.** Fixed Spec for `LogP=2.5`, four batches (\(n=48\)): batch-mean LogP std **0.123**; union **136** unique molecules, **114** scaffolds, Murcko \(H=4.44\).

**Figure 2.** `runs/gen/fig_logp_ladder.{png,pdf}` (true \(S\) / property \(y\) / scramble / random).

**Honesty.** E3 (FiLM): 3/5 frozen codes pass cycle-style checks; one C1 seed remains. G1 hit rates among accepted ≈ 0.3; G3: FAN confidence does **not** track acceptance (\(\rho \approx -0.47\)).

### 3.3 Ablations — why \(S\), not descriptors alone (Part C)

**G6b — payload necessity** (fixed FAN Spec; mutate only `flat_S`; filter off):

| Condition | LogP \(\rho\) | hit ±0.5 | mean Murcko \(H\) |
|-----------|---------------|----------|-------------------|
| true \(S\) | **0.997** | **0.273** | **3.88** |
| scramble dims | −0.226 | 0.104 | 1.82 |
| random \(\|S\|_2\) | −0.133 | 0.091 | 1.72 |

**G6a — matched property decoder** (LogP z-score FiLM, same GRU recipe):

| Model | LogP \(\rho\) | hit ±0.5 | mean Murcko \(H\) | mean scaffolds |
|-------|---------------|----------|-------------------|----------------|
| true \(S\) | **0.985** | **0.281** | **3.92** | **56.5** |
| property \(y\) | 0.906 | 0.190 | 1.95 | 25.8 |
| random \(S\) | −0.486 | 0.094 | 2.56 | 36.8 |

Descriptors give partial control; \(S\) preserves control **and** diversity. Property-arm val PPL is worse under the same recipe (~8.8 vs ~2.7), consistent with \(S\) as a better conditioning language for chemistry.

### 3.4 Scope limit (E9)

Frozen transfer under a strict multi-task protocol succeeds on **1/5** tasks (solubility only). We treat this as a **negative result / scope boundary**: MVP \(S\) is a design interface on surrogate-aligned chemistry, not a universal frozen representation.

---

## 4. Discussion

**Interface ≠ embedding.** The productive claim is that \(S\) is a *controllable behavioral program* for generation, with neighborhoods that trade structure for behavior coherence relative to fingerprints.

**Two notions of specificity.** Encoder neighborhood \(R\) and generation reliability are different concepts (G3). Do not collapse them in the narrative.

**Why not only condition on LogP?** G6a: matched \(y\)-conditioning recovers much of the ladder correlation but collapses scaffolds. The Spec carries a richer on-manifold chemical program.

**What this paper is not.** Environment encoders, pocket/CO₂ planning, Arm B, free-\(S\) optimisation, and “language of chemistry” rhetoric are deferred. Phase II asks whether FAN can infer Specs from InteractionEnvironments under fixed objectives — a separate scientific gate.

---

## 5. Limitations

- Surrogate physicochemical proxies, not wet-lab endpoints.
- Corpus A chemistry; held-out scaffold / external sets not yet the primary reported protocol.
- Absolute calibration offset on the LogP ladder; residual E3 C1 seed.
- Soft neighborhoods currently at \(n_{\mathrm{probes}}=80\); larger CIs and descriptor-residualized \(v_{\mathrm{beh}}\) are optional strengtheners, not blockers.
- Generator quality is evidence *for the interface*, not a claim of SOTA de novo design.

---

## 6. Conclusion

A surrogate-trained discrete bottleneck induces a behavioral geometry of chemical space and a controllable Functional Specification interface: objectives map through on-manifold Specs to diverse molecules with predictable surrogate behavior, beyond matched descriptor conditioning. That is the MVP contribution. Environment-derived Interaction Specs remain future work.

---

## Appendix A — Artifact map

| Role | Path |
|------|------|
| P1 checkpoint | `runs/p1/p1_best.pt` |
| Soft neighborhoods | `runs/p1/spec_specificity/` |
| Spec bank | `runs/gen/spec_bank.npz` |
| FiLM Arm A | `runs/p2_selfies_cond/p2_best.pt` |
| Property baseline | `runs/p2_property/` |
| G1–G5 summary | `runs/gen/eval/gen_eval_summary.json` |
| G6a / G6b | `runs/gen/g6a/`, `runs/gen/g6b/` |
| Figure 1 | `runs/p1/spec_specificity/fig_behavioral_geometry.{png,pdf}` |
| Figure 2 | `runs/gen/fig_logp_ladder.{png,pdf}` |
| Model card | `docs/MODEL_CARD.md` |
| Regen commands | `docs/BEHAVIORAL_GEOMETRY_PAPER.md` §6 |

## Appendix B — Suggested section → figure mapping

| Section | Figure | Path |
|---------|--------|------|
| Intro / Method | Fig 0 pipeline (**placeholder — replace**) | `runs/gen/paper_figs/fig_pipeline_placeholder.{png,pdf}` |
| §3.1 | Fig 1 behavioral geometry | `runs/p1/spec_specificity/fig_behavioral_geometry.{png,pdf}` |
| §3.1 | Fig 1b neighborhood cartoon | `runs/gen/paper_figs/fig_neighborhood_cartoon.{png,pdf}` |
| §3.2 G4 | Fig 2 LogP ladder | `runs/gen/fig_logp_ladder.{png,pdf}` |
| §3.2 G5 | Fig 3 G5 diversity | `runs/gen/paper_figs/fig_g5_diversity.{png,pdf}` |
| §3.3 | (reuse Fig 2 overlays) | `runs/gen/fig_logp_ladder.{png,pdf}` |
| §3.1 / E2 | Fig 4 E2 probes | `runs/gen/paper_figs/fig_e2.{png,pdf}` |

Regenerate extras: `uv run python scripts/plot_mvp_figures.py --out-dir runs/gen/paper_figs`

## Appendix C — Ablation suite commands

Two distinct stories (do not conflate in text or tables):

1. **Design necessity** — remove/alter a claimed ingredient (VQ, contrastive, surrogate set, FAN compose). Expect the ablated run to **fail** E2 / G4 relative to full. That *supports* the design.
2. **Hyperparameter search** — sweep \|C\|, β, K, cond-dropout. Expect **pass** E2 across a range; report the selected knobs (top-2 \|C\| = 512, 256).

```bash
cd FunctionalSpec
# Full Tier 1–2 suite → runs/ablations/summary.json
uv run python scripts/run_mvp_ablations.py --phase all --out runs/ablations
# Smoke
uv run python scripts/run_mvp_ablations.py --phase all --epochs 3 --recon-epochs 3 --n-probes 40 --n-gen 16 --out runs/ablations_smoke
# Figures (split scorecard A=design / B=HPS)
uv run python scripts/plot_ablation_figures.py --root runs/ablations
```

See `docs/BEHAVIORAL_GEOMETRY_PAPER.md` §6 for phase flags, residualization, and the A/B table.

## Appendix D — Writing checklist (do not reopen)

- [ ] Paste Abstract + Claims into overleaf/venue template
- [ ] Replace pipeline placeholder with designed schematic
- [ ] Import Figs 1–4; caption with \(\rho\) and diversity numbers
- [ ] One paragraph Method citing MODEL_CARD hyperparameters
- [ ] Explicit “do not claim” box or footnote (E9, Phase II, G3)
- [ ] Incorporate ablation suite table from `runs/ablations/summary.json` when ready
- [ ] Do **not** wait on Phase II, Arm B, G3 repair, or graph enrichment
