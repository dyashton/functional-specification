# Metric Card — Operational Pass/Fail Thresholds

All thresholds are **MVP go/no-go** gates. Tune only with a frozen protocol (document any change).

---

## Notation

- \(S\): Functional Specification (slot + VQ codes)
- Recon-VQ: structure autoencoder control
- Surrogate vector \(y\): z-scored physchem/QM/IE proxies used in training
- Structure probes: motif-ID accuracy, ECFP bit AUROC / R², scaffold top-1

---

## E0 — Sanity

| Metric | Pass | Fail |
|--------|------|------|
| Validity (RDKit parse + sanitize) | ≥ 0.80 of samples | < 0.50 or mode-collapse to ≤3 unique SMILES |
| Unique@1000 | ≥ 0.40 | < 0.10 |

---

## E1 — Surrogate sufficiency

| Metric | Pass | Fail |
|--------|------|------|
| E1a: mean Spearman ρ (held-out surrogates from \(S\)) | ≥ 0.70 × fingerprint→property Spearman | < 0.40 absolute or ≪ FP baseline |
| E1b: IE Spearman on Corpus B test (after A-only or light B FT) | > 0.25 and better than shuffled-\(S\) | ≤ chance |

---

## E2 — Surrogate vs structure probes **(publishability gate)**

Compute on held-out scaffolds.

| Metric | Pass | Fail |
|--------|------|------|
| Surrogate probe score \(P_f\) (mean Spearman / AUROC on surrogates) | High | — |
| Structure probe score \(P_s\) (motif top-1, ECFP R², scaffold Acc) | Low vs Recon-VQ | — |
| Gap \(\Delta = P_f - P_s\) (after min-max normalizing each family to [0,1] within the experiment) | \(\Delta \ge 0.25\) **and** \(P_s(\text{ours}) \le P_s(\text{Recon-VQ}) - 0.15\) | \(P_s\) within 0.05 of Recon-VQ |
| NN test: mean Tanimoto of \(S\)-NN vs FP-NN | \(T(S\text{-NN}) \le T(\text{FP-NN}) - 0.15\) among strong/interesting sets | \(S\)-NN ≈ FP-NN |

**Fail ⇒ “just a latent bottleneck.”**

---

## E3 — One \(S\) → many structures / one behavior **(primary generative)**

Protocol: 1000 samples per \(S^*\), ≥5 specs; dedupe.

### Operational definitions (fixed)

\[
D_{\mathrm{struct}} = 1 - \overline{T}_{\mathrm{tanimoto}}
\]

(mean pairwise ECFP4 Tanimoto among unique valid molecules). Also **report** Murcko scaffold count \(N_{\mathrm{scaf}}\) and BRICS/learned-motif entropy \(H_{\mathrm{frag}}\).

\[
V_{\mathrm{beh}} = \frac{1}{d}\sum_{k=1}^{d} \mathrm{Var}(\hat{y}_k^{\mathrm{z}})
\]

where \(\hat{y}^{\mathrm{z}}\) are z-scored surrogate predictions (or oracle labels when available) on the sample set; \(d\) = number of surrogate dims used in matching.

\[
R = \frac{D_{\mathrm{struct}}}{V_{\mathrm{beh}} + \varepsilon},\quad \varepsilon=10^{-6}
\]

### Matched-behavior baseline

Filter/resample Recon-VQ (or property-conditional) generations until

\[
\| \bar{y}_{S^*} - \bar{y}_{\mathrm{base}} \|_2 / \sqrt{d} \le \tau,\quad \tau=0.15
\]

(config `eval.behavior_match_tol`). Compare \(D_{\mathrm{struct}}\), \(N_{\mathrm{scaf}}\), \(R\) only under match.

| Metric | Pass | Fail |
|--------|------|------|
| \(N_{\mathrm{scaf}}\) @1000 unique | ≥ 30 (median over \(S^*\)) | ≤ 5 or same core motif in ≥80% samples |
| \(V_{\mathrm{beh}}\) | ≤ 0.35 (z-space) | > 1.0 (behavior drifts) |
| \(R\) vs matched baseline | \(R \ge 1.25 \times R_{\mathrm{base}}\) **or** \(N_{\mathrm{scaf}} \ge 1.5 \times N_{\mathrm{scaf,base}}\) at match | No gain at matched behavior |
| IE mass (Corpus B–conditioned specs) | ≥ 40% samples in strong band by proxy/oracle | Indistinguishable from random GuacaMol |

---

## E4 — Motif arm

Same E0/E3 gates for Arm A vs Arm B. Prefer Arm A if B unstable. BRICS control must not be the sole winner attributed to the planner.

---

## E5 — Controllability

| Metric | Pass | Fail |
|--------|------|------|
| Success@k (IE/PFC threshold) at matched diversity | ≥ direct conditional | Dominated on both success and diversity |

---

## E6 — Token intervention

| Metric | Pass | Fail |
|--------|------|------|
| Max \|Δmean\| on any surrogate dim (z-scored) when removing token \(t\) | ≥ 0.4 on ≥1 dim, consistent across 3 seeds | All \|Δ\| < 0.1 or only scaffold identity changes |

---

## E7 — Token atlas (descriptive)

Distinct mean surrogate vectors across frequent tokens (pairwise cosine of mean-\(y\) ≤ 0.85 for ≥50% token pairs). No hard kill; supports E6.

---

## E8 — \(S\)-interpolation

| Metric | Pass | Fail |
|--------|------|------|
| Validity along path (10 steps) | ≥ 0.70 mean | < 0.40 |
| Surrogate path smoothness (mean jump / total variation) | gradual | Abrupt structural mode flip with flat/noisy behavior |

---

## E9 — Frozen \(S\) → unseen tasks **(primary transfer)**

Linear / 1-hidden MLP heads; identical budget for ECFP, graph emb, Recon-VQ, continuous latent.

| Metric | Pass | Fail |
|--------|------|------|
| Low-\(n\) curve (n ∈ {32,64,128,256}) + full-data | Task win = (majority finite-\(n\) rank ≤ ECFP) **and** full-data score ≥ ECFP − 0.02; overall ≥3/5 task wins | Soft rank wins without full-data competitiveness; loses to ECFP on ≥4/5 |

Document train↔held-out label correlations; flag any \(|\rho|>0.7\) pair as leakage risk.

---

## Codebook sweep

Report E2 \(\Delta\), E3 \(R\) (+ matched), E9 low-\(n\), validity for \(|C|\in\{32,64,128,256,512\}\). Primary = best val E3 \(R\) subject to E2 pass.
