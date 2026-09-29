# Model Card — Functional Specification MVP

**Paper draft:** [`papers/MVP_MANUSCRIPT.md`](papers/MVP_MANUSCRIPT.md) · evidence note: [`BEHAVIORAL_GEOMETRY_PAPER.md`](BEHAVIORAL_GEOMETRY_PAPER.md)

## Claimed object

**Functional Specification / Molecular Program** \(S\): discrete slot + VQ bottleneck trained for surrogate-behavior prediction **without** molecule reconstruction on the planner. In the generation stack, \(S\) is exposed as `FunctionalSpecification` (payload today: `flat_S`).

Not claimed in MVP: Interaction Specification (requires Environment Encoder + FAN trained from environments — Phase II).

---

## Three-question stack

```text
1. Environment → Embedding     (Phase II; MVP uses DesignObjective instead)
2. FAN → FunctionalSpecification
3. Generator(Spec) → Molecule
```

**FAN** = Functional Abstraction Network (MVP: on-manifold bank retrieve → rank → compose). Generator never sees raw environment or raw design objectives — only Spec.

---

## Architecture

```text
Molecule graph M
    → GINE / Graph Transformer encoder
    → K slot embeddings (default K=12)
    → Vector-Quantized codebook |C| ∈ {32,64,128,256,512}
    → S = (slot codes, optional continuous residuals)
        ├─ Surrogate heads gφ(S)     [P1]
        ├─ Multi-view contrastive    [P1]
        ├─ Arm A: p(M | flat_S)      [P2]  FiLM per-step + cond dropout
        └─ Arm B: p(motifs | S)→atoms [P2] optional / deferred
```

### Encoder
- **Default:** GINE (`hidden_dim=256`, `num_layers=4`)
- Alt: Graph Transformer (same width)
- Node feats: atom type, degree, formal charge, aromaticity, hybridization
- Edge feats: bond type, conjugation, ring

### Slots + VQ
- \(K=12\) learned query slots (cross-attn or pooled projections onto molecule tokens)
- Codebook sizes swept: **32, 64, 128, 256, 512**
- Primary default: **128** (override by val E3 \(R\) subject to E2 pass)
- Commitment cost \(\beta=0.25\)
- EMA codebook updates optional

### Surrogate heads
MLP on flattened/mean slot embeddings → multi-task regression over Corpus A views (+ IE/PFC on B).

### Contrastive
InfoNCE on projected \(S\); positive weight uses multi-view sim **minus** ECFP sim (see `configs/default.yaml`).

### Generators

| Arm | Path | Status |
|-----|------|--------|
| **A** | Spec payload `flat_S` → SELFIES GRU with **per-step FiLM** (`film_per_step`, optional `cond_dropout`) | **Primary** — `runs/p2_selfies_cond/` |
| **B** | \(S\) → learned motif VQ/JT fragments → atom expansion | Deferred |

BRICS is **not** Arm B. BRICS may appear as E2 structure probe or deprecated control.

### Generation API

| Piece | Path |
|-------|------|
| `DesignObjective` | `functionalspec/gen/objective.py` |
| `FunctionalSpecification` / `FlatSPayload` | `functionalspec/gen/spec.py` |
| Spec bank | `functionalspec/gen/spec_bank.py` · `scripts/build_spec_bank.py` |
| FAN_MVP | `functionalspec/gen/fan.py` |
| `generate_from_spec` | `functionalspec/gen/generate.py` · `scripts/generate_from_spec.py` |
| G1–G5 | `functionalspec/eval/gen_eval.py` · `scripts/run_gen_eval.py` |

Outcomes: `exists` \| `uncertain` \| `unsupported`. Confidence is multi-factor; neighborhood specificity \(R\) is **internal** to FAN ranking.

### Cycle
Freeze \(S^*\), sample \(M'\), encode \(S'\); score vs **nearest real training \(S\)** + token-usage entropy (anti mean-collapse).

---

## Training schedule

1. **P1** — encoder + VQ + surrogate + contrastive only (no generation)
2. **P2** — freeze encoder; train Arm A (and B); teacher-forced \(S=\mathrm{sg}[f(M)]\)
3. **P3** — light joint + cycle; still **no** \(M\to S\to M\) recon as planner loss

---

## Baselines (capacity-matched where possible)

| ID | Role |
|----|------|
| Recon-VQ | Structure AE with same VQ size — E2 structure upper bound; E3 matched-behavior source |
| Continuous-\(z\) | Function heads, no discrete tokens |
| Direct property-conditional | Generator conditioned on surrogate/IE bin, no \(S\) |
| Matched-behavior structural | Recon-VQ/GAE filtered to \(\tau=0.15\) behavior match for E3 |
| TransPharmer+CO₂ | Generation quality floor (existing workflow) |
| E9 controls | ECFP, frozen GNN, Recon-VQ latent, continuous-\(z\) + same head |

---

## Parameter budget (targets)

| Module | Approx params |
|--------|----------------|
| Encoder | ~2–4M |
| VQ + slots | ~0.5M |
| Surrogate + proj | ~0.5M |
| Arm A decoder | ~5–15M |
| Recon-VQ baseline | match encoder+VQ+decoder |

Exact counts logged at train start (`scripts/count_params.py` optional).

---

## Code map

| Module | Path |
|--------|------|
| Config | `configs/default.yaml` |
| Encoder | `functionalspec/models/encoder.py` |
| VQ / slots | `functionalspec/models/vq.py` |
| Surrogate heads | `functionalspec/models/surrogate_heads.py` |
| Contrastive | `functionalspec/models/contrastive.py` |
| Arm A | `functionalspec/models/generator_a.py` |
| Arm B | `functionalspec/models/generator_b.py` |
| Baselines | `functionalspec/models/baselines.py` |
| Planner bundle | `functionalspec/models/planner.py` |
| FAN / Spec / generate | `functionalspec/gen/` |

---

## Non-goals

- Environment encoder (Phase II)
- Supervised token meanings (post-hoc interpretation only)
- Free-\(S\) optimization off-manifold (bank compose only in MVP)
- Free-form interaction graphs
- RL until E3/E5/E8/E9 pass
- Adversarial min \(I(S;\mathrm{Structure})\) as primary train loss
