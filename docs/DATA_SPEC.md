# Data Specification — Functional Specification MVP

## Two-corpus design

| Corpus | Role | Target size | Labels |
|--------|------|-------------|--------|
| **A** | Primary train (P1–P3) | ≥50k diverse druglike | Surrogate functional views |
| **B** | Shared-env transfer / E3 relevance | ~486 gold IE hosts | Psi4 CO₂ IE |

Never train the planner on Corpus B alone. Never train Corpus A without surrogate heads.

---

## Corpus A — scale + diversity

### Paper target
GuacaMol or ZINC-filtered druglike molecules, scaffold-split 80/10/10, seed 42.

### Bootstrap (in-repo)
[`Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv`](../../Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv) (~20k).

Build with:

```bash
uv run python scripts/prepare_corpus_a.py \
  --input ../Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv \
  --out data/processed/corpus_a
```

### Surrogate functional views (train heads; imperfect Function proxies)

| View | Source | Notes |
|------|--------|-------|
| MW, LogP, TPSA, QED, HBA, HBD, nRot | RDKit | Always computable |
| HallKierAlpha, apol, basicity, PFC_composite | Bridge CSV / RDKit+workflow | Prefer CSV when present |
| dipole, HOMO, LUMO, gap | QM9 / external (optional subset) | Cleaner Function≠structure signal |

**Paper language:** these are *surrogate functional views*, not mechanisms.

### Structure views (E2 probes only — never planner train targets)
- ECFP4 (2048 bits)
- Murcko scaffold SMILES
- BRICS fragment multiset (probe / control only)
- Learned motif IDs (if Arm B trained)

### Scaffold split
Murcko scaffold hashing → stratified assignment. Molecules with rare scaffolds go to train preferentially when needed to fill quotas.

---

## Corpus B — CO₂ shared design problem

Source: [`CO2_IE_Dataset/data/compiled.csv`](../../CO2_IE_Dataset/data/compiled.csv)

| Band | Criterion (kcal/mol) | Use |
|------|----------------------|-----|
| Strong | `EI ≤ -6.0` | Contrastive positives, E3 relevance |
| Mid | `-6.0 < EI < -3.0` | Soft / ignore for hard contrastive |
| Weak | `EI ≥ -3.0` | Contrastive negatives |

```bash
uv run python scripts/prepare_corpus_b.py \
  --input ../CO2_IE_Dataset/data/compiled.csv \
  --out data/processed/corpus_b
```

Outputs: `molecules.parquet`, `splits.json`, `bands.json`, `strong_weak_pairs.json`.

Bridge physchem (same chemistry distribution as A, CO₂-shaped proxies): scored GuacaMol tables under `Molecule_Generation_Workflow/publication/co2/data/`.

---

## E9 task suite — train vs held-out

### Planner train tasks (A–C style; freeze after)

| Task ID | Label | Suggested source |
|---------|-------|------------------|
| `homo` | HOMO energy | QM9 / MoleculeNet |
| `lumo` | LUMO energy | QM9 |
| `dipole` | Dipole moment | QM9 |
| `co2_ie` | Host–CO₂ IE | Corpus B (light fine-tune only) |
| `pfc` | PFC composite | Bridge / workflow |

Exact set locked in [`configs/default.yaml`](../configs/default.yaml) as `e9_train_tasks`.

### Held-out eval tasks (never used in planner training)

| Task ID | Label | Suggested source |
|---------|-------|------------------|
| `solubility` | Aqueous solubility | ESOL / AqSolDB |
| `toxicity` | ClinTox / Tox21 proxy | MoleculeNet ClinTox |
| `permeability` | Membrane permeability | PAMPA / Caco-2 public sets |
| `heldout_binding` | Distinct bioassay | ChEMBL single-assay holdout |
| `flexibility` | Conformational flexibility | nRot + related RDKit proxies (if nRot was *not* a train head; else use radius-of-gyration / fraction rotatable from 3D) |

**Prepared locally via** `scripts/prepare_e9.py` (no DeepChem install):

| Task | Concrete source written to `data/processed/e9/{task}.csv` |
|------|----------------------------------------------------------|
| solubility | MoleculeNet ESOL (Delaney measured logS) |
| toxicity | MoleculeNet ClinTox (`CT_TOX`) |
| permeability | MoleculeNet Lipophilicity (exp logD; membrane-partition **proxy**) |
| heldout_binding | MoleculeNet BACE (pIC50) |
| flexibility | RDKit `FractionCSP3` (nRot avoided — used in P1 heads) |

Document pairwise correlations between train and held-out labels in `data/processed/e9/correlations/` after download so “unseen” is not leaked. Expect solubility↔LogP \(|\rho|>0.7\); flag in `e9_prepare_meta.json`.

Loaders: `functionalspec/data/e9_tasks.py` + `prepare_e9.py`. Runner: `scripts/run_e9.py`.

---

## Multi-view contrastive weights

Default (see config):

```text
sim_+ ∝ w_desc·sim_desc + w_qm·sim_qm + w_bind·sim_bind
         + w_interaction·sim_interaction − w_ecfp·sim_ECFP
```

Missing view ⇒ weight 0. Prefer pairs close in surrogate space and far in fingerprint space. Corpus B: strong–strong positives; hard negatives = high ECFP + weak IE.

---

## Explicit non-goals

- Train only on B (~486)
- Corpus A without surrogate heads
- MMP lead→optimized reconstruction as planner objective
- Calling MVP tokens “interaction tokens”
