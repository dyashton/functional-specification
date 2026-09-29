# Phase II — Planning Architecture

Separate paper from behavioral geometry. Generator contract unchanged.

## Three concepts (do not mix)

| Concept | Role | Module |
|---------|------|--------|
| **InteractionEnvironment** | External physics (atom/interaction graph) | `functionalspec/env/` |
| **DesignObjective** | User wants (LogP, ADMET, …) — not physics | `functionalspec/gen/objective.py` |
| **FunctionalSpecification** | Molecule-centric requirements | `functionalspec/gen/spec.py` |

```text
InteractionEnvironment ──► EnvEncoder ──► InteractionRepresentation ─┐
                                                                      ├──► ContextFAN ──► Spec ──► frozen Generator
DesignObjective (optional) ───────────────────────────────────────────┘
```

EnvEncoder **never** sees DesignObjective fields.

## First slice (CO₂)

```bash
# train ContextFAN (poses from CO2_IE_Dataset runs/*/complexes)
uv run python scripts/train_phase2_fan.py \
  --p1-checkpoint runs/p1/p1_best.pt \
  --bank runs/gen/spec_bank.npz \
  --co2-root ../CO2_IE_Dataset \
  --out runs/phase2_co2 \
  --epochs 15

# I1–I5
uv run python scripts/run_phase2_eval.py \
  --fan runs/phase2_co2/fan_context.pt \
  --generator runs/p2_selfies_cond/p2_best.pt \
  --out runs/phase2_co2/eval
```

Env graph MVP: host atoms near CO₂ + CO₂ (distance spatial edges + covalent). See `env/graph_co2.py`.

## II.4 — Enrich later (pipeline unchanged)

When extending beyond the first paper:

1. **Richer node features** on the same graph: partial charge, SASA, electrostatic potential, flexibility — append dims or replace placeholders in `NODE_DIM` slots; retrain EnvEncoder only.
2. **Heterogeneous interaction graph**: protein atoms, waters, ions, metals, pseudo-nodes; edge types distance / H-bond / vdW / covalent — still `InteractionEnvironment` tensors.
3. **New env types**: pocket, MOF pore, membrane — new builders under `env/`, same `EnvironmentEncoder` / `ContextFAN` APIs.
4. **Richer DesignObjective**: ADMET, MoA strings — still passed only into FAN, never EnvEncoder.
5. **Pluggable generators**: any Spec consumer; no planner retrain.

Do **not** fold synthesizability / toxicity / LogP into InteractionEnvironment.

## API

```python
from functionalspec.env import co2_only_environment, environment_from_complex_xyz
from functionalspec.gen.fan_context import ContextFAN
from functionalspec.gen.objective import DesignObjective

fan = ContextFAN.load("runs/phase2_co2/fan_context.pt", bank)
env = environment_from_complex_xyz(path)  # or co2_only_environment()
specs = fan.plan(DesignObjective(), environment=env, n_strategies=3)
```
