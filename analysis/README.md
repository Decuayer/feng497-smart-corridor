# analysis/: Queueing, Webster, BPR/DTA and economics

- **Owner:** Ayça Avcı (IE)
- **Status:** planned

## Purpose

Analytical and operations-research foundations and evaluation:

- queueing models (M/M/1, M/M/c, M/G/1) for the intersection approaches;
- the **Webster** optimal cycle and green splits as a baseline controller;
- the BPR link-delay function and dynamic traffic assignment (DTA) formulation;
- fuel consumption and CO₂ estimation (KPI: 185 → 148 g/km);
- CAPEX/OPEX analysis.

## Scope

**Belongs here:** model derivations, parameter estimation, notebooks, reproducible analysis scripts, economic models.

**Does not belong here:** the runtime routing service (`backend/`), RL controllers (`ai/`), SUMO scenario files (`simulation/`).

## Planned sub-structure

```
analysis/
├── queueing/    # M/M/1, M/M/c, M/G/1 models and validation
├── webster/     # Webster cycle/split computation, parameters per intersection
├── bpr_dta/     # BPR parameters, DTA formulation
├── economics/   # fuel/CO₂ and CAPEX/OPEX
├── notebooks/   # exploratory notebooks (clear outputs before committing)
└── tests/
```

## Interfaces

- **Produces (requested by `ai/`):** Webster parameters per intersection (saturation flows, lost time, critical flow ratios, resulting cycle and splits) in a machine-readable file. Until this exists, `ai/` computes a simple local Webster plan marked ASSUMPTION. BPR parameters for the routing service (`backend/`). CO₂ factors for KPI evaluation.
- **Consumes:** IZUM data ([`data/README.md`](../data/README.md)), simulation outputs.

## Setup, run, test

To be added. Notebooks must not contain raw data or credentials. Clear outputs before committing.
