# ADR 0001: Monorepo with one top-level folder per team module

- **Status:** Accepted (licence: pending team confirmation)
- **Date:** 2026-11-24
- **Deciders:** Demir Cücü; to be reviewed by the full team

## Context

Six team members across Computer, Electrical-Electronics and Industrial Engineering build tightly coupled parts of one system: edge vision, V2I hardware, a SUMO digital twin, AI controllers, a backend and a dashboard. The parts exchange messages (queue telemetry, emergency requests, signal commands, logs), and those formats must stay consistent. Before this decision the code was a single-person prototype with no version control. The repository will be public on GitHub.

## Decision

1. **One repository (monorepo)** with one top-level folder per module, each owned by one person:
   `simulation/` (Öykü), `ai/` (Demir), `backend/` and `dashboard/` (Ario), `edge/` (Nihat), `hardware/` (Tolga), `analysis/` (Ayça).
2. **`shared/contracts/`** holds every cross-team interface: intersection registry, MQTT topic tree, and JSON Schemas. A contract change needs approval from every affected owner. A validation script and CI keep the schemas, the examples and the intersection ids consistent.
3. **Each module manages its own dependencies** (`requirements.txt` / `pyproject.toml` / `package.json` inside the module). There is no single shared Python environment, because module stacks differ (PyTorch, FastAPI, TensorRT, ESP32 toolchains).
4. **Ownership is enforced** with `.github/CODEOWNERS` and branch protection: PRs into `develop` need a review from the owner of every touched folder.
5. **Data, weights and simulation outputs are never committed.** `data/README.md` documents how to obtain the data. Large-file and secret rules are in CONTRIBUTING.md.
6. **Root documentation**: README (overview), CONTRIBUTING (process), `docs/` (architecture, ADRs, interface summary, meeting template), and CLAUDE.md (rules for AI coding assistants).
7. **Licence: MIT, pending team confirmation.** Ultralytics YOLOv8 (AGPL-3.0) may only be an optional dependency of `edge/`. It is never bundled or vendored, so the MIT licence of the rest of the code is unaffected. The team must confirm this before the first public release.

## Alternatives considered

- **One repository per module.** Cleaner isolation, but contracts would drift between repositories, cross-module changes would need several synchronised PRs, and a six-person student team would carry six sets of CI and review settings. Rejected.
- **Monorepo with a single shared Python package.** Simpler imports, but forces one dependency set on modules with conflicting stacks (e.g. TensorRT on Jetson vs. PyTorch training on a laptop) and blurs ownership. Rejected.
- **Git submodules for hardware/firmware.** Adds friction without clear benefit at this team size. Can be revisited if firmware needs its own release cycle.

## Consequences

- Positive: one place for contracts, docs and CI; atomic cross-module PRs; clear ownership.
- Negative: CI must handle per-module setups. Repository size must be watched (2 MB rule, gzipped SUMO nets).
- The existing MARL prototype is kept runnable under `ai/prototype/` until it is replaced by the `ai/` package (see `ai/README.md`, Known issues).
- Git history starts at the import of the prototype on 2026-11-24. There was no earlier version control.

## Open items

- Team confirmation of the MIT licence.
- GitHub usernames for CODEOWNERS (currently `@TODO-<name>` placeholders).
- Branch protection settings on GitHub (`main` and `develop`).
