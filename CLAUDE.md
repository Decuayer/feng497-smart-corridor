# CLAUDE.md

Guidance for Claude Code sessions in this repository. It applies to every team member's session.

## Project

**Izmir Smart Corridor** (FENG 497 capstone, Izmir University of Economics) is AI-driven adaptive traffic-signal control and emergency green-corridor management for the Alsancak–Konak arterial network in Izmir. It uses a SUMO digital twin, edge vision and V2I. The six pilot intersections are Alsancak Station, Lozan, Montrö, Basmane, Çankaya and Konak Pier. Validation happens **only in simulation and hardware-in-the-loop testbenches**, never on live municipal controllers.

See [README.md](README.md) for the architecture and KPIs, and [shared/contracts/](shared/contracts/) for cross-team interfaces.

## Team and ownership

| Folder | Owner | Scope |
|---|---|---|
| `simulation/` | Öykü Doğa Dabak | OSM→SUMO network, TraCI integration, O/D matrices, scenarios |
| `ai/` | Demir Cücü | MARL signal control, emergency green corridor, forecasting, LLM decision support |
| `backend/` | Ario Bashiri | FastAPI, MQTT/Mosquitto, WebSocket, REST, BPR routing service |
| `dashboard/` | Ario Bashiri | React/Next.js operator dashboard |
| `edge/` | Nihat Efe Kargı | Camera pipeline, YOLOv8-nano/TensorRT, ROI queue estimation |
| `hardware/` | Tolga Uçar | V2I (RF/LoRa/BLE, HMAC/AES), ESP32 relay board, interlock, watchdog |
| `analysis/` | Ayça Avcı | Queueing, Webster baseline, BPR/DTA, CO₂, CAPEX/OPEX |
| `shared/contracts/` | all (DRAFT) | Intersection ids, MQTT topics, JSON Schemas |

## Rules

- **Only modify your own module.** Other modules get placeholders, READMEs or contract proposals only. Never write implementation code in another member's folder. If you need something from another module, write it down as a request.
- Changes to `shared/contracts/` stay **DRAFT** until every affected owner approves them. Keep the `intersection_id` enums in sync with `intersections.yaml` (`python scripts/validate_contracts.py`).
- **Signal safety limits** must be respected by every controller, baseline, test and message: **min green 10 s, max green 60 s, yellow 3 s, all-red 1 s**. Never emit conflicting greens. The hardware interlock and watchdog are authoritative.
- Fallbacks: if the network or central link fails, use local adaptive Webster mode. If a learned policy diverges, clamp actions to the green-time bounds.
- **Never commit secrets, data or model weights**, or SUMO outputs and large binaries (over 2 MB, except gzipped SUMO nets). Config goes through environment variables, and every new key goes into `.env.example`.
- Mark assumptions with `ASSUMPTION` in code comments and docs.
- Use `git mv` to move files. Do not push, merge or rewrite history unless the user asks.

## Conventions

- Branches: `<type>/<module>-<short-desc>`. Commits: Conventional Commits with module scope, e.g. `feat(ai): ...`. PRs target `develop`.
- Python: ruff + black (line length 100). JS/TS: eslint + prettier.
- Unit tests live in `<module>/tests/` and must run without SUMO. Mark SUMO tests with `@pytest.mark.sumo`.
- SUMO comes from the pip `eclipse-sumo` package. Set `SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"` inside the venv.

Full process: [CONTRIBUTING.md](CONTRIBUTING.md).
