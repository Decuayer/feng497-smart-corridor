# ai/: MARL signal control, emergency green corridor, forecasting and LLM decision support

- **Owner:** Demir Cücü (CE)
- **Status:** in progress. Legacy prototype only; the new `corridor_ai` package is planned (TASK-301/302/303).

## Purpose

Decision-making layer of the corridor:

| Task | Scope |
|---|---|
| TASK-301 | Multi-agent RL signal control (PPO / DQN, PyTorch), one agent per intersection, with action masking under the signal safety limits. Fixed-time and Webster baselines. TensorBoard logging. |
| TASK-302 | Proactive emergency green corridor: when an ambulance or fire truck requests priority, pre-clear its current and next two intersections, then hand control back to MARL safely. |
| TASK-303 | 15-minute-ahead density forecasting (validated with rolling-origin backtests) and an LLM decision-support agent that writes grounded briefings and incident summaries from system logs. |

The original task list (Turkish) is in [`docs/tasks_tr.md`](docs/tasks_tr.md).

## Scope

**Belongs here:** RL environments, agents and training; controller logic and arbitration; emergency preemption; forecasting models and backtests; LLM client, prompts and grounding; the thin TraCI adapter the controllers need; small *placeholder* SUMO fixtures for development and tests.

**Does not belong here:** the real Alsancak–Konak SUMO network, O/D matrices and scenarios (`simulation/`, Öykü); MQTT broker, REST/WebSocket services and the log store (`backend/`, Ario); firmware and V2I authentication (`hardware/`, Tolga); camera inference (`edge/`, Nihat); analytical queueing / Webster / CO₂ models (`analysis/`, Ayça).

## Current contents

```
ai/
├── README.md
├── docs/tasks_tr.md                  # original task list (Turkish)
├── prototype/                        # legacy MAPPO prototype (pre-restructure), kept runnable
│   ├── prototype_paths.py            # all paths, resolved relative to this folder
│   ├── envs/sumo_marl_env.py         # TraCI env, 3 agents, Keep/Switch actions
│   ├── models/mappo_networks.py      # per-agent actors + centralised critic
│   ├── train.py · evaluate.py · visualize_gui.py · plot_results.py
│   ├── configs/intersections_config.{json,txt}
│   ├── scripts/test_env.py · scripts/test_model.py   # smoke scripts (not pytest)
│   └── requirements.txt              # versions the prototype was tested with
├── sim_fixtures/osm_alsancak_v0/     # PLACEHOLDER OSM-wizard network (see its README)
├── checkpoints/                      # gitignored: model weights
└── runs/                             # gitignored: TensorBoard logs, plots
```

## Planned sub-structure

To be refined in the Part B plan:

```
ai/
├── pyproject.toml, requirements.txt
├── configs/           marl_ppo.yaml, marl_dqn.yaml, emergency.yaml, forecasting.yaml, llm.yaml
├── docs/              mdp_formulation.md, emergency_corridor.md, forecasting.md, llm_agent.md
├── src/corridor_ai/   common/, sim/, marl/, emergency/, forecasting/, decision_support/
├── sim_fixtures/      toy_corridor/ (netgenerate placeholder), osm_alsancak_v0/
├── scripts/
└── tests/
```

## Interfaces

| Direction | Contract |
|---|---|
| Consumes | [`intersections.yaml`](../shared/contracts/intersections.yaml), [`queue_telemetry`](../shared/contracts/schemas/queue_telemetry.schema.json), [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json), [`system_log_event`](../shared/contracts/schemas/system_log_event.schema.json), SUMO network + scenarios from `simulation/` (configurable path) |
| Produces | [`signal_command`](../shared/contracts/schemas/signal_command.schema.json), `system_log_event` (ai source), forecasts and briefings (format TBD with Ario) |

## Setup

From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ai/prototype/requirements.txt
export SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
```

Tested with Python **3.14.4** and SUMO **1.28.0** (pip `eclipse-sumo`) on macOS.

## Run the prototype

```bash
cd ai/prototype
python scripts/test_model.py   # builds the MAPPO model with dummy observations
python scripts/test_env.py     # 20 random steps in headless SUMO
python evaluate.py             # "baseline" vs MAPPO; needs ai/checkpoints/mappo_corridor.pt
python train.py                # 50 episodes; writes ai/checkpoints/mappo_corridor.pt
python visualize_gui.py        # sumo-gui replay of the trained policy
```

The weights file `ai/checkpoints/mappo_corridor.pt` is not committed. Run `train.py` to create it.

Reference output of `evaluate.py` with the local checkpoint, before and after the restructure (identical):

| Metric | "Baseline" | MAPPO |
|---|---|---|
| Avg. waiting time (s, summed over monitored lanes) | 914.70 | 221.95 |
| Avg. queue (vehicles) | 25.02 | 14.76 |

**These numbers are not valid evidence of improvement.** See Known issues.

## Tests

There are no pytest tests yet; `prototype/scripts/test_*.py` are manual smoke scripts. Unit tests will live in `ai/tests/` and run without SUMO. SUMO integration tests will be marked `@pytest.mark.sumo`.

## Known issues (legacy prototype)

These are documented, not fixed. They will be fixed in TASK-301a.

1. **The controller does not actually control the signals.** `SumoTTLEnv` calls `traci.trafficlight.setPhase()` but never pins the phase duration, so SUMO's built-in program keeps advancing phases on its own. In a 600 s headless run where every agent always chooses KEEP, the controller's phase state differed from SUMO's actual phase in **239 of 360 checks (66 %)**.
2. **The reported "−79.7 % waiting time" result is invalid.** It comes from hard-coded numbers in `plot_results.py` that the current `evaluate.py` does not reproduce (it gives −75.7 %). Because of issue 1, neither number measures what the policy does. `benchmark_comparison.png` is therefore not committed.
3. **Signal safety limits are not met.** Yellow is 6 s (spec: 3 s). There is no all-red phase (spec: 1 s) except one unused 1 s phase at `dokuz_eylul`. There is no max-green cap (spec: 60 s), so KEEP can hold a green forever. Montrö phase 4 is a 6 s green in the SUMO program, below the 10 s minimum.
4. **The "fixed-time" baseline is not fixed-time.** It runs the network's default programs, and Lozan's is `delay_based` (actuated). Cycle lengths and splits are undocumented.
5. **Trip counting is wrong.** `completed_trips` uses `traci.simulation.getArrivedNumber()`, which counts only arrivals in the *last* simulation step (hence "2 vs 1").
6. **Not reproducible.** No SUMO or PyTorch seeds, evaluation uses `--scale 2.0` while training uses the default scale, and nothing is logged to TensorBoard.
7. **Hard-coded structure.** Observation dimensions (`{"lozan": 18, "montro": 19, "dokuz_eylul": 20}`), reward weights (`0.7`, `0.3`, the ×10 stop factor and the /1000 scale) and hyper-parameters are constants in the code, not configuration.
8. **Only 3 of the 6 pilot intersections are modelled**, and `dokuz_eylul` stands in for Basmane (**ASSUMPTION**).
9. **The network covers more than the pilot area.** Its bounding box is 38.393–38.455 N, 27.067–27.174 E; the pilot bounding box is 38.4150–38.4410 N, 27.1250–27.1550 E.
10. **The action space is only Keep/Switch** (cyclic phase order). The task also asks for extend/shorten actions.
11. **The reward is a team sum, not a per-agent reward** (`step_reward = sum(rewards.values())`), and the actor loss uses the shared advantage for every agent.
12. The ruff/black CI lint excludes `ai/prototype/`. The code has unused imports and is left as-is on purpose.
