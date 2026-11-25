# ai/: MARL signal control, emergency green corridor, forecasting and LLM decision support

- **Owner:** Demir Cücü (CE)
- **Status:** in progress. TASK-301a done (signal-control layer, environment, baselines, PPO smoke training); TASK-302 done (emergency green corridor). TASK-301b and 303 are planned.

## Purpose

| Task | Scope | Status |
|---|---|---|
| TASK-301 | Multi-agent RL signal control (PPO / DQN, PyTorch), one agent per intersection, action masking under the signal safety limits; fixed-time and Webster baselines; TensorBoard | 301a done, 301b planned |
| TASK-302 | Proactive emergency green corridor: pre-clear the current and next two intersections, then hand control back to MARL safely | done |
| TASK-303 | 15-min density forecasting (rolling-origin backtests) and an LLM decision-support agent with grounded briefings and incident summaries | planned |

The original task list (Turkish) is in [`docs/tasks_tr.md`](docs/tasks_tr.md).

**Documentation for the course report:**
- [`docs/signal_control.md`](docs/signal_control.md): phase machine, safety monitor, fixed-time and Webster baselines, `signal_command` output.
- [`docs/mdp_formulation.md`](docs/mdp_formulation.md): state, action and mask, reward, IPPO-PS, training protocol.
- [`docs/emergency_corridor.md`](docs/emergency_corridor.md): route look-ahead, ETA, queue-flush lead time, arbitration, release and recovery, acceptance method.

## Scope

**Belongs here:** RL environments, agents and training; controller logic and arbitration; emergency preemption; forecasting; LLM decision support; the thin SUMO adapter the controllers need; small *placeholder* SUMO fixtures.

**Does not belong here:** the real SUMO network, O/D matrices and scenarios (`simulation/`, Öykü); MQTT, REST/WebSocket and the log store (`backend/`, Ario); firmware and V2I authentication (`hardware/`, Tolga); camera inference (`edge/`, Nihat); analytical queueing / Webster / CO₂ models (`analysis/`, Ayça).

## Layout

```
ai/
├── pyproject.toml · requirements.txt · requirements-dev.txt
├── configs/       base.yaml (safety limits, sim, env, reward) · fixtures.yaml · marl_ppo.yaml · baselines.yaml
├── docs/          signal_control.md · mdp_formulation.md · tasks_tr.md
├── src/corridor_ai/
│   ├── common/    config (typed, limits enforced) · paths · registry (contracts + fixtures) · contracts · seeding
│   ├── sim/       tls_model · phase_machine · safety_monitor · session (libsumo/TraCI) · demand · metrics · core
│   ├── control/   base · fixed_time · webster · command_sink (signal_command JSONL)
│   ├── marl/      env (PettingZoo ParallelEnv) · observations · actions · reward · networks · agents/ppo · train · evaluate
│   └── emergency/ intake · network (movements) · preemption (look-ahead, ETA, arbitration) · scenario · recovery · evaluate
├── scripts/       make_toy_corridor.py · build_osm_corridor.py
├── sim_fixtures/  toy_corridor/ (6 junctions, 3.0 km) · osm_alsancak_v0/ (OSM wizard, 29 lights, corridor.yaml)   ← PLACEHOLDERS
├── tests/         unit/ (no SUMO; CI) · integration/ (@pytest.mark.sumo)
├── prototype/     legacy MAPPO prototype (kept for reference, see Known issues)
├── runs/          gitignored: TensorBoard, evaluation outputs, route cache
└── checkpoints/   gitignored: model weights
```

## Interfaces

| Direction | Contract |
|---|---|
| Consumes | [`intersections.yaml`](../shared/contracts/intersections.yaml) (agent ids), later [`queue_telemetry`](../shared/contracts/schemas/queue_telemetry.schema.json), [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json), [`system_log_event`](../shared/contracts/schemas/system_log_event.schema.json), SUMO network + scenarios (configurable path in `configs/fixtures.yaml`) |
| Produces | [`signal_command`](../shared/contracts/schemas/signal_command.schema.json) (schema-validated JSONL via `control/command_sink.py`, incl. `mode: preemption` / `recovery`; MQTT later through the backend), [`system_log_event`](../shared/contracts/schemas/system_log_event.schema.json) (emergency events, `synthetic: true`), [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json) (for simulated EVs) |

## Setup

From the repository root. Tested with **Python 3.14.4** and **SUMO 1.28.0** (pip `eclipse-sumo`) on macOS. **ASSUMPTION:** Python ≥ 3.12, because the pinned numpy/pandas/scipy versions require it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ai/requirements-dev.txt && pip install -e ai --no-deps
export SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
pre-commit install                      # ruff, black, size check, contracts, ai unit tests
```

## Experiments (one line each)

All commands run from the repository root. Outputs go to `ai/runs/` (TensorBoard, CSV) and `ai/checkpoints/` (weights). Neither is committed.

| Experiment | Command |
|---|---|
| Unit tests (no SUMO) | `cd ai && python -m pytest tests -m "not sumo"` |
| SUMO integration tests incl. drift regression (~30 s) | `cd ai && python -m pytest tests -m sumo` |
| 301a PPO smoke training (20 episodes, ~50 s) | `python -m corridor_ai.marl.train --config ai/configs/marl_ppo.yaml --fixture toy_corridor --seed 0 --episodes 20 --run-name 301a_smoke_seed0` |
| TensorBoard | `tensorboard --logdir ai/runs/ppo_ippo/toy_corridor` |
| 301a baseline comparison (~1 min) | `python -m corridor_ai.marl.evaluate --config ai/configs/marl_ppo.yaml --fixture toy_corridor --controllers fixed_time webster ppo actuated --checkpoint ai/checkpoints/ppo_ippo/toy_corridor/301a_smoke_seed0/model_final.pt --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 --scales 1.0 --out ai/runs/eval/toy_corridor/301a_smoke` |
| 302 OSM-corridor PPO smoke policy (40 episodes, ~4 min) | `python -m corridor_ai.marl.train --config ai/configs/marl_ppo.yaml --fixture osm_corridor --seed 0 --episodes 40 --run-name 302_osm_smoke_seed0` |
| 302 acceptance, toy corridor (~10 min) | `python -m corridor_ai.emergency.evaluate --config ai/configs/emergency.yaml --fixture toy_corridor --controllers fixed_time ppo --checkpoint ai/checkpoints/ppo_ippo/toy_corridor/301a_smoke_seed0/model_final.pt --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 --scales 1.0 1.5 --name 302_acceptance` |
| 302 acceptance, OSM corridor (~10 min) | `python -m corridor_ai.emergency.evaluate --config ai/configs/emergency.yaml --fixture osm_corridor --controllers fixed_time ppo --checkpoint ai/checkpoints/ppo_ippo/osm_corridor/302_osm_smoke_seed0/model_final.pt --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 --scales 1.0 --name 302_acceptance` |
| Rebuild the toy fixture / OSM corridor routes | `python ai/scripts/make_toy_corridor.py` · `python ai/scripts/build_osm_corridor.py` |

Any config value can be overridden with `--set key=value`, e.g. `--set env.reward.beta=0.2 limits.all_red_mode=conflicting_only`.

## Results

### 301a: PPO smoke run vs baselines

> **Fixture: toy_corridor. Relative improvement on a placeholder fixture, not the real digital twin.**

Setup: 1800 s horizon after a 300 s fixed-time warm-up; demand scale 1.0; 10 held-out seeds (1000–1009). PPO = IPPO-PS after **20 training episodes** (smoke run, seed 0), greedy. Mean ± 95 % CI; Δ relative to fixed-time.

| Controller | Travel time (s) | Waiting time (s) | Stops / veh | CO₂ (g/km) | Arrived | Safety violations |
|---|---|---|---|---|---|---|
| fixed-time | 268.9 ± 4.6 | 97.8 ± 2.1 | 3.26 ± 0.07 | 294.9 ± 1.9 | 1163 ± 15 | 0 |
| Webster | 176.1 ± 3.3 (−34.5 %) | 14.9 ± 0.8 (−84.8 %) | 1.67 ± 0.11 (−48.9 %) | 200.1 ± 2.5 (−32.1 %) | 1251 ± 18 | 0 |
| PPO (20 episodes) | 179.7 ± 2.3 (−33.2 %) | 17.4 ± 0.3 (−82.2 %) | 1.86 ± 0.03 (−43.1 %) | 204.6 ± 0.8 (−30.6 %) | 1247 ± 18 | 0 |
| actuated (reference, not safety-compliant) | 263.2 ± 4.3 (−2.1 %) | 92.4 ± 1.8 (−5.6 %) | 3.17 ± 0.05 | 290.3 ± 1.5 | 1160 ± 14 | 276 |

How to read this:
- The toy fixture's fixed-time plan is netconvert's untuned 42 s/42 s split, so it is a weak baseline. **Webster is the meaningful reference.**
- After only 20 episodes PPO is close to Webster but still behind it (waiting 17.4 vs 14.9 s). Closing that gap is the goal of the full training in TASK-301b.
- Learning is visible on the fixed evaluation seed (TensorBoard `eval/*`): return −56.6 → −25.2 and waiting 24.0 → 17.3 s between episodes 5 and 20. Value loss fell 8.1 → 0.13.
- Every run of every safety-enforcing controller had **0 safety violations and 0 drift**.

### 302: emergency green corridor (acceptance)

> **Relative improvement on a placeholder fixture, not the real digital twin.**

Setup:
- 10 EVs per run on the KPI corridor route; EVs **obey signals** (no rescue lane) with speedFactor 1.3;
- horizon 3600 s; held-out seeds 1000–1009;
- baseline = same scenario, same seeds, same EV insertion times, **no preemption** (ASSUMPTION);
- improvement = 1 − median(preempt) / median(baseline);
- MARL = IPPO *smoke* policies (toy: 301a, 20 episodes; OSM: 40 episodes). Full training is 301b.

Full tables: `ai/runs/emergency/<fixture>/302_acceptance/summary.md`. Method: [`docs/emergency_corridor.md`](docs/emergency_corridor.md).

| Fixture | Base controller | Scale | EV transit median: base → preempt (s) | Improvement | ≥ 50 % | Recovery ≤ 180 s (of releases) | Non-EV travel Δ | Violations / drift |
|---|---|---|---|---|---|---|---|---|
| toy_corridor (3.0 km) | fixed-time | 1.0 | 447 → 229 | 48.8 % | no | 88 % | −24 % | 0 / 0 |
| toy_corridor | fixed-time | **1.5 (peak)** | 613 → 302 | **50.7 %** | **yes** | 72 % | −27 % | 0 / 0 |
| toy_corridor | MARL | 1.0 | 241 → 222 | 7.7 % | no | 97 % | −0.2 % | 0 / 0 |
| toy_corridor | MARL | 1.5 | 290 → 245 | 15.7 % | no | 80 % | −0.5 % | 0 / 0 |
| osm_corridor (2.97 km) | fixed-time | 1.0 | 451 → 396 | 12.2 % (mean −20.6 %) | no | 53 % | +11 % | 0 / 0 |
| osm_corridor | MARL | 1.0 | 436 → 370 | 15.1 % | no | 72 % | +1.3 % | 0 / 0 |

How to read this:
- **Physical ceiling.** On an empty toy corridor a preempted EV needs 178 s. Without a rescue lane, an EV behind cars travels at about 50 km/h. Against the MARL baseline (241 s), even a perfect corridor could save at most about 26 % at scale 1.0. Preempted EVs already wait only about 9 s in total.
- **OSM, fixed-time.** Preemption **gridlocked 2 of 10 seeds** (1002, 1004). Without the 90 s hold budget it was 3 of 10. The mean therefore got worse. The placeholder scenario's fixed-time baseline is itself close to saturation. With MARL as the base controller there was no gridlock and the gain was consistent.
- **Recovery** counts releases (including hold timeouts) whose intersection queue stayed no worse than in the matched baseline within 180 s. The rest are mostly *censored*: the next EV's preemption at the same intersection came first (10 EVs/h).

## Tests

| Suite | Count | Covers |
|---|---|---|
| `tests/unit` (CI) | 76 | Config limit enforcement, phase-machine timing (incl. Hypothesis property test), safety-monitor rules, TLS/foe extraction, masks (property test), observations, reward (both S definitions), PPO GAE/masking/save-load, Webster maths, demand determinism, tripinfo parsing, `signal_command` schema |
| `tests/integration` (`-m sumo`) | 19 | **Drift regression** (0 mismatches over 3600 s, both fixtures, keep-only / fixed-time / random legal actions, both all-red modes, every light on the `"online"` program), PettingZoo API, determinism, cumulative arrivals == tripinfo, Webster safety, `signal_command` log; **emergency corridor**: preemption on both fixtures (all six intersections pre-cleared, rank ≤ 2, logs/commands schema-valid), MARL handback and KEEP-only masks, external V2I message intake (unverified rejected), hold budget |

## Requests to teammates

| To | Request | Needed for |
|---|---|---|
| **Öykü Doğa Dabak** | SUMO TLS ids of the six pilot intersections and the order of the 3 km emergency corridor. Please also confirm the DRAFT candidates in `shared/contracts/intersections.yaml` (`dokuz_eylul` = Basmane; `14455843` = Alsancak Station; `joinedS_12610689253_…` = Çankaya; `joinedS_5306649169_…` = Konak Pier) | Real-network agents, 302 corridor KPI |
| **Ayça Avcı** | Saturation flow (veh/h/lane) and lost-time values, ideally per intersection/approach, for the Webster baseline (currently 1800 veh/h/lane and 4 s per phase, ASSUMPTION) | 301 Webster baseline, fallback mode |
| **Tolga Uçar** | `phase_id` format on the ESP32, and any **phase-order constraints** (must phases follow a fixed cycle? can phases be skipped? minimum intergreen per movement?). For 302: does the ESP32 accept a phase *hold* (repeated command for the same phase) and preemption commands mid-cycle? | `signal_command`, `phase_order` free vs cyclic, preemption |
| **Ario Bashiri** | Confirm or change the DRAFT schemas `signal_command`, `system_log_event` and `emergency_request`, and the transport (MQTT vs REST/WebSocket). 302 now emits all three (examples in `ai/runs/emergency/*/302_acceptance/logs/`) | 302 intake, 303 logs, command output |
| **Nihat Efe Kargı & Ario Bashiri** | Where YOLO inference runs: edge (Jetson/RPi, per the spec) or central (per the team diagrams)? | Telemetry latency and topics |

## Known issues (legacy prototype in `prototype/`)

The prototype is kept unchanged for reference. Each issue is fixed in the new `corridor_ai` package:

| # | Prototype issue | Fixed in `corridor_ai` by |
|---|---|---|
| 1 | **Controller did not control the signals**: `setPhase` without pinning, 239/360 (66 %) phase mismatches | Phase machine + `setRedYellowGreenState` ("online" program); drift regression test, 0 mismatches |
| 2 | **"−79.7 % waiting" result invalid** (hard-coded in `plot_results.py`; `evaluate.py` gave −75.7 %; both affected by #1) | New `evaluate.py` with seeds, CIs and disclaimers (results above) |
| 3 | Yellow 6 s, no all-red, no max green, Montrö 6 s green | Phase machine (3 s / 1 s / 10–60 s) + independent safety monitor; min-green clamping in the fixed-time plan |
| 4 | "Fixed-time" baseline was SUMO's default programs (Lozan actuated) | True fixed-time plan (documented cycles) + Webster; actuated reported only as a reference |
| 5 | `completed_trips` counted only the last step's arrivals | Cumulative arrivals + tripinfo; test asserts they are equal |
| 6 | No seeds, scale mismatch between train and eval, no TensorBoard | Seed scheme, same scale per scenario, TensorBoard logging |
| 7 | Hard-coded observation sizes and reward weights | Derived from the TLS model; all weights in YAML |
| 8 | Only 3 of 6 intersections | 6 agents on both fixtures (3 OSM candidates DRAFT) |
| 9 | Network extent larger than the pilot bbox | All 29 lights lie inside the pilot bbox; a cropped pilot fixture is a NICE item |
| 10 | Keep/Switch only | KEEP + switch-to-any-phase with masking (extend / shorten / switch) |
| 11 | Team-summed reward | Per-agent reward and per-agent GAE |
| 12 | Lint excluded | Prototype stays excluded from lint; the new package is linted |
