# simulation/: Digital twin (SUMO)

- **Owner:** Öykü Doğa Dabak (CE)
- **Status:** planned

## Purpose

The SUMO digital twin of the Alsancak–Konak arterial network. Other modules develop, train and validate against it: the AI controllers, the emergency corridor, the KPI evaluation and hardware-in-the-loop tests.

## Scope

**Belongs here:**
- OSM → SUMO network for the pilot area (bounding box 38.4150–38.4410 N, 27.1250–27.1550 E), built with `netconvert`
- the six signalised intersections (Alsancak Station, Lozan, Montrö, Basmane, Çankaya, Konak Pier) and their phase plans
- the 3 km emergency KPI corridor
- O/D matrices from IZUM hourly counts
- scenarios: rush hour, accidents and emergency vehicles (vTypes `emergency` / `firebrigade`)
- detectors (E1/E2) and output definitions
- bidirectional TraCI integration helpers
- secondary regions (Bornova, Karşıyaka, Balçova, Buca) for macro routing

**Does not belong here:** RL controllers and emergency preemption logic (`ai/`), analytical models (`analysis/`).

## Planned sub-structure

```
simulation/
├── network/     # *.net.xml.gz (gzipped), netconvert configs, TLS programs
├── scenarios/   # *.sumocfg, routes, vTypes, detectors per scenario
├── traci/       # TraCI integration helpers
├── tools/       # OSM extraction, O/D generation, calibration scripts
└── tests/
```

## Interfaces

- **Produces:** network and scenario files at stable paths, SUMO TLS ids for [`shared/contracts/intersections.yaml`](../shared/contracts/intersections.yaml) (currently `TBD`), and the corridor order for the emergency KPI.
- **Consumes:** IZUM counts (see [`data/README.md`](../data/README.md)), OpenStreetMap data.
- **Open requests from other modules:** see [`docs/interfaces.md`](../docs/interfaces.md#open-questions-by-owner) (questions 1–4).

Note: `ai/sim_fixtures/osm_alsancak_v0/` holds a placeholder OSM-wizard network made by Demir for the AI prototype. You can use it as a starting point; it is not a deliverable of this module.

## Setup, run, test

SUMO is installed through pip (`eclipse-sumo`). See the [root README](../README.md#python-environment-and-sumo). Commit networks gzipped (`*.net.xml.gz`) and never commit SUMO outputs. Commands will be added when the first scenario exists.
