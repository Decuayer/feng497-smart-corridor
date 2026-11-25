# Interfaces between modules

Human-readable summary of [`shared/contracts/`](../shared/contracts/). **All interfaces are DRAFT v0.1.0.** The JSON Schemas are the source of truth; this page explains them.

## Intersection ids

Defined in [`intersections.yaml`](../shared/contracts/intersections.yaml) and used in every message.

| id | Display name | SUMO TLS id (real network) | Prototype fixture |
|---|---|---|---|
| `alsancak_station` | Alsancak Station | TBD (Öykü) | DRAFT candidate (see yaml) |
| `lozan` | Lozan | TBD (Öykü) | yes |
| `montro` | Montrö | TBD (Öykü) | yes |
| `basmane` | Basmane | TBD (Öykü) | yes, as legacy agent `dokuz_eylul` (**ASSUMPTION**) |
| `cankaya` | Çankaya | TBD (Öykü) | DRAFT candidate (see yaml) |
| `konak_pier` | Konak Pier | TBD (Öykü) | DRAFT candidate (see yaml) |

## Message flows

| Message | Schema | Producer → consumer | Transport (draft) |
|---|---|---|---|
| Queue telemetry | [`queue_telemetry`](../shared/contracts/schemas/queue_telemetry.schema.json) | edge (Nihat) → backend (Ario) → ai, dashboard | MQTT `edge/{id}/queue` |
| Emergency request | [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json) | V2I (Tolga) → backend (Ario, verifies) → ai (Demir) | MQTT `emergency/request` |
| Signal command | [`signal_command`](../shared/contracts/schemas/signal_command.schema.json) | ai (Demir) → backend (Ario) → ESP32 (Tolga) | MQTT `ai/{id}/signal_command` → `control/{id}/signal_command` |
| System log event | [`system_log_event`](../shared/contracts/schemas/system_log_event.schema.json) | all → backend log store (Ario) → ai/decision_support (Demir) | MQTT `logs/{source}` |

Full topic list: [`mqtt_topics.md`](../shared/contracts/mqtt_topics.md).

## Key semantics

- **Queue telemetry** reports per-lane `queue_vehicles`, `queue_length_m`, `stopped_vehicles` (speed < 0.1 m/s), `occupancy` ∈ [0, 1] and optional per-class counts (car, light commercial, bus, ambulance, fire truck).
- **Emergency request**: one `request_id` per trip, with `request_type` request → update… → cancel. `route.intersection_ids` lists the upcoming signalised intersections in order. The backend sets `auth.verified` after checking HMAC/AES, and the AI ignores unverified requests. In simulation, `source = "simulation"` and `auth.method = "simulation"`.
- **Signal command** is a *request* for the next green phase. The schema enforces `green_duration_s` ∈ [10, 60], `yellow_s = 3` and `all_red_s = 1`. `sequence` is monotonic per intersection. The hardware interlock and watchdog stay authoritative. The `mode` value records why the command was issued (`adaptive`, `preemption`, `recovery`, `fallback_webster`, `fixed_time`).
- **System log event** is a structured event with `source`, `severity`, `event_type`, optional `intersection_id`, a short `message` and a structured `data` payload. Events generated from simulation set `synthetic = true`. The LLM summariser may only reference facts that appear in these events.

## Open questions (by owner)

| # | Question | Owner |
|---|---|---|
| 1 | SUMO TLS ids for all six intersections, and the order of the 3 km pilot corridor | Öykü |
| 2 | Is `dokuz_eylul` (9 Eylül Meydanı) the right stand-in for Basmane? | Öykü |
| 3 | Phase plan ids (`phase_id`) per intersection, shared by SUMO and the ESP32 | Öykü, Tolga |
| 4 | Mapping from edge ROI `lane_id` to SUMO lane ids | Nihat, Öykü |
| 5 | Does the backend forward to the AI over MQTT or REST/WebSocket? Log store technology? | Ario |
| 6 | Payloads for edge heartbeat, raw V2I frames and lamp-state feedback | Nihat, Tolga |
| 7 | Telemetry publish rate (assumed 1 Hz) | Nihat |
