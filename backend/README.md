# backend/: Distributed backend and BPR routing service

- **Owner:** Ario Bashiri (CE)
- **Status:** planned

## Purpose

The integration hub of the system. It runs the MQTT broker (Eclipse Mosquitto) and FastAPI services (REST + WebSocket). It verifies emergency requests, forwards telemetry and commands between modules, stores system logs, and serves the dashboard and the mobile app. It also hosts the BPR link-delay dynamic routing service.

## Scope

**Belongs here:** FastAPI app; MQTT client/bridge; Mosquitto configuration (with no credentials committed); HMAC/AES verification of V2I requests on the server side, in agreement with Tolga; forwarding of `signal_command` messages to the ESP32 controllers; the log store; the REST/WebSocket API for the dashboard and the mobile app; the BPR routing service; the fallback-to-Webster trigger when the central link fails.

**Does not belong here:** signal-control decisions (`ai/`), firmware (`hardware/`), the dashboard UI (`dashboard/`), the analytical BPR/DTA formulation (`analysis/`).

## Planned sub-structure

```
backend/
├── app/          # FastAPI application (routers, services, models)
├── mqtt/         # broker config templates, topic bridge
├── routing/      # BPR dynamic routing service
├── tests/
└── requirements.txt / pyproject.toml
```

## Interfaces

- **Consumes:** [`queue_telemetry`](../shared/contracts/schemas/queue_telemetry.schema.json), raw V2I frames (TBD), [`signal_command`](../shared/contracts/schemas/signal_command.schema.json) from `ai/`, [`system_log_event`](../shared/contracts/schemas/system_log_event.schema.json) from every module.
- **Produces:** verified [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json), `signal_command` to the ESP32 controllers, REST/WebSocket APIs, the log store.
- Topic tree: [`shared/contracts/mqtt_topics.md`](../shared/contracts/mqtt_topics.md). **Owner: Ario.** Please confirm or change the drafts.

## Setup, run, test

To be added. Configuration through `MQTT_*` and related keys in [`.env.example`](../.env.example).
