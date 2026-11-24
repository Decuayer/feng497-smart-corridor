# MQTT topics (DRAFT v0.1.0)

> **Status: DRAFT.** Owner of the broker and topic tree: **Ario Bashiri** (`backend/`).
> All changes go through a PR reviewed by the producer and consumer owners.

Broker: Eclipse Mosquitto. Payloads are UTF-8 JSON and must validate against the schema listed.
Broker host, port and credentials come from environment variables (`MQTT_*` in `.env.example`); they are never committed.

**ASSUMPTION:** topic prefix `izmir/corridor/v1`. `{intersection_id}` is an `id` from
[`intersections.yaml`](intersections.yaml).

| Topic | Payload schema | Producer → consumer | QoS | Retained |
|---|---|---|---|---|
| `izmir/corridor/v1/edge/{intersection_id}/queue` | [`queue_telemetry`](schemas/queue_telemetry.schema.json) | edge (Nihat) → backend (Ario) | 0 | no |
| `izmir/corridor/v1/edge/{intersection_id}/status` | TBD (heartbeat) | edge (Nihat) → backend (Ario) | 1 | yes |
| `izmir/corridor/v1/v2i/emergency/raw` | TBD (signed transponder frame) | V2I gateway (Tolga) → backend (Ario) | 1 | no |
| `izmir/corridor/v1/emergency/request` | [`emergency_request`](schemas/emergency_request.schema.json), `auth.verified` set by backend | backend (Ario) → ai (Demir), dashboard | 1 | no |
| `izmir/corridor/v1/ai/{intersection_id}/signal_command` | [`signal_command`](schemas/signal_command.schema.json) | ai (Demir) → backend (Ario) | 1 | no |
| `izmir/corridor/v1/control/{intersection_id}/signal_command` | [`signal_command`](schemas/signal_command.schema.json) | backend (Ario) → ESP32 (Tolga) | 1 | no |
| `izmir/corridor/v1/control/{intersection_id}/state` | TBD (actual lamp state / phase feedback) | ESP32 (Tolga) → backend (Ario) | 1 | yes |
| `izmir/corridor/v1/logs/{source}` | [`system_log_event`](schemas/system_log_event.schema.json) | all modules → backend log store → ai/decision_support | 1 | no |

## Rules

- The backend is the only component that talks to both the AI and the hardware. The AI never publishes directly to `control/...`.
- Emergency requests reach the AI only after the backend has verified HMAC/AES (`auth.verified = true`).
- Signal commands are *requests*. The hardware interlock (< 35 ms) and the 1000 ms watchdog (flashing-yellow fallback) are authoritative.
- If the central link fails, intersection controllers fall back to local adaptive Webster mode (see `mode: fallback_webster`).
- Production brokers must use TLS and per-client credentials with topic ACLs.

## Open questions

1. Does the backend forward to the AI over MQTT (as above) or over REST/WebSocket? (Ario)
2. Payload of `edge/.../status`, `v2i/emergency/raw` and `control/.../state`. (Nihat, Tolga)
3. Expected publish rate for queue telemetry (assumed 1 Hz). (Nihat)
