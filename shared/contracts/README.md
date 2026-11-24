# shared/contracts

Cross-team interface definitions. **Everything here is DRAFT (v0.1.0)** until the producer and consumer owners have both approved it in a PR.

| File | Purpose | Producer → consumer |
|---|---|---|
| [`intersections.yaml`](intersections.yaml) | Stable intersection ids, display names, SUMO TLS ids (TBD) | Öykü → everyone |
| [`mqtt_topics.md`](mqtt_topics.md) | MQTT topic tree, QoS, payload schemas | Ario → everyone |
| [`schemas/queue_telemetry.schema.json`](schemas/queue_telemetry.schema.json) | Lane queue telemetry | Nihat → Ario |
| [`schemas/emergency_request.schema.json`](schemas/emergency_request.schema.json) | Emergency vehicle priority request | Tolga → Ario → Demir |
| [`schemas/signal_command.schema.json`](schemas/signal_command.schema.json) | Requested signal phase | Demir → Ario → Tolga |
| [`schemas/system_log_event.schema.json`](schemas/system_log_event.schema.json) | Structured system log event | all → Ario → Demir |

The schemas use JSON Schema draft 2020-12. `schemas/examples/` holds one valid example per schema.

## Rules

- Any module may **propose** a change, but the PR needs approval from every owner the change affects (see `.github/CODEOWNERS`).
- Breaking changes bump `schema_version` and the `$id` suffix.
- The `intersection_id` enum in every schema must match the ids in `intersections.yaml`.

## Validate

```bash
pip install jsonschema pyyaml
python scripts/validate_contracts.py
```

This checks that every schema is valid, that every example validates, and that the intersection enums match `intersections.yaml`. CI runs the same check.

A human-readable summary is in [`docs/interfaces.md`](../../docs/interfaces.md).
