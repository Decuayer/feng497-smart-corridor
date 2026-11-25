"""Build ``system_log_event`` messages (shared/contracts, DRAFT v0.1.0).

Every event is validated against the JSON Schema. Events generated in
simulation carry ``synthetic: true`` and ``sim_time_s``. Their timestamp is
derived from a fixed synthetic epoch (ASSUMPTION), so logs are reproducible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from corridor_ai.common.contracts import validate

SIM_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def sim_timestamp(sim_time_s: float) -> str:
    return (SIM_EPOCH + timedelta(seconds=sim_time_s)).isoformat().replace("+00:00", "Z")


def build_log_event(
    source: str,
    severity: str,
    event_type: str,
    message: str,
    sim_time_s: float,
    intersection_id: str | None = None,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    synthetic: bool = True,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": "0.1.0",
        "event_id": str(uuid.uuid4()),
        "timestamp": sim_timestamp(sim_time_s),
        "source": source,
        "severity": severity,
        "event_type": event_type,
        "message": message[:500],
        "sim_time_s": float(sim_time_s),
        "synthetic": synthetic,
    }
    if intersection_id is not None:
        event["intersection_id"] = intersection_id
    if data:
        event["data"] = data
    if correlation_id is not None:
        event["correlation_id"] = correlation_id
    validate("system_log_event", event)
    return event
