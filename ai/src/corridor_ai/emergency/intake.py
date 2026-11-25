"""Emergency priority requests (``emergency_request`` contract, DRAFT v0.1.0).

Two sources produce the same message format:

* simulation: SUMO vehicles of vType ``emergency`` / ``firebrigade``
  (``source: simulation``, ``auth.method: simulation``);
* external messages (V2I -> backend -> ai). These are accepted only if they
  validate against the schema and the backend verified them
  (``auth.verified = true``). Unverified requests are rejected unless
  ``source == "simulation"``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from corridor_ai.common.contracts import validation_errors
from corridor_ai.common.log_events import sim_timestamp

VTYPE_TO_VEHICLE_TYPE = {"emergency": "ambulance", "firebrigade": "fire_truck"}


@dataclass(frozen=True)
class EmergencyRequest:
    request_id: str
    vehicle_id: str
    vehicle_type: str
    request_type: str
    priority: int
    route_intersections: tuple[str, ...]
    sumo_vehicle_id: str | None
    source: str


def build_request_message(
    request_id: str,
    vehicle_id: str,
    vehicle_type: str,
    request_type: str,
    sim_time_s: float,
    route_intersections: list[str],
    speed_mps: float | None = None,
    approaching: str | None = None,
    sumo_vehicle_id: str | None = None,
    priority: int = 1,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "schema_version": "0.1.0",
        "request_id": request_id,
        "timestamp": sim_timestamp(sim_time_s),
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle_type,
        "request_type": request_type,
        "priority": priority,
        "source": "simulation",
        "route": {"intersection_ids": list(route_intersections)},
        "auth": {"verified": True, "method": "simulation"},
        "sim": {"sim_time_s": float(sim_time_s)},
    }
    if speed_mps is not None:
        msg["speed_mps"] = max(float(speed_mps), 0.0)
    if approaching is not None:
        msg["approaching_intersection_id"] = approaching
    if sumo_vehicle_id is not None:
        msg["sim"]["sumo_vehicle_id"] = sumo_vehicle_id
    return msg


def new_request_id() -> str:
    return str(uuid.uuid4())


def parse_request(message: dict[str, Any]) -> tuple[EmergencyRequest | None, str | None]:
    """Validate an incoming message. Returns (request, None) or (None, rejection reason)."""
    errors = validation_errors("emergency_request", message)
    if errors:
        return None, f"schema: {errors[:3]}"
    auth = message["auth"]
    if not auth["verified"] and message["source"] != "simulation":
        return None, "unverified request (auth.verified = false)"
    return (
        EmergencyRequest(
            request_id=message["request_id"],
            vehicle_id=message["vehicle_id"],
            vehicle_type=message["vehicle_type"],
            request_type=message["request_type"],
            priority=int(message.get("priority", 1)),
            route_intersections=tuple((message.get("route") or {}).get("intersection_ids", [])),
            sumo_vehicle_id=(message.get("sim") or {}).get("sumo_vehicle_id"),
            source=message["source"],
        ),
        None,
    )
