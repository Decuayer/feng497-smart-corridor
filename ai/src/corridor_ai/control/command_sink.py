"""Emit ``signal_command`` messages (shared/contracts, DRAFT v0.1.0).

The recorder is attached as a per-second hook to a CorridorSim. Whenever the
PhaseMachine of an *agent* intersection starts a phase change, it builds one
message per the schema. Messages are validated and written as JSON lines.
The backend (Ario) will later replace the JSONL file with MQTT publishing.

Field conventions (DRAFT, to confirm with Tolga / Ario):
* ``phase_id``: green-phase index in the fixture's program, as a string
  (ASSUMPTION until the shared phase-id format exists);
* ``green_duration_s``: the planned green for plan controllers. For adaptive
  (MARL) control it is the committed minimum (min green). The controller may
  extend it with further commands, always up to max green;
* ``valid_for_ms``: the decision interval for adaptive control.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from corridor_ai.common.contracts import validate

if TYPE_CHECKING:
    from corridor_ai.sim.core import CorridorSim

SIM_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)  # ASSUMPTION: synthetic timestamps for sim time


def build_signal_command(
    intersection_id: str,
    sequence: int,
    phase_index: int,
    green_duration_s: float,
    issuer: str,
    mode: str,
    sim_time_s: float,
    valid_for_ms: int | None = None,
    reason: str | None = None,
    yellow_s: int = 3,
    all_red_s: int = 1,
) -> dict:
    msg = {
        "schema_version": "0.1.0",
        "command_id": str(uuid.uuid4()),
        "timestamp": (SIM_EPOCH + timedelta(seconds=sim_time_s)).isoformat().replace("+00:00", "Z"),
        "intersection_id": intersection_id,
        "sequence": sequence,
        "issuer": issuer,
        "mode": mode,
        "target_phase": {"phase_id": str(phase_index), "green_duration_s": green_duration_s},
        "transition": {"yellow_s": yellow_s, "all_red_s": all_red_s},
    }
    if valid_for_ms is not None:
        msg["valid_for_ms"] = valid_for_ms
    if reason:
        msg["reason"] = reason[:200]
    return msg


class SignalCommandRecorder:
    def __init__(
        self,
        agent_tls: dict[str, str],  # contract id -> TLS id
        issuer: str,
        mode: str,
        green_for: dict[str, list[int]] | None = None,  # TLS -> planned greens (plans)
        valid_for_ms: int | None = None,
        path: Path | None = None,
    ) -> None:
        self.agent_tls = agent_tls
        self.issuer, self.mode = issuer, mode
        self.green_for = green_for or {}
        self.valid_for_ms = valid_for_ms
        self.path = path
        self.messages: list[dict] = []
        self._seq = {cid: 0 for cid in agent_tls}
        self._switches: dict[str, int] = {}

    def sync(self, sim: CorridorSim) -> None:
        """Mark all switches so far as seen without emitting (e.g. after warm-up)."""
        for tls in self.agent_tls.values():
            self._switches[tls] = sim.machines[tls].switches

    def __call__(self, sim: CorridorSim) -> None:
        for cid, tls in self.agent_tls.items():
            m = sim.machines[tls]
            if m.switches == self._switches.get(tls, 0):
                continue
            self._switches[tls] = m.switches
            plan = self.green_for.get(tls)
            green = float(plan[m.phase]) if plan else float(sim.cfg.limits.min_green_s)
            self._seq[cid] += 1
            msg = build_signal_command(
                cid, self._seq[cid], m.phase, green, self.issuer, self.mode, sim.time_s,
                self.valid_for_ms, yellow_s=sim.cfg.limits.yellow_s,
                all_red_s=sim.cfg.limits.all_red_s,
            )  # fmt: skip
            validate("signal_command", msg)
            self.messages.append(msg)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(msg) + "\n")
