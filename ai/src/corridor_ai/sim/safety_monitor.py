"""Independent safety monitor for displayed signal states.

The monitor watches the state SUMO actually shows each second
(``trafficlight.getRedYellowGreenState``). It does not depend on the
PhaseMachine's internal bookkeeping. Every second's state is classified as:

* ``green(k)``: equal to allowed green state k of the network program;
* ``yellow``: contains a yellow light;
* ``clearance``: anything else (the all-red interval).

Rules checked (spec: min green 10 s, max green 60 s, yellow 3 s, all-red 1 s):

1. ``conflict``: two foe links both show priority green 'G' (checked every second);
2. ``min_green`` / ``max_green``: length of every green run (the final run is
   truncated by the end of the simulation and only checked against max green;
   single-phase lights are exempt from max green);
3. ``yellow``: every yellow run lasts exactly ``yellow_s``;
4. ``all_red``: every clearance run lasts exactly ``all_red_s`` and, in
   ``full`` mode, shows only red;
5. ``sequence``: green -> yellow -> clearance -> green, with no shortcuts;
6. ``green_to_red``: no link goes directly from green to red without yellow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.tls_model import GREEN_CHARS, YELLOW_CHARS, TLSModel


@dataclass(frozen=True)
class Violation:
    tls_id: str
    time_s: float
    rule: str
    detail: str


@dataclass
class _Tracker:
    model: TLSModel
    green_index: dict[str, int]
    kind: str | None = None  # "green" | "yellow" | "clearance"
    phase: int | None = None
    run_start: float = 0.0
    run_len: int = 0
    prev_kind: str | None = None
    prev_state: str | None = None
    started: bool = False


@dataclass
class SafetyMonitor:
    models: dict[str, TLSModel]
    limits: SignalLimits
    violations: list[Violation] = field(default_factory=list)
    seconds_observed: int = 0

    def __post_init__(self) -> None:
        self._trackers = {
            tid: _Tracker(m, {s: k for k, s in enumerate(m.green_states)})
            for tid, m in self.models.items()
        }

    # ------------------------------------------------------------------ API
    def observe(self, time_s: float, states: dict[str, str]) -> None:
        """Record the states displayed during the second ending at `time_s`."""
        self.seconds_observed += 1
        for tid, state in states.items():
            self._observe_one(self._trackers[tid], time_s, state)

    def finalize(self, time_s: float) -> None:
        """Close all open runs at the end of the simulation (final run truncated)."""
        for tr in self._trackers.values():
            if tr.started:
                self._close_run(tr, time_s, truncated=True)
                tr.started = False

    @property
    def count(self) -> int:
        return len(self.violations)

    def counts_by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.rule] = out.get(v.rule, 0) + 1
        return out

    # ------------------------------------------------------------ internals
    def _add(self, tid: str, t: float, rule: str, detail: str) -> None:
        self.violations.append(Violation(tid, t, rule, detail))

    def _classify(self, tr: _Tracker, state: str) -> tuple[str, int | None]:
        if state in tr.green_index:
            return "green", tr.green_index[state]
        if any(c in YELLOW_CHARS for c in state):
            return "yellow", None
        return "clearance", None

    def _observe_one(self, tr: _Tracker, t: float, state: str) -> None:
        tid = tr.model.tls_id
        conflicts = tr.model.conflicting_greens(state)
        if conflicts:
            self._add(tid, t, "conflict", f"foe links both green: {conflicts[:3]} in {state}")
        if tr.prev_state is not None:
            for k, (a, b) in enumerate(zip(tr.prev_state, state, strict=False)):
                if a in GREEN_CHARS and b in "rs":
                    self._add(tid, t, "green_to_red", f"link {k} went {a}->{b} without yellow")
                    break

        kind, phase = self._classify(tr, state)
        if not tr.started:
            tr.started, tr.kind, tr.phase, tr.run_start, tr.run_len = True, kind, phase, t, 1
        elif kind == tr.kind and phase == tr.phase:
            tr.run_len += 1
        else:
            self._close_run(tr, t, truncated=False)
            self._check_sequence(tr, kind, t)
            tr.prev_kind = tr.kind
            tr.kind, tr.phase, tr.run_start, tr.run_len = kind, phase, t, 1
        if kind == "clearance" and self.limits.all_red_mode == "full" and set(state) - set("rs"):
            self._add(tid, t, "all_red", f"clearance state not all red: {state}")
        tr.prev_state = state

    def _check_sequence(self, tr: _Tracker, new_kind: str, t: float) -> None:
        allowed = {"green": "yellow", "yellow": "clearance", "clearance": "green"}
        if allowed.get(tr.kind or "") != new_kind:
            self._add(tr.model.tls_id, t, "sequence", f"{tr.kind} -> {new_kind}")

    def _close_run(self, tr: _Tracker, t: float, truncated: bool) -> None:
        lim, tid, n = self.limits, tr.model.tls_id, tr.run_len
        if tr.kind == "green":
            if not truncated and n < lim.min_green_s:
                self._add(tid, t, "min_green", f"phase {tr.phase} green for {n} s")
            if tr.model.n_phases > 1 and n > lim.max_green_s:
                self._add(tid, t, "max_green", f"phase {tr.phase} green for {n} s")
        elif tr.kind == "yellow" and not truncated and n != lim.yellow_s:
            self._add(tid, t, "yellow", f"yellow for {n} s")
        elif tr.kind == "clearance" and not truncated and n != lim.all_red_s:
            self._add(tid, t, "all_red", f"all-red for {n} s")
