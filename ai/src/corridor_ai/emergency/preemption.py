"""Proactive emergency green corridor: look-ahead, ETA, lead time, arbitration, release.

``PreemptionManager`` runs once per simulated second (as a CorridorSim hook):

1. **Intake.** Newly departed SUMO vehicles of an emergency vType, or external
   ``emergency_request`` messages naming a SUMO vehicle, become tracked
   emergency vehicles (EVs). Each gets a schema-valid request message and a
   ``emergency_request_received`` log event.
2. **Route look-ahead.** The EV's route is mapped to the ordered list of agent
   intersections it will cross (movement in-edge -> out-edge). Only the next
   1 + ``lookahead`` unpassed intersections are candidates ("current + next 2").
3. **ETA.** Distance d to each stop line along the route.
   ETA_opt = free-flow time at the EV's desired speed (limit x speedFactor);
   ETA_nom = max(ETA_opt, d / max(EMA speed, min_speed_frac x free speed)).
4. **Lead time.** A light is pre-cleared (preemption activated) when
       ETA_opt <= T_flush + T_switch + margin
   T_flush = startup + n x headway, where n is the largest number of vehicles
   between the EV and the stop line on one approach lane (``flush_count:
   vehicles_ahead``, default: the moving platoon in front of the EV has to clear
   too) or the largest queue (``halted``). T_switch = time until the serving
   phase can be green (remaining min green + 3 s yellow + 1 s all-red, or the
   remaining transition).
5. **Arbitration.** One active preemption per light. Among competing EVs:
   lower priority number first, then earliest nominal ETA. A running
   preemption is only displaced by a strictly higher priority. Preemption
   overrides the base controller (fixed-time, Webster or MARL) for that light.
6. **Scheduling under safety.** The manager only *requests* the serving phase;
   the PhaseMachine still enforces min/max green, yellow and all-red. If
   holding the green would exceed max green before the EV arrives, the light
   re-services the other phases (min green) and comes back ("reservice"). This
   happens only if even the optimistic ETA leaves time for a full re-service
   plus the flush time, so the EV never meets the re-serviced phase.
   **Hold budget:** a pre-clear held for ``max_hold_s`` without the EV passing
   (e.g. it is stuck behind a non-preempted light) is released
   (``hold_timeout``). That EV/intersection may then only re-activate on its
   *nominal* ETA, so cross streets are not starved for a blocked vehicle.
7. **Release and handback.** When the EV has passed the light (its route index
   moves beyond the approach edge), the preemption ends. The base controller
   resumes from the current, safe phase-machine state (no forced switch).
   The next phase change is logged as ``mode: recovery``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from corridor_ai.common.config import ExperimentConfig
from corridor_ai.common.log_events import build_log_event
from corridor_ai.common.registry import Fixture
from corridor_ai.emergency.intake import (
    VTYPE_TO_VEHICLE_TYPE,
    build_request_message,
    new_request_id,
    parse_request,
)
from corridor_ai.emergency.network import Movement, load_road_network
from corridor_ai.sim.phase_machine import GREEN, YELLOW, PhaseMachine

if TYPE_CHECKING:
    from corridor_ai.sim.core import CorridorSim


@dataclass
class Target:
    index: int  # route index of the approach edge
    movement: Movement
    intersection_id: str
    eta_opt: float = float("inf")
    eta_nom: float = float("inf")
    distance_m: float = float("inf")


@dataclass
class TrackedEV:
    vehicle_id: str
    request_id: str
    vehicle_type: str
    priority: int
    route: tuple[str, ...]
    targets: list[Target]
    depart_s: float
    ema_speed: float = 0.0
    passed: set[int] = field(default_factory=set)
    route_index: int = 0
    timed_out: set[int] = field(default_factory=set)  # targets released by the hold budget


@dataclass
class ActivePreemption:
    tls_id: str
    intersection_id: str
    vehicle_id: str
    request_id: str
    target_index: int
    movement: Movement
    start_s: float
    eta_at_start: float
    queue_at_start: int
    eta_nom: float
    eta_opt: float = float("inf")
    t_flush: float = 0.0
    reservices: int = 0
    hold_s: int = 0


def time_to_green(m: PhaseMachine, phase: int, min_green: int, yellow: int, all_red: int) -> float:
    """Seconds until `phase` can be displayed, respecting the safety limits."""
    if m.mode == GREEN:
        if m.phase == phase:
            return 0.0
        return max(0, min_green - m.green_elapsed) + yellow + all_red
    remaining = (yellow - m.t_in_mode + all_red) if m.mode == YELLOW else (all_red - m.t_in_mode)
    remaining = max(remaining, 0)
    if m.phase == phase:
        return float(remaining)
    return float(remaining + min_green + yellow + all_red)


class PreemptionManager:
    def __init__(self, fixture: Fixture, cfg: ExperimentConfig, enabled: bool) -> None:
        self.fixture = fixture
        self.cfg = cfg
        self.em = cfg.emergency
        self.limits = cfg.limits
        self.enabled = enabled
        self.net = load_road_network(fixture.net)
        self.agent_by_tls = {tls: cid for cid, tls in fixture.agents.items()}
        self.evs: dict[str, TrackedEV] = {}
        self.finished: dict[str, TrackedEV] = {}
        self.pending_messages: dict[str, dict[str, Any]] = {}  # sumo vehicle id -> message
        self.active: dict[str, ActivePreemption] = {}
        self.recovering: dict[str, str] = {}  # tls -> request id of the last preemption
        self.request_messages: list[dict[str, Any]] = []
        self.rejected: list[tuple[str, str]] = []
        self.events: list[dict[str, Any]] = []
        self.activations: list[dict[str, Any]] = []
        self.releases: list[dict[str, Any]] = []
        self._recovery_watch: dict[str, tuple[float, int, str]] = {}  # tls -> (t, queue, req)

    # ------------------------------------------------------------------ intake
    def submit_message(self, message: dict[str, Any], sim: CorridorSim | None = None) -> bool:
        """Accept an external emergency_request (V2I -> backend). Returns True if accepted."""
        req, reason = parse_request(message)
        t = sim.time_s if sim is not None else 0.0
        if req is None:
            self.rejected.append((message.get("request_id", "?"), reason or ""))
            self._log(t, "warning", "other", f"Emergency request rejected: {reason}", None,
                      {"rejected": True}, message.get("request_id"))  # fmt: skip
            return False
        if req.request_type == "cancel":
            for ev in list(self.evs.values()):
                if ev.request_id == req.request_id:
                    self._finish(ev, t, "cancelled")
            return True
        if req.sumo_vehicle_id is None:
            self.rejected.append((req.request_id, "no sim.sumo_vehicle_id to track in simulation"))
            return False
        self.pending_messages[req.sumo_vehicle_id] = message
        if sim is not None:
            self._register_pending(sim)
        return True

    def _register_pending(self, sim: CorridorSim) -> None:
        if not self.pending_messages:
            return
        present = set(sim.api.vehicle.getIDList())
        for vid in [v for v in self.pending_messages if v in present]:
            msg = self.pending_messages.pop(vid)
            if vid not in self.evs:
                self._register(sim, vid, msg["request_id"], msg["vehicle_type"],
                               int(msg.get("priority", 1)), external=True)  # fmt: skip

    def _register(
        self, sim: CorridorSim, vid: str, request_id: str, vehicle_type: str, priority: int,
        external: bool = False,
    ) -> None:  # fmt: skip
        route = tuple(sim.api.vehicle.getRoute(vid))
        targets = []
        for j in range(len(route) - 1):
            mv = self.net.movements.get((route[j], route[j + 1]))
            if mv and mv.tls_id in self.agent_by_tls and mv.phases:
                targets.append(Target(j, mv, self.agent_by_tls[mv.tls_id]))
        ev = TrackedEV(vid, request_id, vehicle_type, priority, route, targets, sim.time_s)
        self.evs[vid] = ev
        ids = [tg.intersection_id for tg in targets]
        if not external:
            msg = build_request_message(
                request_id, vid, vehicle_type, "request", sim.time_s, ids,
                speed_mps=sim.api.vehicle.getSpeed(vid),
                approaching=ids[0] if ids else None, sumo_vehicle_id=vid, priority=priority,
            )  # fmt: skip
            self.request_messages.append(msg)
        self._log(sim.time_s, "info", "emergency_request_received",
                  f"Priority request from {vehicle_type} {vid}; route crosses "
                  f"{len(ids)} controlled intersections.",
                  ids[0] if ids else None, {"route_intersections": ids, "priority": priority},
                  request_id)  # fmt: skip

    def _finish(self, ev: TrackedEV, t: float, reason: str) -> None:
        for tls, act in list(self.active.items()):
            if act.vehicle_id == ev.vehicle_id:
                self._release(tls, t, reason)
        self.evs.pop(ev.vehicle_id, None)
        self.finished[ev.vehicle_id] = ev

    # ------------------------------------------------------------------ update
    def update(self, sim: CorridorSim) -> None:
        api, t = sim.api, sim.time_s
        for vid in api.simulation.getDepartedIDList():
            vtype = api.vehicle.getTypeID(vid)
            if vtype in self.em.ev_vtypes and vid not in self.evs:
                self._register(
                    sim, vid, new_request_id(), VTYPE_TO_VEHICLE_TYPE.get(vtype, "ambulance"), 1
                )
        self._register_pending(sim)
        for vid in api.simulation.getArrivedIDList():
            if vid in self.evs:
                self._finish(self.evs[vid], t, "arrived")

        for ev in list(self.evs.values()):
            try:
                idx = int(api.vehicle.getRouteIndex(ev.vehicle_id))
                pos = float(api.vehicle.getLanePosition(ev.vehicle_id))
                road = api.vehicle.getRoadID(ev.vehicle_id)
                speed = float(api.vehicle.getSpeed(ev.vehicle_id))
            except Exception:  # noqa: BLE001 - vehicle left the network
                self._finish(ev, t, "left_network")
                continue
            ev.route_index = idx
            ev.ema_speed = 0.7 * ev.ema_speed + 0.3 * speed
            for tg in ev.targets:
                if tg.index in ev.passed:
                    continue
                if idx > tg.index:
                    ev.passed.add(tg.index)
                    act = self.active.get(tg.movement.tls_id)
                    if act and act.vehicle_id == ev.vehicle_id and act.target_index == tg.index:
                        self._release(tg.movement.tls_id, t, "passed")
                    continue
                self._update_eta(ev, tg, idx, pos, road)

        if self.enabled:
            self._enforce_hold_budget(sim)
            self._arbitrate(sim)
        self._watch_recovery(sim)
        for act in self.active.values():
            act.hold_s += 1

    def _update_eta(self, ev: TrackedEV, tg: Target, idx: int, pos: float, road: str) -> None:
        net, sf = self.net, self.em.ev_speed_factor
        cur = ev.route[idx]
        length = net.edge_length[cur]
        rem = 0.0 if road.startswith(":") else max(length - pos, 0.0)
        d = rem
        free = net.free_time(cur, sf) * (rem / length if length > 0 else 0.0)
        for e in ev.route[idx + 1 : tg.index + 1]:
            d += net.edge_length[e]
            free += net.free_time(e, sf)
        v_floor = self.em.min_speed_frac * net.edge_speed[cur] * sf
        tg.distance_m = d
        tg.eta_opt = free
        tg.eta_nom = max(free, d / max(ev.ema_speed, v_floor, 0.1))

    # ------------------------------------------------------------- arbitration
    def _arbitrate(self, sim: CorridorSim) -> None:
        best: dict[str, tuple[tuple, TrackedEV, Target]] = {}
        for ev in self.evs.values():
            upcoming = [tg for tg in ev.targets if tg.index not in ev.passed]
            for tg in upcoming[: self.em.lookahead + 1]:
                key = (ev.priority, tg.eta_nom)
                cur = best.get(tg.movement.tls_id)
                if cur is None or key < cur[0]:
                    best[tg.movement.tls_id] = (key, ev, tg)

        lim = self.limits
        for tls, (_key, ev, tg) in best.items():
            act = self.active.get(tls)
            if act is not None:
                if act.vehicle_id == ev.vehicle_id and act.target_index == tg.index:
                    act.eta_nom, act.eta_opt = tg.eta_nom, tg.eta_opt
                    act.t_flush = self._flush_time(sim, ev, tg)[1]
                    continue
                current_ev = self.evs.get(act.vehicle_id)
                if current_ev is not None and ev.priority >= current_ev.priority:
                    continue  # only a strictly higher priority displaces a running preemption
                self._release(tls, sim.time_s, "displaced")
            m = sim.machines[tls]
            phase = self._serving_phase(m, tg.movement)
            t_switch = time_to_green(m, phase, lim.min_green_s, lim.yellow_s, lim.all_red_s)
            n_q, t_flush = self._flush_time(sim, ev, tg)
            eta = tg.eta_nom if tg.index in ev.timed_out else tg.eta_opt
            if eta <= t_flush + t_switch + self.em.margin_s:
                self._activate(sim, tls, ev, tg, n_q, t_flush, t_switch)

    def _flush_time(self, sim: CorridorSim, ev: TrackedEV, tg: Target) -> tuple[int, float]:
        """(vehicles to clear on the worst approach lane, flush time in seconds)."""
        api = sim.api
        if self.em.flush_count == "halted":
            n = max((sim.lane_halting(ln) for ln in tg.movement.in_lanes), default=0)
        else:
            on_edge = ev.route_index == tg.index
            ev_pos = float(api.vehicle.getLanePosition(ev.vehicle_id)) if on_edge else -1.0
            n = 0
            for ln in tg.movement.in_lanes:
                ahead = [
                    v for v in api.lane.getLastStepVehicleIDs(ln)
                    if v != ev.vehicle_id and float(api.vehicle.getLanePosition(v)) > ev_pos
                ]  # fmt: skip
                n = max(n, len(ahead))
        t = (self.em.startup_lost_s + n * self.em.saturation_headway_s) if n else 0.0
        return n, t

    def _enforce_hold_budget(self, sim: CorridorSim) -> None:
        for tls, act in list(self.active.items()):
            if act.hold_s >= self.em.max_hold_s:
                ev = self.evs.get(act.vehicle_id)
                if ev is not None:
                    ev.timed_out.add(act.target_index)
                self._release(tls, sim.time_s, "hold_timeout")

    @staticmethod
    def _serving_phase(m: PhaseMachine, mv: Movement) -> int:
        if m.mode == GREEN and m.phase in mv.phases:
            return m.phase
        if m.mode != GREEN and m.phase in mv.phases:  # already changing to a serving phase
            return m.phase
        return mv.phases[0]

    def _activate(self, sim, tls, ev, tg, n_q, t_flush, t_switch) -> None:
        q_total = self._intersection_queue(sim, tls)
        act = ActivePreemption(tls, tg.intersection_id, ev.vehicle_id, ev.request_id, tg.index,
                               tg.movement, sim.time_s, tg.eta_opt, q_total, tg.eta_nom,
                               tg.eta_opt, t_flush)  # fmt: skip
        self.active[tls] = act
        self.recovering.pop(tls, None)
        self._recovery_watch.pop(tls, None)
        rank = [t.index for t in ev.targets if t.index not in ev.passed].index(tg.index)
        rec = {"tls_id": tls, "intersection_id": tg.intersection_id, "vehicle_id": ev.vehicle_id,
               "request_id": ev.request_id, "start_s": sim.time_s,
               "eta_opt_s": round(tg.eta_opt, 1),
               "vehicles_to_clear": n_q, "t_flush_s": t_flush, "t_switch_s": t_switch,
               "lookahead_rank": rank}  # fmt: skip
        self.activations.append(rec)
        self._log(sim.time_s, "info", "preemption_started",
                  f"Pre-clearing for {ev.vehicle_type} {ev.vehicle_id}: ETA {tg.eta_opt:.0f} s, "
                  f"{n_q} vehicles to clear on its approach (lookahead rank {rank}).",
                  tg.intersection_id,
                  {"eta_s": round(tg.eta_opt, 1), "vehicles_to_clear": n_q, "lookahead_rank": rank,
                   "t_flush_s": t_flush, "t_switch_s": t_switch},
                  ev.request_id)  # fmt: skip

    def _release(self, tls: str, t: float, reason: str) -> None:
        act = self.active.pop(tls, None)
        if act is None:
            return
        self.recovering[tls] = act.request_id
        rec = {"tls_id": tls, "intersection_id": act.intersection_id, "vehicle_id": act.vehicle_id,
               "start_s": act.start_s, "release_s": t, "duration_s": t - act.start_s,
               "reason": reason, "reservices": act.reservices}  # fmt: skip
        self.releases.append(rec)
        self._recovery_watch[tls] = (t, act.queue_at_start, act.request_id)
        self._log(t, "info", "preemption_ended",
                  f"Preemption ended ({reason}) after {t - act.start_s:.0f} s; "
                  "control returned to the base controller.",
                  act.intersection_id, {"duration_s": t - act.start_s, "reason": reason,
                                        "reservices": act.reservices},
                  act.request_id)  # fmt: skip

    # --------------------------------------------------------------- requests
    def requests(self, sim: CorridorSim) -> dict[str, int]:
        """Phase requests for every light under active preemption (override base control)."""
        out: dict[str, int] = {}
        lim, em = self.limits, self.em
        transition = lim.yellow_s + lim.all_red_s
        for tls, act in self.active.items():
            m = sim.machines[tls]
            phase = self._serving_phase(m, act.movement)
            if (
                em.reservice
                and m.mode == GREEN
                and m.phase in act.movement.phases
                and m.n_phases > len(act.movement.phases)
                and m.green_elapsed + act.eta_nom > lim.max_green_s
                and act.eta_opt > lim.min_green_s + 2 * transition + em.margin_s + act.t_flush
            ):
                other = next(p for p in range(m.n_phases) if p not in act.movement.phases)
                out[tls] = other
                act.reservices += 1
            else:
                out[tls] = phase
        return out

    @property
    def preempted_tls(self) -> set[str]:
        return set(self.active)

    def command_mode(self, tls: str) -> tuple[str, str, str] | None:
        """(issuer, mode, request id) override for signal_command, or None for base control."""
        if tls in self.active:
            return "emergency_preemption", "preemption", self.active[tls].request_id
        if tls in self.recovering:
            return "", "recovery", self.recovering.pop(tls)  # "" = keep the base issuer
        return None

    # --------------------------------------------------------------- recovery
    def _intersection_queue(self, sim: CorridorSim, tls: str) -> int:
        return sum(sim.lane_halting(ln) for ln in sim.models[tls].incoming_lanes)

    def _watch_recovery(self, sim: CorridorSim) -> None:
        """Online indicator: queue back to its pre-preemption level (+ tolerance)."""
        for tls, (t_rel, q_ref, req_id) in list(self._recovery_watch.items()):
            q = self._intersection_queue(sim, tls)
            tol = max(self.em.recovery_tolerance_frac * q_ref, self.em.recovery_tolerance_abs)
            if q <= q_ref + tol:
                self._recovery_watch.pop(tls)
                self._log(sim.time_s, "info", "recovery_completed",
                          f"Queue back to pre-preemption level ({q} vs {q_ref} vehicles) "
                          f"{sim.time_s - t_rel:.0f} s after release.",
                          self.agent_by_tls[tls],
                          {"recovery_s": sim.time_s - t_rel, "queue": q, "queue_ref": q_ref},
                          req_id)  # fmt: skip

    # ------------------------------------------------------------------- logs
    def _log(self, t, severity, event_type, message, intersection_id, data, correlation_id=None):
        self.events.append(
            build_log_event("ai", severity, event_type, message, t, intersection_id, data,
                            correlation_id)  # fmt: skip
        )
