"""Simulation core: SUMO session + phase machines + safety monitor + metrics.

``CorridorSim`` owns the per-second loop shared by every controller and by the
MARL environment:

1. controllers place requests with the phase machines;
2. each machine produces the state for the coming second, which is sent to SUMO
   with ``setRedYellowGreenState`` (only when it changed);
3. SUMO advances one second;
4. the states SUMO actually displays are read back. Any difference from what the
   machines displayed counts as **drift** and must stay 0 (regression test for
   the prototype bug). The same states are fed to the SafetyMonitor.

With ``controlled=False`` the machines are bypassed and SUMO runs the network's
own programs (the "actuated reference"). The monitor still reports how those
programs compare with the safety limits.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from corridor_ai.common.config import ExperimentConfig
from corridor_ai.common.registry import Fixture
from corridor_ai.sim.demand import build_demand
from corridor_ai.sim.metrics import parse_tripinfo
from corridor_ai.sim.phase_machine import PhaseMachine
from corridor_ai.sim.safety_monitor import SafetyMonitor
from corridor_ai.sim.session import SumoSession
from corridor_ai.sim.tls_model import TLSModel, load_tls_models


class CorridorSim:
    def __init__(
        self,
        fixture: Fixture,
        cfg: ExperimentConfig,
        seed: int,
        output_dir: Path | None = None,
        scale: float | None = None,
        controlled: bool = True,
        end_s: int | None = None,
    ) -> None:
        self.fixture = fixture
        self.cfg = cfg
        self.seed = seed
        self.scale = cfg.sim.demand_scale if scale is None else scale
        self.controlled = controlled
        self.output_dir = output_dir
        self.end_s = end_s if end_s is not None else cfg.sim.warmup_s + cfg.sim.horizon_s
        self.models: dict[str, TLSModel] = load_tls_models(fixture.net)
        self.session = SumoSession(cfg.sim.backend, cfg.sim.gui)
        self.machines: dict[str, PhaseMachine] = {}
        self.monitor = SafetyMonitor(self.models, cfg.limits)
        self.displayed: dict[str, str] = {}
        self.drift_mismatches = 0
        self.drift_checks = 0
        self.arrived = 0
        self.teleports = 0
        self.time_s = 0.0
        self.tripinfo_path: Path | None = None
        self.second_hooks: list[Callable[[CorridorSim], None]] = []
        self.started = False

    # ---------------------------------------------------------------- setup
    def start(self) -> None:
        demand = build_demand(self.fixture, self.scale, self.seed, self.end_s)
        args = [
            "-n", str(self.fixture.net),
            "-r", str(demand.route_file),
            "--seed", str(self.seed),
            "--scale", f"{demand.sumo_scale:g}",
            "--begin", "0",
            "--end", str(self.end_s),
            "--step-length", "1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", str(self.cfg.sim.time_to_teleport_s),
            "--waiting-time-memory", str(self.cfg.sim.waiting_time_memory_s),
            "--device.emissions.probability", "1",
        ]  # fmt: skip
        if self.cfg.sim.write_tripinfo and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.tripinfo_path = self.output_dir / f"tripinfo_seed{self.seed}.xml"
            args += [
                "--tripinfo-output", str(self.tripinfo_path),
                "--tripinfo-output.write-unfinished", "true",
            ]  # fmt: skip
        args += list(self.cfg.sim.extra_args)
        self.session.start(args)
        if self.controlled:
            self.machines = {
                tid: PhaseMachine(tid, m.green_states, self.cfg.limits)
                for tid, m in self.models.items()
            }
        self.started = True

    # ----------------------------------------------------------------- loop
    @property
    def api(self):
        return self.session.api

    def step_second(self, requests: dict[str, int | None] | None = None) -> None:
        api = self.session.api
        if self.controlled:
            for tid, target in (requests or {}).items():
                self.machines[tid].request(target)
            for tid, machine in self.machines.items():
                state = machine.next_state()
                if self.displayed.get(tid) != state:
                    api.trafficlight.setRedYellowGreenState(tid, state)
                    self.displayed[tid] = state
        self.session.step()
        self.time_s = float(api.simulation.getTime())
        observed = {tid: api.trafficlight.getRedYellowGreenState(tid) for tid in self.models}
        if self.controlled:
            for tid, state in observed.items():
                self.drift_checks += 1
                if state != self.displayed[tid]:
                    self.drift_mismatches += 1
        self.monitor.observe(self.time_s, observed)
        self.arrived += api.simulation.getArrivedNumber()
        self.teleports += api.simulation.getStartingTeleportNumber()
        for hook in self.second_hooks:
            hook(self)

    def run_controllers(self, controllers, seconds: int) -> None:
        for _ in range(seconds):
            requests: dict[str, int | None] = {}
            for c in controllers:
                requests.update(c.act(self))
            self.step_second(requests)

    @property
    def done(self) -> bool:
        return self.time_s >= self.end_s

    # ----------------------------------------------------------------- end
    def close(self, min_depart_s: float = 0.0) -> dict[str, float]:
        """Close SUMO and return episode metrics."""
        self.monitor.finalize(self.time_s)
        self.session.close()
        self.started = False
        metrics: dict[str, float] = {
            "sim_time_s": self.time_s,
            "arrived_cumulative": float(self.arrived),
            "teleports": float(self.teleports),
            "safety_violations": float(self.monitor.count),
            "drift_mismatches": float(self.drift_mismatches),
        }
        if self.tripinfo_path is not None and self.tripinfo_path.exists():
            metrics.update(parse_tripinfo(self.tripinfo_path, min_depart_s=min_depart_s))
        return metrics

    # -------------------------------------------------------------- helpers
    def lane_halting(self, lane: str) -> int:
        return int(self.session.api.lane.getLastStepHaltingNumber(lane))

    def lane_waiting(self, lane: str) -> float:
        return float(self.session.api.lane.getWaitingTime(lane))

    def lane_occupancy(self, lane: str) -> float:
        return float(self.session.api.lane.getLastStepOccupancy(lane)) / 100.0
