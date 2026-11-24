"""Fixed-time (pre-timed) controller.

The plan for each traffic light is the original phase order of the network
program, with each green duration clamped to [min_green_s, max_green_s].
Yellow (3 s) and all-red (1 s) are inserted by the PhaseMachine. The cycle
length is sum(greens) + K * (yellow + all-red). Offset 0 for every light
(ASSUMPTION: no coordination).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.tls_model import TLSModel

if TYPE_CHECKING:
    from corridor_ai.sim.core import CorridorSim


@dataclass(frozen=True)
class SignalPlan:
    tls_id: str
    greens_s: tuple[int, ...]
    yellow_s: int
    all_red_s: int

    @property
    def cycle_s(self) -> int:
        return sum(self.greens_s) + len(self.greens_s) * (self.yellow_s + self.all_red_s)

    def as_dict(self) -> dict:
        return {
            "greens_s": list(self.greens_s),
            "yellow_s": self.yellow_s,
            "all_red_s": self.all_red_s,
            "cycle_s": self.cycle_s,
        }


def clamp_green(value: float, limits: SignalLimits) -> int:
    return int(min(max(round(value), limits.min_green_s), limits.max_green_s))


def fixed_time_plan(model: TLSModel, limits: SignalLimits) -> SignalPlan:
    greens = tuple(clamp_green(d, limits) for d in model.planned_green_s)
    return SignalPlan(model.tls_id, greens, limits.yellow_s, limits.all_red_s)


class PlanController:
    """Runs a static cyclic plan (used for fixed-time and Webster plans)."""

    def __init__(self, plans: dict[str, SignalPlan]) -> None:
        self.plans = plans
        self.tls_ids = tuple(plans)

    def reset(self, sim: CorridorSim) -> None:  # noqa: ARG002 - interface
        return None

    def act(self, sim: CorridorSim) -> dict[str, int | None]:
        requests: dict[str, int | None] = {}
        for tid, plan in self.plans.items():
            m = sim.machines[tid]
            if m.n_phases < 2 or m.in_transition:
                continue
            if m.green_elapsed >= plan.greens_s[m.phase]:
                requests[tid] = (m.phase + 1) % m.n_phases
        return requests


def fixed_time_controller(
    models: dict[str, TLSModel], limits: SignalLimits, tls_ids=None
) -> PlanController:
    ids = tuple(tls_ids) if tls_ids is not None else tuple(models)
    return PlanController({t: fixed_time_plan(models[t], limits) for t in ids})
