"""Webster baseline: optimal cycle and green splits from measured flows.

For each traffic light with K green phases:

    y_k = max over lanes l served by phase k of  q_l / s          (critical flow ratio)
    Y   = sum_k y_k                    (capped at y_max)
    L   = K * lost_time_per_phase                                (yellow + all-red)
    C0  = (1.5 L + 5) / (1 - Y)                                  (Webster's optimum)
    C   = clamp(C0, K*min_green + L, min(cycle_max, K*max_green + L))
    g_k = (C - L) * y_k / Y            clamped to [min_green, max_green]

Lane flows q_l come from a calibration run under the fixed-time plan on a
calibration seed (eval seed + offset), never the evaluation seed itself.
ASSUMPTIONS until analysis/ (Ayça Avcı) provides parameters: saturation flow
s = 1800 veh/h/lane; lost time per phase = yellow + all-red (start-up loss
equal to end gain); a lane served by several phases counts towards each of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corridor_ai.common.config import ExperimentConfig, SignalLimits, load_yaml
from corridor_ai.common.paths import CONFIG_DIR
from corridor_ai.common.registry import Fixture
from corridor_ai.control.fixed_time import PlanController, SignalPlan, fixed_time_controller
from corridor_ai.sim.tls_model import TLSModel


@dataclass(frozen=True)
class WebsterParams:
    saturation_flow_vphpl: float = 1800.0
    lost_time_per_phase_s: float = 4.0
    cycle_max_s: float = 150.0
    y_max: float = 0.95
    calibration_seed_offset: int = 100000
    calibration_duration_s: int = 3600

    @classmethod
    def from_config(cls, path: Path = CONFIG_DIR / "baselines.yaml") -> WebsterParams:
        return cls(**load_yaml(path).get("webster", {}))


def webster_plan(
    model: TLSModel,
    lane_flows_vph: dict[str, float],
    limits: SignalLimits,
    params: WebsterParams,
) -> tuple[SignalPlan, dict[str, Any]]:
    k = model.n_phases
    if k < 2:
        plan = SignalPlan(model.tls_id, (limits.max_green_s,), limits.yellow_s, limits.all_red_s)
        return plan, {"note": "single phase"}
    s = params.saturation_flow_vphpl
    y = [
        max((lane_flows_vph.get(lane, 0.0) / s for lane in lanes), default=0.0)
        for lanes in model.lanes_by_phase
    ]
    y_sum_raw = sum(y)
    y_sum = min(y_sum_raw, params.y_max)
    lost = k * params.lost_time_per_phase_s
    c_min = k * limits.min_green_s + lost
    c_max = min(params.cycle_max_s, k * limits.max_green_s + lost)
    c0 = (1.5 * lost + 5.0) / (1.0 - y_sum) if y_sum < 1.0 else float("inf")
    cycle = min(max(c0, c_min), c_max)
    effective = cycle - lost
    if y_sum_raw > 0:
        raw = [effective * yk / y_sum_raw for yk in y]
    else:
        raw = [effective / k] * k
    greens = tuple(int(min(max(round(g), limits.min_green_s), limits.max_green_s)) for g in raw)
    plan = SignalPlan(model.tls_id, greens, limits.yellow_s, limits.all_red_s)
    details = {
        "y": [round(v, 4) for v in y],
        "Y": round(y_sum_raw, 4),
        "L_s": lost,
        "C0_s": round(c0, 1) if c0 != float("inf") else None,
        "C_s": round(cycle, 1),
        **plan.as_dict(),
    }
    return plan, details


def measure_lane_flows(
    fixture: Fixture, cfg: ExperimentConfig, seed: int, scale: float | None = None
) -> dict[str, float]:
    """Run the fixed-time plan and count distinct vehicles per incoming lane (veh/h).

    Counting starts after the warm-up and lasts for the calibration duration.
    """
    from corridor_ai.sim.core import CorridorSim  # local import avoids a cycle

    params = WebsterParams.from_config()
    end = cfg.sim.warmup_s + params.calibration_duration_s
    sim = CorridorSim(fixture, cfg, seed=seed, output_dir=None, scale=scale, end_s=end)
    sim.start()
    lanes = sorted({lane for m in sim.models.values() for lane in m.incoming_lanes})
    seen: dict[str, set[str]] = {lane: set() for lane in lanes}

    def count(s: CorridorSim) -> None:
        if s.time_s <= cfg.sim.warmup_s:
            return
        for lane in lanes:
            seen[lane].update(s.api.lane.getLastStepVehicleIDs(lane))

    sim.second_hooks.append(count)
    controller = fixed_time_controller(sim.models, cfg.limits)
    try:
        while not sim.done:
            sim.run_controllers([controller], 1)
    finally:
        sim.close()
    hours = params.calibration_duration_s / 3600.0
    return {lane: len(ids) / hours for lane, ids in seen.items()}


def webster_controller(
    models: dict[str, TLSModel],
    lane_flows_vph: dict[str, float],
    limits: SignalLimits,
    params: WebsterParams | None = None,
    tls_ids=None,
) -> tuple[PlanController, dict[str, dict[str, Any]]]:
    params = params or WebsterParams.from_config()
    ids = tuple(tls_ids) if tls_ids is not None else tuple(models)
    plans, details = {}, {}
    for tid in ids:
        plans[tid], details[tid] = webster_plan(models[tid], lane_flows_vph, limits, params)
    return PlanController(plans), details
