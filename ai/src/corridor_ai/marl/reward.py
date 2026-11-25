"""Reward  R_i,t = -(α · W̄_i / W_ref + β · S̄_i / S_ref).

For agent i, over the decision interval Δ (averaged over its seconds):

* W_i = Σ_l W(l, t): cumulative waiting time of the vehicles on the agent's
  incoming lanes. SUMO ``lane.getWaitingTime`` sums each vehicle's current
  waiting time (time spent below 0.1 m/s since it last moved).
* S_i = Σ_l S(l, t), depending on ``stop_definition`` (open question):
    - ``halted``      number of halted vehicles (speed < 0.1 m/s) - default;
    - ``stop_events`` number of new stops (a vehicle on the lane goes from
                      moving to halted) during the second.

With ``normalize_by_lanes`` both sums are divided by the number of incoming
lanes, so intersections of different sizes give rewards on the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corridor_ai.common.config import RewardConfig

HALT_SPEED_MPS = 0.1


@dataclass
class RewardAccumulator:
    lanes: tuple[str, ...]
    cfg: RewardConfig
    w_sum: float = 0.0
    s_sum: float = 0.0
    seconds: int = 0
    stop_events_total: int = 0
    _moving: dict[str, bool] = field(default_factory=dict)

    def observe_second(self, api) -> None:
        w = sum(float(api.lane.getWaitingTime(ln)) for ln in self.lanes)
        if self.cfg.stop_definition == "halted":
            s = sum(int(api.lane.getLastStepHaltingNumber(ln)) for ln in self.lanes)
        else:  # per-vehicle polling only when this definition is selected
            s = self._count_stop_events(api)
        self.w_sum += w
        self.s_sum += s
        self.seconds += 1

    def _count_stop_events(self, api) -> int:
        events = 0
        moving_now: dict[str, bool] = {}
        for ln in self.lanes:
            for vid in api.lane.getLastStepVehicleIDs(ln):
                moving = float(api.vehicle.getSpeed(vid)) >= HALT_SPEED_MPS
                if self._moving.get(vid) is True and not moving:  # moving -> halted
                    events += 1
                moving_now[vid] = moving
        self._moving = moving_now
        self.stop_events_total += events
        return events

    def pop_reward(self) -> tuple[float, dict[str, float]]:
        n = max(self.seconds, 1)
        lanes = max(len(self.lanes), 1) if self.cfg.normalize_by_lanes else 1
        w_mean = self.w_sum / n / lanes
        s_mean = self.s_sum / n / lanes
        reward = -(
            self.cfg.alpha * w_mean / self.cfg.w_ref_s + self.cfg.beta * s_mean / self.cfg.s_ref
        )
        info = {"W_mean": w_mean, "S_mean": s_mean}
        self.w_sum = self.s_sum = 0.0
        self.seconds = 0
        return float(reward), info
