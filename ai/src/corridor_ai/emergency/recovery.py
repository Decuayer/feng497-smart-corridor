"""Recovery after an emergency preemption, measured against the matched run.

A per-second queue series (halted vehicles on all incoming lanes) is recorded
for every agent intersection, both in the preemption run and in the matched
run without preemption (same fixture, seed, demand, controller and EVs).
For each release at time t_r, the **recovery time** is the smallest τ ≥ 0 such
that, for ``sustain_s`` consecutive seconds starting at t_r + τ, the intersection
is no longer worse off than without preemption:

    Q_pre(t) - Q_base(t) <= max(tolerance_frac * Q_base(t), tolerance_abs)

(one-sided: a queue *shorter* than in the baseline counts as recovered)
where both series are smoothed with a centred moving average of
``smooth_s`` seconds (two stochastic runs never match exactly). If the window
closes before that (the next preemption at the same intersection, or the end
of the run), the release is **censored**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from corridor_ai.common.config import EmergencyConfig

if TYPE_CHECKING:
    from corridor_ai.sim.core import CorridorSim


@dataclass
class QueueRecorder:
    """Hook: per-second total queue for each agent intersection."""

    agent_tls: dict[str, str]  # contract id -> TLS id
    start_s: float = 0.0
    times: list[float] = field(default_factory=list)
    series: dict[str, list[int]] = field(default_factory=dict)

    def __call__(self, sim: CorridorSim) -> None:
        if sim.time_s < self.start_s:
            return
        self.times.append(sim.time_s)
        for cid, tls in self.agent_tls.items():
            q = sum(sim.lane_halting(ln) for ln in sim.models[tls].incoming_lanes)
            self.series.setdefault(cid, []).append(q)

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return np.asarray(self.times), {k: np.asarray(v, float) for k, v in self.series.items()}


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) == 0:
        return x.astype(float)
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x.astype(float), (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def recovery_time(
    times: np.ndarray,
    q_pre: np.ndarray,
    q_base: np.ndarray,
    release_s: float,
    window_end_s: float,
    cfg: EmergencyConfig,
) -> float | None:
    """Recovery time in seconds after `release_s`, or None if censored."""
    qp, qb = smooth(q_pre, cfg.recovery_smooth_s), smooth(q_base, cfg.recovery_smooth_s)
    ok = (qp - qb) <= np.maximum(cfg.recovery_tolerance_frac * qb, cfg.recovery_tolerance_abs)
    start = int(np.searchsorted(times, release_s))
    end = int(np.searchsorted(times, window_end_s))
    n = cfg.recovery_sustain_s
    run = 0
    for i in range(start, min(end, len(ok))):
        run = run + 1 if ok[i] else 0
        if run >= n:
            return float(times[i - n + 1] - release_s)
    return None


def recoveries(
    times: np.ndarray,
    pre: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    releases: list[dict],
    activations: list[dict],
    cfg: EmergencyConfig,
) -> list[dict]:
    out = []
    end_of_run = float(times[-1]) if len(times) else 0.0
    for rel in releases:
        cid, t_r = rel["intersection_id"], rel["release_s"]
        nxt = [
            a["start_s"] for a in activations if a["intersection_id"] == cid and a["start_s"] > t_r
        ]
        window_end = min(nxt) if nxt else end_of_run
        tau = recovery_time(times, pre[cid], base[cid], t_r, window_end, cfg)
        out.append({**rel, "recovery_s": tau, "censored": tau is None,
                    "window_s": window_end - t_r})  # fmt: skip
    return out
