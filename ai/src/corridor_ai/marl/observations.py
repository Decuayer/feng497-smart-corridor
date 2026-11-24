"""Per-agent observation vector (fixed size, so one network can serve every agent).

Features are aggregated **per green phase** rather than per lane. Intersections
with different lane counts then share one input size, and feature k lines up
with action k+1 ("switch to phase k"). With K = K_max and N = number of agents:

    [ queue_k     (K)  Σ halting vehicles on lanes served by phase k / Σ lane capacity
      wait_k      (K)  Σ waiting time on those lanes / (Σ capacity · wait_norm_s)
      occ_k       (K)  mean lane occupancy
      phase_k     (K)  one-hot of the green currently shown (all zero in transition)
      t_phase     (1)  green elapsed / max_green
      min_ok      (1)  1 if min green is satisfied
      in_trans    (1)  1 during yellow / all-red
      starve_k    (K)  time since phase k was last green / max_red_ref
      agent_id    (N)  one-hot agent index (parameter sharing) ]

All entries lie in [0, 1]. Padded phases (k >= the light's own K) are 0.
Lane capacity = lane length / vehicle_length_m (7.5 m).
"""

from __future__ import annotations

import numpy as np

from corridor_ai.common.config import EnvConfig, SignalLimits
from corridor_ai.sim.phase_machine import PhaseMachine
from corridor_ai.sim.tls_model import TLSModel


def observation_size(k_max: int, n_agents: int) -> int:
    return 5 * k_max + 3 + n_agents


def lane_capacity(model: TLSModel, lane: str, vehicle_length_m: float) -> float:
    return max(model.lane_lengths.get(lane, vehicle_length_m) / vehicle_length_m, 1.0)


def build_observation(
    model: TLSModel,
    machine: PhaseMachine,
    lane_stats: dict[str, tuple[float, float, float]],  # lane -> (halting, waiting_s, occupancy)
    agent_index: int,
    n_agents: int,
    k_max: int,
    limits: SignalLimits,
    env_cfg: EnvConfig,
) -> np.ndarray:
    queue = np.zeros(k_max, dtype=np.float32)
    wait = np.zeros(k_max, dtype=np.float32)
    occ = np.zeros(k_max, dtype=np.float32)
    for k, lanes in enumerate(model.lanes_by_phase[:k_max]):
        if not lanes:
            continue
        cap = sum(lane_capacity(model, ln, env_cfg.vehicle_length_m) for ln in lanes)
        queue[k] = sum(lane_stats[ln][0] for ln in lanes) / cap
        wait[k] = sum(lane_stats[ln][1] for ln in lanes) / (cap * env_cfg.wait_norm_s)
        occ[k] = sum(lane_stats[ln][2] for ln in lanes) / len(lanes)

    phase = np.zeros(k_max, dtype=np.float32)
    if machine.current_green is not None and machine.current_green < k_max:
        phase[machine.current_green] = 1.0
    timing = np.array(
        [
            machine.green_elapsed / limits.max_green_s if not machine.in_transition else 0.0,
            1.0 if machine.min_green_satisfied else 0.0,
            1.0 if machine.in_transition else 0.0,
        ],
        dtype=np.float32,
    )
    starve_ref = float(env_cfg.max_red_s or 120)
    starve = np.zeros(k_max, dtype=np.float32)
    for k in range(min(machine.n_phases, k_max)):
        starve[k] = machine.time_since_served[k] / starve_ref
    agent = np.zeros(n_agents, dtype=np.float32)
    agent[agent_index] = 1.0
    obs = np.concatenate([queue, wait, occ, phase, timing, starve, agent])
    return np.clip(obs, 0.0, 1.0).astype(np.float32)
