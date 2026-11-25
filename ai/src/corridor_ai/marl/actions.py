"""Action space and action masking.

Action a ∈ {0, 1, ..., K_max}:

* a = 0      KEEP: extend the current green by one decision interval Δ;
* a = k ≥ 1  SWITCH to green phase k-1. Yellow (3 s) and all-red (1 s) are
             inserted by the PhaseMachine.

"Shorten" in the task description means switching before the nominal split.

Mask rules, evaluated at a decision point with green elapsed t:

1. during a transition (yellow / all-red) only KEEP is legal (it is a no-op);
2. a light with one green phase: only KEEP;
3. SWITCH is illegal while t < min_green; switching to the current phase or to
   a padded (non-existent) phase is always illegal;
4. KEEP is illegal if t + Δ > max_green, so the agent must choose a switch;
5. phase_order = cyclic: only KEEP or the next phase in program order;
6. starvation guard (max_red_s): once min green is met, if some phase has
   waited longer than max_red_s, the only legal action is switching to the
   most-starved phase (ASSUMPTION: 120 s).

At least one action is always legal.
"""

from __future__ import annotations

import numpy as np

from corridor_ai.common.config import EnvConfig, SignalLimits
from corridor_ai.sim.phase_machine import PhaseMachine

KEEP = 0


def action_to_request(action: int, machine: PhaseMachine) -> int | None:
    """Translate an action into a PhaseMachine request (None = keep)."""
    if action == KEEP:
        return None
    return action - 1


def action_mask(
    machine: PhaseMachine, k_max: int, limits: SignalLimits, env_cfg: EnvConfig
) -> np.ndarray:
    mask = np.zeros(k_max + 1, dtype=bool)
    mask[KEEP] = True
    k = machine.n_phases
    if machine.in_transition or k < 2:
        return mask

    t = machine.green_elapsed
    dt = env_cfg.decision_interval_s
    can_switch = t >= limits.min_green_s
    if can_switch:
        if env_cfg.phase_order == "cyclic":
            mask[1 + (machine.phase + 1) % k] = True
        else:
            for p in range(k):
                if p != machine.phase:
                    mask[1 + p] = True
        if t + dt > limits.max_green_s:
            mask[KEEP] = False

        if env_cfg.max_red_s is not None:
            waits = [(machine.time_since_served[p], p) for p in range(k) if p != machine.phase]
            worst_wait, worst = max(waits)
            if worst_wait > env_cfg.max_red_s:
                mask[:] = False
                mask[1 + worst] = True
    return mask
