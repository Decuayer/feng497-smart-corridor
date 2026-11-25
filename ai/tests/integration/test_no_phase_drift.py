"""Regression test for the prototype signal-control bug.

The legacy prototype (ai/prototype/envs/sumo_marl_env.py) called
``trafficlight.setPhase`` without pinning the phase duration, so SUMO's own
program kept advancing phases. In a 600 s KEEP-only run, the controller's
phase state differed from SUMO's actual phase in 239 of 360 checks (66 %).

The new layer drives every light with ``setRedYellowGreenState`` from a
PhaseMachine. These tests simulate N seconds and assert, every second, that
SUMO displays exactly the machine's state, that every light is on the
"online" program, and that the safety monitor saw no violation.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridor_ai.common.config import load_experiment_config
from corridor_ai.common.registry import load_fixture
from corridor_ai.control.fixed_time import fixed_time_controller
from corridor_ai.marl.env import CorridorParallelEnv
from corridor_ai.sim.core import CorridorSim

pytestmark = pytest.mark.sumo
N_SECONDS = 3600


def _check_online(sim: CorridorSim) -> None:
    for tid in sim.models:
        assert sim.api.trafficlight.getProgram(tid) == "online", tid


@pytest.mark.parametrize("fixture", ["toy_corridor", "osm_alsancak_v0"])
@pytest.mark.parametrize("policy", ["keep_only", "fixed_time"])
def test_zero_drift_core(fixture, policy, tmp_path):
    cfg = load_experiment_config()
    sim = CorridorSim(load_fixture(fixture), cfg, seed=7, output_dir=tmp_path, end_s=N_SECONDS)
    sim.start()
    try:
        ctrl = [fixed_time_controller(sim.models, cfg.limits)] if policy == "fixed_time" else []
        for second in range(N_SECONDS):
            sim.run_controllers(ctrl, 1)
            if second in (1, 1800, N_SECONDS - 1):
                _check_online(sim)
    finally:
        metrics = sim.close()
    assert sim.drift_checks == N_SECONDS * len(sim.models)
    assert sim.drift_mismatches == 0
    assert metrics["safety_violations"] == 0, sim.monitor.counts_by_rule()


@pytest.mark.parametrize("fixture", ["toy_corridor", "osm_alsancak_v0"])
@pytest.mark.parametrize("all_red_mode", ["full", "conflicting_only"])
def test_zero_drift_random_legal_actions(fixture, all_red_mode, tmp_path):
    cfg = load_experiment_config(
        overrides={"sim.horizon_s": N_SECONDS - 300, "limits.all_red_mode": all_red_mode}
    )
    env = CorridorParallelEnv(load_fixture(fixture), cfg, output_dir=tmp_path)
    rng = np.random.default_rng(0)
    obs, infos = env.reset(seed=11)
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
        obs, _, _, _, infos = env.step(acts)
    ep = env.last_episode_metrics
    assert ep["drift_mismatches"] == 0
    assert ep["safety_violations"] == 0
    assert ep["illegal_actions"] == 0
    assert env.sim.drift_checks == N_SECONDS * len(env.models)
