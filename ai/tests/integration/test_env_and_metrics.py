"""Environment API, determinism, trip counting, baselines and command output."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from corridor_ai.common.config import load_experiment_config
from corridor_ai.common.contracts import validate
from corridor_ai.common.registry import load_fixture
from corridor_ai.control.fixed_time import fixed_time_controller
from corridor_ai.control.webster import WebsterParams, measure_lane_flows, webster_controller
from corridor_ai.marl.env import CorridorParallelEnv
from corridor_ai.sim.core import CorridorSim
from corridor_ai.sim.metrics import count_tripinfo_arrivals

pytestmark = pytest.mark.sumo


def _cfg(**over):
    return load_experiment_config(overrides={"sim.horizon_s": 600, **over})


@pytest.mark.parametrize("fixture", ["toy_corridor", "osm_alsancak_v0"])
def test_parallel_api(fixture, tmp_path):
    env = CorridorParallelEnv(load_fixture(fixture), _cfg(), output_dir=tmp_path)
    parallel_api_test(env, num_cycles=40)
    env.close()


def _random_episode(tmp_path, seed, action_seed):
    env = CorridorParallelEnv(load_fixture("toy_corridor"), _cfg(), output_dir=tmp_path)
    rng = np.random.default_rng(action_seed)
    obs, infos = env.reset(seed=seed)
    total = 0.0
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
        obs, rew, _, _, infos = env.step(acts)
        total += sum(rew.values())
    return total, env.last_episode_metrics


def test_determinism_same_seed(tmp_path):
    r1, m1 = _random_episode(tmp_path / "a", 5, 1)
    r2, m2 = _random_episode(tmp_path / "b", 5, 1)
    r3, m3 = _random_episode(tmp_path / "c", 6, 1)
    assert r1 == r2 and m1["mean_travel_time_s"] == m2["mean_travel_time_s"]
    assert m1["mean_travel_time_s"] != m3["mean_travel_time_s"]


def test_cumulative_arrivals_match_tripinfo(tmp_path):
    cfg = _cfg()
    sim = CorridorSim(load_fixture("toy_corridor"), cfg, seed=3, output_dir=tmp_path)
    sim.start()
    ctrl = fixed_time_controller(sim.models, cfg.limits)
    while not sim.done:
        sim.run_controllers([ctrl], 1)
    m = sim.close()
    assert m["arrived_cumulative"] > 100
    assert m["arrived_cumulative"] == count_tripinfo_arrivals(sim.tripinfo_path)


def test_webster_controller_is_safe(tmp_path):
    cfg = _cfg()
    fx = load_fixture("toy_corridor")
    flows = measure_lane_flows(fx, cfg, seed=100003)
    assert sum(flows.values()) > 0
    sim = CorridorSim(fx, cfg, seed=3, output_dir=tmp_path)
    sim.start()
    ctrl, details = webster_controller(sim.models, flows, cfg.limits, WebsterParams())
    while not sim.done:
        sim.run_controllers([ctrl], 1)
    m = sim.close()
    assert m["safety_violations"] == 0 and m["drift_mismatches"] == 0
    assert all(10 <= g <= 60 for d in details.values() for g in d["greens_s"])


def test_command_log_is_schema_valid(tmp_path):
    log = tmp_path / "commands.jsonl"
    env = CorridorParallelEnv(
        load_fixture("toy_corridor"), _cfg(), output_dir=tmp_path, command_log=log
    )
    rng = np.random.default_rng(0)
    _, infos = env.reset(seed=2)
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
        _, _, _, _, infos = env.step(acts)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines, "expected at least one signal_command"
    seqs: dict[str, int] = {}
    for msg in lines:
        validate("signal_command", msg)
        assert msg["sequence"] > seqs.get(msg["intersection_id"], 0)
        seqs[msg["intersection_id"]] = msg["sequence"]


def test_dqn_training_smoke(tmp_path, monkeypatch):
    import corridor_ai.marl.train as tr
    from corridor_ai.common.config import load_experiment_config as load

    monkeypatch.setattr(tr, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tr, "CHECKPOINT_DIR", tmp_path / "ckpt")
    cfg = load(
        "ai/configs/marl_dqn.yaml",
        overrides={
            "sim.horizon_s": 300,
            "dqn.episodes": 2,
            "dqn.learning_starts": 100,
            "dqn.eval_every": 1,
            "dqn.checkpoint_every": 0,
            "name": "t",
        },
    )
    out = tr.train(cfg, "smoke")
    assert out["checkpoint"].exists()
    lines = (out["run_dir"] / "episodes.csv").read_text().splitlines()
    assert len(lines) == 3 and "td_loss" in lines[0]
