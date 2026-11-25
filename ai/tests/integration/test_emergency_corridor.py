"""Emergency corridor in SUMO: preemption, safe handback, logs and commands."""

from __future__ import annotations

import json

import numpy as np
import pytest

from corridor_ai.common.config import load_experiment_config
from corridor_ai.common.contracts import validate
from corridor_ai.common.registry import load_fixture
from corridor_ai.emergency.evaluate import run_fixed_time
from corridor_ai.emergency.intake import build_request_message, new_request_id
from corridor_ai.emergency.preemption import PreemptionManager
from corridor_ai.marl.env import CorridorParallelEnv
from corridor_ai.sim.core import CorridorSim

pytestmark = pytest.mark.sumo


def _cfg(**over):
    return load_experiment_config(
        "ai/configs/emergency.yaml",
        overrides={
            "sim.horizon_s": 1500,
            "emergency.ev_count": 3,
            "emergency.ev_last_before_end_s": 600,
            **over,
        },
    )


@pytest.mark.parametrize("fixture", ["toy_corridor", "osm_corridor"])
def test_preemption_speeds_up_evs_and_stays_safe(fixture, tmp_path):
    cfg = _cfg()
    fx = load_fixture(fixture)
    base = run_fixed_time(fx, cfg, 2002, 1.0, False, tmp_path / "b")
    log = tmp_path / "p" / "commands.jsonl"
    pre = run_fixed_time(fx, cfg, 2002, 1.0, True, tmp_path / "p", command_log=log)
    for r in (base, pre):
        assert r["safety_violations"] == 0 and r["drift_mismatches"] == 0
        assert len(r["ev"]) == 3 and all(e["arrived"] for e in r["ev"])
    assert np.mean([e["duration_s"] for e in pre["ev"]]) < np.mean(
        [e["duration_s"] for e in base["ev"]]
    )
    assert pre["activations"] and len(pre["releases"]) == len(pre["activations"])
    assert {a["intersection_id"] for a in pre["activations"]} == set(
        fx.agents
    )  # all six pre-cleared
    assert max(a["lookahead_rank"] for a in pre["activations"]) <= cfg.emergency.lookahead
    assert not base["activations"]  # baseline only observes
    # logs and commands are contract-valid
    kinds = {e["event_type"] for e in pre["events"]}
    assert {"emergency_request_received", "preemption_started", "preemption_ended"} <= kinds
    for e in pre["events"]:
        validate("system_log_event", e)
    for m in pre["requests"]:
        validate("emergency_request", m)
    cmds = [json.loads(x) for x in log.read_text().splitlines()]
    modes = {c["mode"] for c in cmds}
    assert "preemption" in modes and "recovery" in modes
    for c in cmds:
        validate("signal_command", c)
        if c["mode"] == "preemption":
            assert c["issuer"] == "emergency_preemption" and "emergency_request_id" in c


def test_marl_env_handback_and_masks(tmp_path):
    cfg = _cfg()
    env = CorridorParallelEnv(
        load_fixture("toy_corridor"), cfg, output_dir=tmp_path, ev_scenario=True, preempt=True
    )
    rng = np.random.default_rng(0)
    obs, infos = env.reset(seed=2003)
    saw_preempted = False
    while env.agents:
        for a in env.agents:
            if infos[a].get("preempted"):
                saw_preempted = True
                assert infos[a]["action_mask"].tolist() == [True] + [False] * env.k_max
        acts = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
        obs, _, _, _, infos = env.step(acts)
    ep = env.last_episode_metrics
    assert saw_preempted and ep["preemptions"] > 0
    assert (
        ep["safety_violations"] == 0 and ep["drift_mismatches"] == 0 and ep["illegal_actions"] == 0
    )


def test_external_message_is_tracked_and_unverified_rejected(tmp_path):
    cfg = _cfg(**{"emergency.ev_count": 0})
    fx = load_fixture("toy_corridor")
    sim = CorridorSim(fx, cfg, seed=5, output_dir=tmp_path)
    sim.start()
    try:
        man = PreemptionManager(fx, cfg, enabled=True)
        sim.second_hooks.append(man.update)
        sim.run_controllers([], 400)
        # pick an ordinary eastbound vehicle and announce it as a verified V2I emergency request
        vid = next(
            v for v in sim.api.vehicle.getIDList() if sim.api.vehicle.getRoute(v)[-1] == "J6_E"
        )
        msg = build_request_message(
            new_request_id(), "amb-77", "ambulance", "request", sim.time_s, [], sumo_vehicle_id=vid
        )
        msg.update(source="v2i_lora", auth={"verified": True, "method": "hmac_sha256"})
        bad = dict(
            msg, request_id=new_request_id(), auth={"verified": False, "method": "hmac_sha256"}
        )
        assert man.submit_message(msg, sim) and vid in man.evs
        assert not man.submit_message(bad, sim) and man.rejected
        sim.run_controllers([], 5)
        assert man.evs[vid].targets
    finally:
        sim.close()


def test_hold_budget_releases_and_stays_safe(tmp_path):
    """A tiny hold budget forces hold_timeout releases; EVs still arrive and limits hold."""
    cfg = _cfg(**{"emergency.max_hold_s": 12})
    fx = load_fixture("toy_corridor")
    r = run_fixed_time(fx, cfg, 2002, 1.5, True, tmp_path)
    reasons = {x["reason"] for x in r["releases"]}
    assert "hold_timeout" in reasons
    assert all(x["duration_s"] <= 12 + 1 for x in r["releases"] if x["reason"] == "hold_timeout")
    assert r["safety_violations"] == 0 and r["drift_mismatches"] == 0
    assert all(e["arrived"] for e in r["ev"])
