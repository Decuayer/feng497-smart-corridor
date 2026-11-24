import numpy as np
import pytest

from corridor_ai.common.config import EnvConfig, RewardConfig, SignalLimits
from corridor_ai.marl.observations import build_observation, observation_size
from corridor_ai.marl.reward import RewardAccumulator
from corridor_ai.sim.phase_machine import PhaseMachine


def test_observation_shape_range_and_content(two_phase_model):
    m = PhaseMachine("T", two_phase_model.green_states, SignalLimits())
    for _ in range(12):
        m.next_state()
    stats = {ln: (5.0, 600.0, 0.5) for ln in two_phase_model.incoming_lanes}
    obs = build_observation(two_phase_model, m, stats, 1, 3, 4, SignalLimits(), EnvConfig())
    assert obs.shape == (observation_size(4, 3),) == (26,)
    assert obs.dtype == np.float32 and obs.min() >= 0 and obs.max() <= 1
    k = 4
    assert obs[0] == pytest.approx(10 / 20)  # 2 lanes x 5 halting / (2 x capacity 10)
    assert obs[2] == 0 and obs[3] == 0  # padded phases
    assert obs[3 * k + 0] == 1 and obs[3 * k + 1] == 0  # phase one-hot
    assert obs[4 * k] == pytest.approx(12 / 60)
    assert obs[4 * k + 1] == 1 and obs[4 * k + 2] == 0  # min ok, not in transition
    assert obs[-3:].tolist() == [0, 1, 0]  # agent one-hot


class FakeAPI:
    """Minimal stand-in for libsumo with scripted lane/vehicle values."""

    def __init__(self, frames):
        self.frames, self.i = frames, 0
        api = self

        class Lane:
            def getWaitingTime(self, ln):
                return api.frames[api.i]["wait"][ln]

            def getLastStepHaltingNumber(self, ln):
                return api.frames[api.i]["halt"][ln]

            def getLastStepVehicleIDs(self, ln):
                return list(api.frames[api.i]["veh"].get(ln, {}))

        class Vehicle:
            def getSpeed(self, vid):
                for lane in api.frames[api.i]["veh"].values():
                    if vid in lane:
                        return lane[vid]
                raise KeyError(vid)

        self.lane, self.vehicle = Lane(), Vehicle()


FRAMES = [
    {"wait": {"a": 10.0, "b": 0.0}, "halt": {"a": 2, "b": 0}, "veh": {"a": {"v1": 5.0, "v2": 0.0}}},
    {
        "wait": {"a": 20.0, "b": 10.0},
        "halt": {"a": 3, "b": 1},
        "veh": {"a": {"v1": 0.0, "v2": 0.0}},
    },
]


def run_acc(cfg):
    api = FakeAPI(FRAMES)
    acc = RewardAccumulator(("a", "b"), cfg)
    for i in range(len(FRAMES)):
        api.i = i
        acc.observe_second(api)
    return acc.pop_reward()


def test_reward_halted_definition():
    r, info = run_acc(RewardConfig(alpha=1.0, beta=0.5, w_ref_s=100.0, s_ref=5.0))
    # W mean per second = (10 + 30)/2 = 20, per lane 10; S = (2 + 4)/2 = 3, per lane 1.5
    assert info["W_mean"] == pytest.approx(10.0) and info["S_mean"] == pytest.approx(1.5)
    assert r == pytest.approx(-(10.0 / 100.0 + 0.5 * 1.5 / 5.0))


def test_reward_stop_events_definition():
    r, info = run_acc(RewardConfig(stop_definition="stop_events", normalize_by_lanes=False))
    # only v1 goes moving -> halted (v2 was never seen moving)
    assert info["S_mean"] == pytest.approx(0.5)


def test_reward_is_reset_after_pop():
    api = FakeAPI(FRAMES)
    acc = RewardAccumulator(("a", "b"), RewardConfig())
    acc.observe_second(api)
    acc.pop_reward()
    assert acc.seconds == 0 and acc.w_sum == 0
