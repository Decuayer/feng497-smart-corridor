from hypothesis import given, settings
from hypothesis import strategies as st

from corridor_ai.common.config import EnvConfig, SignalLimits
from corridor_ai.marl.actions import KEEP, action_mask, action_to_request
from corridor_ai.sim.phase_machine import PhaseMachine

STATES = ["GGrrrr", "rrGGrr", "rrrrGG"]


def machine_at(green_elapsed, limits=None, states=STATES):
    m = PhaseMachine("T", states, limits or SignalLimits())
    for _ in range(green_elapsed):
        m.next_state()
    return m


def mask(m, k_max=4, **env):
    return action_mask(m, k_max, SignalLimits(), EnvConfig(**env))


def test_before_min_green_only_keep():
    assert mask(machine_at(5)).tolist() == [True, False, False, False, False]


def test_after_min_green_free_order():
    assert mask(machine_at(10)).tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]  # padded phase 3 illegal


def test_keep_illegal_near_max_green():
    assert mask(machine_at(56)).tolist() == [False, False, True, True, False]


def test_cyclic_order():
    assert mask(machine_at(20), phase_order="cyclic").tolist() == [True, False, True, False, False]


def test_transition_only_keep():
    m = machine_at(10)
    m.request(1)
    m.next_state()  # yellow
    assert m.in_transition
    assert mask(m).tolist() == [True, False, False, False, False]


def test_single_phase_only_keep():
    m = machine_at(100, states=["GG"])
    assert mask(m).tolist() == [True, False, False, False, False]


def test_starvation_guard_forces_most_starved_phase():
    m = machine_at(10)
    m.time_since_served = [0, 50, 130]
    assert mask(m).tolist() == [False, False, False, True, False]
    assert mask(m, max_red_s=None).tolist() == [True, False, True, True, False]


def test_action_to_request():
    m = machine_at(10)
    assert action_to_request(KEEP, m) is None
    assert action_to_request(3, m) == 2


@settings(max_examples=200, deadline=None)
@given(
    t=st.integers(0, 70),
    dt=st.integers(1, 10),
    order=st.sampled_from(["free", "cyclic"]),
    starve=st.lists(st.integers(0, 300), min_size=3, max_size=3),
)
def test_property_at_least_one_legal_action(t, dt, order, starve):
    m = machine_at(min(t, 59))
    m.time_since_served = starve
    mk = action_mask(m, 4, SignalLimits(), EnvConfig(decision_interval_s=dt, phase_order=order))
    assert mk.any() and not mk[1 + m.phase] and not mk[4]
