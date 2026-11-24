"""PhaseMachine timing rules (no SUMO)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.phase_machine import PhaseMachine, all_red_state, yellow_state
from corridor_ai.sim.safety_monitor import SafetyMonitor
from tests.conftest import make_model

NS, EW = "GGrr", "rrGG"


def run(machine, seconds, requests=None):
    """Return the displayed states; requests: {second: target}."""
    out = []
    for t in range(seconds):
        if requests and t in requests:
            machine.request(requests[t])
        out.append(machine.next_state())
    return out


def runs(states):
    """Run-length encode a list of states."""
    out = []
    for s in states:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return [(s, n) for s, n in out]


def test_switch_sequence_full_all_red(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    states = run(m, 30, {10: 1})
    assert runs(states) == [(NS, 10), ("yyrr", 3), ("rrrr", 1), (EW, 16)]


def test_early_request_is_deferred_until_min_green(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    states = run(m, 20, {2: 1})
    assert runs(states)[0] == (NS, 10)  # switch only after 10 s of green


def test_max_green_forces_cyclic_switch(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    states = run(m, 70)
    assert runs(states)[:3] == [(NS, 60), ("yyrr", 3), ("rrrr", 1)]
    assert m.switches == 1


def test_keep_clears_pending_request(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    states = run(m, 15, {2: 1, 5: None})
    assert set(states) == {NS}


def test_request_during_transition_is_served_after(limits):
    m = PhaseMachine("T", ["GGrrrr", "rrGGrr", "rrrrGG"], limits)
    states = run(m, 40, {10: 1, 11: 2})
    r = runs(states)
    # goes to phase 1 first (min green), then to phase 2 after 10 s
    assert r[3] == ("rrGGrr", 10)
    assert r[6][0] == "rrrrGG"


def test_single_phase_stays_green(limits):
    m = PhaseMachine("T", ["GGG"], limits)
    assert set(run(m, 200, {5: 0})) == {"GGG"}


def test_invalid_phase_rejected(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    with pytest.raises(ValueError):
        m.request(2)


def test_yellow_and_all_red_states_modes():
    assert yellow_state("GGgrr", "rrGrr", "full") == "yyyrr"
    assert yellow_state("GGgrr", "rrGrr", "conflicting_only") == "yygrr"
    assert all_red_state("GGgrr", "rrGrr", "full") == "rrrrr"
    assert all_red_state("GGgrr", "rrGrr", "conflicting_only") == "rrgrr"
    # protected -> permissive downgrade counts as losing right-of-way
    assert yellow_state("rrGrr", "GGgrr", "conflicting_only") == "rryrr"
    # nothing loses green -> fall back to the full sequence
    assert yellow_state("GGrr", "GGGr", "conflicting_only") == "yyrr"
    assert all_red_state("GGrr", "GGGr", "conflicting_only") == "rrrr"


def test_time_since_served(limits):
    m = PhaseMachine("T", [NS, EW], limits)
    run(m, 12)
    assert m.time_since_served[0] == 0 and m.time_since_served[1] == 12


@settings(max_examples=150, deadline=None)
@given(
    data=st.lists(st.one_of(st.none(), st.integers(0, 2)), min_size=10, max_size=120),
    every=st.integers(1, 7),
    mode=st.sampled_from(["full", "conflicting_only"]),
)
def test_property_any_request_sequence_is_safe(data, every, mode):
    """Whatever a controller requests, the displayed sequence satisfies every rule."""
    limits = SignalLimits(all_red_mode=mode)
    model = make_model(["GGgrr", "rrGrr", "rrrGG"], foes={(0, 3), (1, 4), (2, 3)})
    m = PhaseMachine(model.tls_id, model.green_states, limits)
    mon = SafetyMonitor({model.tls_id: model}, limits)
    t = 0
    for req in data:
        m.request(req)
        for _ in range(every):
            t += 1
            mon.observe(t, {model.tls_id: m.next_state()})
    mon.finalize(t)
    assert mon.violations == []
