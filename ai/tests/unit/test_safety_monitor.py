"""SafetyMonitor detects each rule violation (no SUMO)."""

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.safety_monitor import SafetyMonitor

NS, EW, Y, R = "GGrr", "rrGG", "yyrr", "rrrr"


def feed(model, runs, limits=None):
    limits = limits or SignalLimits()
    mon = SafetyMonitor({model.tls_id: model}, limits)
    t = 0
    for state, n in runs:
        for _ in range(n):
            t += 1
            mon.observe(t, {model.tls_id: state})
    mon.finalize(t)
    return mon


def rules(mon):
    return sorted(mon.counts_by_rule())


def test_compliant_sequence(two_phase_model):
    mon = feed(two_phase_model, [(NS, 10), (Y, 3), (R, 1), (EW, 60), ("rryy", 3), (R, 1), (NS, 5)])
    assert mon.violations == []  # final run truncated -> no min-green check


def test_short_green(two_phase_model):
    assert rules(feed(two_phase_model, [(NS, 9), (Y, 3), (R, 1), (EW, 20)])) == ["min_green"]


def test_long_green(two_phase_model):
    assert rules(feed(two_phase_model, [(NS, 61), (Y, 3), (R, 1), (EW, 20)])) == ["max_green"]


def test_wrong_yellow(two_phase_model):
    assert "yellow" in rules(feed(two_phase_model, [(NS, 10), (Y, 6), (R, 1), (EW, 20)]))


def test_missing_or_long_all_red(two_phase_model):
    assert "sequence" in rules(feed(two_phase_model, [(NS, 10), (Y, 3), (EW, 20)]))
    assert "all_red" in rules(feed(two_phase_model, [(NS, 10), (Y, 3), (R, 2), (EW, 20)]))


def test_direct_green_to_green_and_green_to_red(two_phase_model):
    r = rules(feed(two_phase_model, [(NS, 10), (EW, 20)]))
    assert "sequence" in r and "green_to_red" in r


def test_conflicting_greens(two_phase_model):
    mon = feed(two_phase_model, [("GGGG", 1)])
    assert "conflict" in rules(mon)


def test_permissive_g_is_not_a_conflict(two_phase_model):
    assert "conflict" not in rules(feed(two_phase_model, [("GGgg", 1)]))


def test_all_red_must_be_red_in_full_mode(three_phase_model):
    m = three_phase_model
    seq = [(m.green_states[0], 10), ("yygrr", 3), ("rrgrr", 1), (m.green_states[1], 20)]
    assert "all_red" in rules(feed(m, seq))
    assert feed(m, seq, SignalLimits(all_red_mode="conflicting_only")).violations == []


def test_single_phase_exempt_from_max_green():
    from tests.conftest import make_model

    m = make_model(["GG"])
    assert feed(m, [("GG", 500)]).violations == []
