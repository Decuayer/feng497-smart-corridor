import pytest

from corridor_ai.common.config import SignalLimits
from corridor_ai.common.registry import load_fixture
from corridor_ai.control.fixed_time import fixed_time_plan
from corridor_ai.control.webster import WebsterParams, webster_plan
from corridor_ai.sim.demand import write_toy_routes
from corridor_ai.sim.metrics import count_tripinfo_arrivals, parse_tripinfo
from tests.conftest import make_model


def test_webster_textbook_case():
    # 2 phases, 1 lane each: q = 900 and 450 veh/h, s = 1800 -> y = 0.5, 0.25, Y = 0.75
    m = make_model(["Gr", "rG"])
    plan, d = webster_plan(m, {"L0_0": 900.0, "L1_0": 450.0}, SignalLimits(), WebsterParams())
    assert d["Y"] == pytest.approx(0.75)
    assert d["C0_s"] == pytest.approx((1.5 * 8 + 5) / 0.25)  # 68 s
    # effective green 60 s split 2:1 -> 40 / 20
    assert plan.greens_s == (40, 20) and plan.cycle_s == 68


def test_webster_respects_limits_and_cycle_cap():
    m = make_model(["Gr", "rG"])
    plan, d = webster_plan(m, {"L0_0": 1700.0, "L1_0": 10.0}, SignalLimits(), WebsterParams())
    assert all(10 <= g <= 60 for g in plan.greens_s)
    assert d["C_s"] <= 128  # min(cycle_max 150, 2*60 + 8)


def test_fixed_time_plan_clamps_original_durations():
    m = make_model(["Gr", "rG", "GG"])
    m = m.__class__(**{**m.__dict__, "planned_green_s": (6.0, 42.0, 90.0)})
    assert fixed_time_plan(m, SignalLimits()).greens_s == (10, 42, 60)


def test_toy_routes_are_deterministic_and_scale(tmp_path):
    a = write_toy_routes(tmp_path / "a.xml", 1.0, seed=5, end_s=1800).read_text()
    b = write_toy_routes(tmp_path / "b.xml", 1.0, seed=5, end_s=1800).read_text()
    c = write_toy_routes(tmp_path / "c.xml", 1.0, seed=6, end_s=1800).read_text()
    d = write_toy_routes(tmp_path / "d.xml", 1.5, seed=5, end_s=1800).read_text()
    assert a == b and a != c
    assert d.count("<trip ") > 1.3 * a.count("<trip ")
    assert load_fixture("toy_corridor").corridor_edges[0] == "W_J1"


def test_parse_tripinfo(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text(
        "<tripinfos>"
        '<tripinfo id="a" depart="10" arrival="110" duration="100" waitingTime="20" timeLoss="30" '
        'waitingCount="2" routeLength="1000"><emissions CO2_abs="200000"/></tripinfo>'
        '<tripinfo id="b" depart="400" arrival="500" duration="100" waitingTime="10" timeLoss="20" '
        'waitingCount="1" routeLength="1000"><emissions CO2_abs="100000"/></tripinfo>'
        '<tripinfo id="c" depart="400" arrival="-1" duration="300" waitingTime="40" timeLoss="50" '
        'waitingCount="3" routeLength="1000"><emissions CO2_abs="300000"/></tripinfo>'
        "</tripinfos>"
    )
    m = parse_tripinfo(p, min_depart_s=300)
    assert (m["trips"], m["arrived"], m["unfinished"]) == (2, 1, 1)
    assert m["mean_travel_time_s"] == 200 and m["mean_stops"] == 2
    assert m["co2_g_per_km"] == pytest.approx(400000 / 2000)  # mg/m == g/km
    assert count_tripinfo_arrivals(p) == 2
