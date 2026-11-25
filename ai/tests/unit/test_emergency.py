"""Emergency corridor: intake, ETA/time-to-green, recovery maths (no SUMO process)."""

import numpy as np
import pytest

from corridor_ai.common.config import EmergencyConfig, SignalLimits, load_experiment_config
from corridor_ai.common.contracts import validate
from corridor_ai.common.log_events import build_log_event
from corridor_ai.common.registry import load_fixture
from corridor_ai.emergency.intake import build_request_message, new_request_id, parse_request
from corridor_ai.emergency.network import load_road_network
from corridor_ai.emergency.preemption import time_to_green
from corridor_ai.emergency.recovery import recovery_time, smooth
from corridor_ai.emergency.scenario import ev_departures
from corridor_ai.sim.phase_machine import PhaseMachine


def _msg(**over):
    msg = build_request_message(
        new_request_id(),
        "amb-1",
        "ambulance",
        "request",
        10.0,
        ["lozan", "montro"],
        speed_mps=12.0,
        sumo_vehicle_id="ev0",
    )
    msg.update(over)
    return msg


def test_simulated_request_is_schema_valid_and_accepted():
    msg = _msg()
    validate("emergency_request", msg)
    req, reason = parse_request(msg)
    assert (
        reason is None
        and req.sumo_vehicle_id == "ev0"
        and req.route_intersections == ("lozan", "montro")
    )


def test_unverified_external_request_is_rejected():
    msg = _msg(source="v2i_lora", auth={"verified": False, "method": "hmac_sha256"})
    req, reason = parse_request(msg)
    assert req is None and "unverified" in reason


def test_verified_external_request_is_accepted():
    msg = _msg(source="v2i_lora", auth={"verified": True, "method": "hmac_sha256"})
    assert parse_request(msg)[0] is not None


def test_invalid_message_is_rejected():
    msg = _msg(vehicle_type="police_car")
    req, reason = parse_request(msg)
    assert req is None and reason.startswith("schema")


def test_log_event_is_schema_valid():
    ev = build_log_event("ai", "info", "preemption_started", "x", 12.0, "lozan", {"eta_s": 5})
    assert ev["synthetic"] is True and ev["timestamp"].endswith("Z")
    with pytest.raises(ValueError):
        build_log_event("ai", "info", "preemption_started", "x", 12.0, "bornova")


def test_time_to_green_respects_limits():
    lim = SignalLimits()
    m = PhaseMachine("T", ["GGrr", "rrGG"], lim)
    for _ in range(4):
        m.next_state()
    assert time_to_green(m, 0, 10, 3, 1) == 0
    assert time_to_green(m, 1, 10, 3, 1) == (10 - 4) + 3 + 1
    for _ in range(10):
        m.next_state()
    m.request(1)
    m.next_state()  # first second of yellow
    assert m.mode == "yellow"
    assert time_to_green(m, 1, 10, 3, 1) == (3 - 1) + 1
    assert time_to_green(m, 0, 10, 3, 1) == (3 - 1) + 1 + 10 + 4


def test_movement_phases_on_toy_corridor():
    net = load_road_network(load_fixture("toy_corridor").net)
    mv = net.movements[("J1_J2", "J2_J3")]  # eastbound through at J2
    assert mv.tls_id == "J2" and mv.phases == (0,)  # phase 0 = arterial, priority green
    assert set(mv.in_lanes) == {"J1_J2_0", "J1_J2_1"}


def test_ev_departures_paired_and_in_window():
    cfg = load_experiment_config("ai/configs/emergency.yaml")
    a = ev_departures(cfg, 1000, 3900)
    assert a == ev_departures(cfg, 1000, 3900) and a != ev_departures(cfg, 1001, 3900)
    assert len(a) == 10 and all(420 <= t <= 3000 for t, _ in a)
    assert {v for _, v in a} <= {"emergency", "firebrigade"}


def test_recovery_time_one_sided():
    cfg = EmergencyConfig(recovery_smooth_s=1, recovery_sustain_s=5, recovery_tolerance_abs=2)
    t = np.arange(100, dtype=float)
    base = np.full(100, 10.0)
    pre = base.copy()
    pre[20:40] = 25.0  # worse than baseline for 20 s after release at t=20
    assert recovery_time(t, pre, base, 20.0, 100.0, cfg) == 20.0
    better = np.full(100, 3.0)  # better than baseline -> recovered immediately
    assert recovery_time(t, better, base, 20.0, 100.0, cfg) == 0.0
    assert recovery_time(t, pre, base, 20.0, 30.0, cfg) is None  # censored


def test_smooth_preserves_length():
    x = np.arange(50.0)
    assert len(smooth(x, 30)) == 50
