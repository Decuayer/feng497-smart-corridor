import pytest

from corridor_ai.common.contracts import validate, validation_errors
from corridor_ai.common.registry import intersection_ids, load_fixture
from corridor_ai.control.command_sink import build_signal_command


def test_registry_reads_contract_ids():
    assert intersection_ids() == [
        "alsancak_station",
        "lozan",
        "montro",
        "basmane",
        "cankaya",
        "konak_pier",
    ]
    for name in ("toy_corridor", "osm_alsancak_v0"):
        fx = load_fixture(name)
        assert list(fx.agents) == intersection_ids()
        assert "placeholder fixture, not the real digital twin" in fx.disclaimer


def test_built_signal_command_is_schema_valid():
    msg = build_signal_command("lozan", 3, 1, 25.0, "marl_ppo", "adaptive", 1234.0, 5000)
    validate("signal_command", msg)
    assert msg["transition"] == {"yellow_s": 3, "all_red_s": 1}
    assert msg["timestamp"].endswith("Z")


@pytest.mark.parametrize("green", [9.0, 61.0])
def test_out_of_bounds_green_is_rejected(green):
    msg = build_signal_command("lozan", 1, 0, green, "marl_ppo", "adaptive", 0.0)
    assert validation_errors("signal_command", msg)


def test_unknown_intersection_rejected():
    msg = build_signal_command("dokuz_eylul", 1, 0, 20.0, "fixed_time", "fixed_time", 0.0)
    with pytest.raises(ValueError):
        validate("signal_command", msg)
