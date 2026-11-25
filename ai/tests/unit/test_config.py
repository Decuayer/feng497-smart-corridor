import pytest

from corridor_ai.common.config import (
    SignalLimits,
    config_to_dict,
    load_experiment_config,
)


def test_spec_limits_are_default():
    lim = SignalLimits()
    assert (lim.min_green_s, lim.max_green_s, lim.yellow_s, lim.all_red_s) == (10, 60, 3, 1)
    assert lim.all_red_mode == "full"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_green_s": 9},
        {"max_green_s": 61},
        {"min_green_s": 30, "max_green_s": 20},
        {"yellow_s": 4},
        {"yellow_s": 2},
        {"all_red_s": 0},
        {"all_red_s": 2},
        {"all_red_mode": "none"},
    ],
)
def test_limits_cannot_be_relaxed(kwargs):
    with pytest.raises(ValueError):
        SignalLimits(**kwargs)


def test_limits_can_be_tightened():
    assert SignalLimits(min_green_s=15, max_green_s=45).max_green_s == 45


def test_base_and_ppo_config_load():
    cfg = load_experiment_config("ai/configs/marl_ppo.yaml")
    assert cfg.name == "ppo_ippo"
    assert cfg.env.reward.stop_definition == "halted"
    assert cfg.limits.all_red_mode == "full"
    assert cfg.env.phase_order == "free"
    assert cfg.ppo.hidden_sizes == (128, 128)


def test_overrides_and_unknown_keys():
    cfg = load_experiment_config(overrides={"env.reward.beta": 0.2, "seed": 7})
    assert cfg.env.reward.beta == 0.2 and cfg.seed == 7
    with pytest.raises(KeyError):
        load_experiment_config(overrides={"env.reward.betta": 0.2})


def test_illegal_limits_in_yaml_are_rejected():
    with pytest.raises(ValueError):
        load_experiment_config(overrides={"limits.yellow_s": 6})


def test_config_round_trip_is_plain_data():
    data = config_to_dict(load_experiment_config())
    assert data["limits"]["min_green_s"] == 10
    assert isinstance(data["ppo"]["hidden_sizes"], list)
