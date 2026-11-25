"""DQN agent, PPO exclusion of preempted steps, evaluation aggregation (no SUMO)."""

import numpy as np
import pytest
import torch

from corridor_ai.common.config import DQNConfig, PPOConfig
from corridor_ai.marl.agents.dqn import DQNAgent, ReplayBuffer, masked
from corridor_ai.marl.agents.ppo import PPOAgent, Trajectory
from corridor_ai.marl.evaluate import add_group_rows, checkpoint_algo, significance_lines


def test_dqn_acts_only_legal_actions_greedy_and_exploring():
    torch.manual_seed(0)
    agent = DQNAgent(5, 4, DQNConfig(), seed=0)
    obs = np.random.rand(400, 5).astype(np.float32)
    masks = np.zeros((400, 4), bool)
    masks[:, [1, 3]] = True
    for eps in (0.0, 0.5, 1.0):
        assert set(agent.act(obs, masks, eps).tolist()) <= {1, 3}


def test_epsilon_schedule():
    agent = DQNAgent(3, 2, DQNConfig(eps_start=1.0, eps_end=0.1, eps_decay_fraction=0.5))
    assert agent.epsilon(0, 100) == 1.0
    assert agent.epsilon(25, 100) == pytest.approx(0.55)
    assert agent.epsilon(50, 100) == pytest.approx(0.1) == agent.epsilon(99, 100)


def test_double_dqn_target_ignores_illegal_next_actions():
    torch.manual_seed(0)
    cfg = DQNConfig(gamma=0.9, target_tau=0.0, learning_rate=0.0)
    agent = DQNAgent(2, 3, cfg)
    with torch.no_grad():  # online prefers action 2, which will be illegal in s'
        for net in (agent.q, agent.target):
            last = net.net[-1]
            last.weight.zero_()
            last.bias.copy_(torch.tensor([1.0, 5.0, 9.0]))
    nmask = torch.tensor([[True, True, False]])
    best = masked(agent.q(torch.zeros(1, 2)), nmask).argmax(dim=-1)
    assert best.item() == 1  # legal argmax, not the illegal action 2


def test_dqn_update_save_load(tmp_path):
    torch.manual_seed(0)
    agent = DQNAgent(4, 3, DQNConfig(batch_size=8), seed=0)
    buf = ReplayBuffer(50, 4, 3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        m = np.array([True, rng.random() < 0.5, True])
        buf.add(rng.random(4), m, 0, -1.0, rng.random(4), m)
    assert buf.size == 30
    stats = agent.update(buf.sample(8))
    assert np.isfinite(stats["td_loss"])
    agent.save(tmp_path / "d.pt", {"k_max": 2})
    assert checkpoint_algo(tmp_path / "d.pt") == "dqn"
    loaded, ckpt = DQNAgent.load(tmp_path / "d.pt", DQNConfig(hidden_sizes=(8,)))
    o = rng.random((5, 4)).astype(np.float32)
    np.testing.assert_array_equal(
        loaded.act(o, np.ones((5, 3), bool)), agent.act(o, np.ones((5, 3), bool))
    )


def test_replay_buffer_wraps():
    buf = ReplayBuffer(3, 1, 2)
    for i in range(5):
        buf.add([i], [True, True], 0, float(i), [i], [True, True])
    assert buf.size == 3 and sorted(buf.reward.tolist()) == [2.0, 3.0, 4.0]


def test_ppo_excludes_preempted_steps():
    torch.manual_seed(0)
    agent = PPOAgent(3, 2, PPOConfig(minibatch_size=4, epochs=1))
    tr = Trajectory()
    for k in range(6):
        tr.obs.append(np.full(3, k / 6, np.float32)), tr.masks.append(np.array([True, True]))
        tr.actions.append(0), tr.logps.append(-0.69), tr.values.append(0.0), tr.rewards.append(-1.0)
        tr.include.append(k % 2 == 0)
    assert agent.update([tr])["samples"] == 3
    tr.include = [False] * 6
    with pytest.raises(ValueError):  # nothing left to train on
        agent.update([tr])


def _rows():
    rows = []
    for seed in (1, 2, 3):
        for c, w in (("fixed_time", 100), ("webster", 20), ("a", 30 + seed), ("b", 34 - seed)):
            rows.append(
                {
                    "fixture": "toy",
                    "controller": c,
                    "scale": 1.0,
                    "seed": seed,
                    "mean_waiting_time_s": float(w),
                    "mean_travel_time_s": float(w + 100),
                }
            )
    return rows


def test_group_rows_average_members_per_seed(monkeypatch):
    import corridor_ai.marl.evaluate as ev

    monkeypatch.setattr(ev, "METRICS", ("mean_waiting_time_s", "mean_travel_time_s"))
    out = add_group_rows(_rows(), {"ab": ["a", "b"]})
    grp = [r for r in out if r["controller"] == "ab"]
    assert len(grp) == 3 and all(r["mean_waiting_time_s"] == 32.0 for r in grp)


def test_significance_lines_compare_against_both_baselines():
    text = "\n".join(significance_lines(_rows()))
    assert "| 1 | a | fixed_time |" in text and "| 1 | a | webster |" in text
    assert "| 1 | webster | fixed_time |" not in text
