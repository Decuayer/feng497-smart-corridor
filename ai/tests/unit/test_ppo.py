import numpy as np
import torch

from corridor_ai.common.config import PPOConfig
from corridor_ai.marl.agents.ppo import PPOAgent, Trajectory, compute_gae


def test_gae_matches_manual_computation():
    r = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    v = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    g, lam, last = 0.9, 0.8, 0.3
    adv, ret = compute_gae(r, v, last, g, lam)
    d2 = r[2] + g * last - v[2]
    d1 = r[1] + g * v[2] - v[1]
    d0 = r[0] + g * v[1] - v[0]
    expected = [d0 + g * lam * (d1 + g * lam * d2), d1 + g * lam * d2, d2]
    np.testing.assert_allclose(adv, expected, rtol=1e-6)
    np.testing.assert_allclose(ret, adv + v, rtol=1e-6)


def test_masked_actions_are_never_sampled():
    torch.manual_seed(0)
    agent = PPOAgent(6, 5, PPOConfig())
    obs = np.random.rand(500, 6).astype(np.float32)
    masks = np.zeros((500, 5), dtype=bool)
    masks[:, [0, 3]] = True
    acts, logps, _ = agent.act(obs, masks)
    assert set(acts.tolist()) <= {0, 3}
    assert np.all(np.isfinite(logps))
    det, _, _ = agent.act(obs, masks, deterministic=True)
    assert set(det.tolist()) <= {0, 3}


def test_update_runs_and_save_load_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = PPOConfig(minibatch_size=16, epochs=2)
    agent = PPOAgent(4, 3, cfg)
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(2):
        tr = Trajectory()
        for _ in range(20):
            o = rng.random(4).astype(np.float32)
            m = np.array([True, True, False])
            a, lp, v = agent.act(o[None], m[None])
            tr.obs.append(o), tr.masks.append(m), tr.actions.append(int(a[0]))
            tr.logps.append(float(lp[0])), tr.values.append(float(v[0])), tr.rewards.append(-1.0)
        trajs.append(tr)
    stats = agent.update(trajs)
    assert all(np.isfinite(v) for v in stats.values()) and stats["samples"] == 40
    agent.save(tmp_path / "m.pt", {"k_max": 2})
    loaded, ckpt = PPOAgent.load(tmp_path / "m.pt", PPOConfig(hidden_sizes=(8,)))
    assert ckpt["k_max"] == 2 and loaded.net.hidden == (128, 128)
    o = rng.random((3, 4)).astype(np.float32)
    np.testing.assert_allclose(loaded.value(o), agent.value(o), rtol=1e-6)
