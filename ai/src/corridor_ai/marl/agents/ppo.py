"""Independent PPO with parameter sharing (IPPO-PS) and action masking.

* One ActorCritic is shared by all agents. Each agent's experience is a
  separate trajectory with its own reward and its own GAE.
* Episodes are truncated, not terminated, by the time limit, so the last
  step bootstraps from V(s_T).
* Standard clipped-surrogate PPO with an entropy bonus, MSE value loss,
  advantage normalisation and gradient-norm clipping.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from corridor_ai.common.config import PPOConfig
from corridor_ai.marl.networks import ActorCritic


@dataclass
class Trajectory:
    obs: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    last_value: float = 0.0  # V(s_T) for bootstrapping at truncation


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ) for a single truncated trajectory (no terminal states inside)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    next_value, gae = last_value, 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values.astype(np.float32)


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = ActorCritic(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ------------------------------------------------------------- acting
    @torch.no_grad()
    def act(
        self, obs: np.ndarray, masks: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        dist = self.net.distribution(o, m)
        a = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return (
            a.cpu().numpy(),
            dist.log_prob(a).cpu().numpy(),
            self.net.value(o).cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.net.value(o).cpu().numpy()

    # ------------------------------------------------------------- update
    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        cfg = self.cfg
        obs, masks, acts, logps, advs, rets, vals = [], [], [], [], [], [], []
        for tr in trajectories:
            if not tr.rewards:
                continue
            values = np.asarray(tr.values, dtype=np.float32)
            adv, ret = compute_gae(
                np.asarray(tr.rewards, dtype=np.float32), values, tr.last_value,
                cfg.gamma, cfg.gae_lambda,
            )  # fmt: skip
            obs.append(np.asarray(tr.obs, dtype=np.float32))
            masks.append(np.asarray(tr.masks, dtype=bool))
            acts.append(np.asarray(tr.actions, dtype=np.int64))
            logps.append(np.asarray(tr.logps, dtype=np.float32))
            advs.append(adv)
            rets.append(ret)
            vals.append(values)
        dev = self.device
        b_obs = torch.as_tensor(np.concatenate(obs), device=dev)
        b_mask = torch.as_tensor(np.concatenate(masks), device=dev)
        b_act = torch.as_tensor(np.concatenate(acts), device=dev)
        b_logp = torch.as_tensor(np.concatenate(logps), device=dev)
        b_adv = torch.as_tensor(np.concatenate(advs), device=dev)
        b_ret = torch.as_tensor(np.concatenate(rets), device=dev)
        b_val = np.concatenate(vals)
        n = len(b_obs)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0}  # fmt: skip
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                dist = self.net.distribution(b_obs[idx], b_mask[idx])
                new_logp = dist.log_prob(b_act[idx])
                log_ratio = new_logp - b_logp[idx]
                ratio = log_ratio.exp()
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(idx) > 1 else adv
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
                policy_loss = pg.mean()
                value_loss = 0.5 * (self.net.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                n_updates += 1
        out = {k: v / max(n_updates, 1) for k, v in stats.items()}
        var_y = float(np.var(b_ret.cpu().numpy()))
        out["explained_variance"] = (
            float(1 - np.var(b_ret.cpu().numpy() - b_val) / var_y) if var_y > 0 else 0.0
        )
        out["samples"] = float(n)
        return out

    # ---------------------------------------------------------- persistence
    def save(self, path, extra: dict | None = None) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "obs_dim": self.net.obs_dim,
                "n_actions": self.net.n_actions,
                "hidden": list(self.net.hidden),
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path, cfg: PPOConfig) -> tuple[PPOAgent, dict]:
        ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
        cfg = dataclasses.replace(cfg, hidden_sizes=tuple(ckpt["hidden"]))
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], cfg)
        agent.net.load_state_dict(ckpt["state_dict"])
        return agent, ckpt
