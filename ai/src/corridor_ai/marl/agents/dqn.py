"""Parameter-shared independent Double DQN with action masking.

* One Q-network is shared by all agents (the agent one-hot in the observation
  lets it specialise). Each agent's transitions go into one replay buffer.
* Illegal actions are excluded everywhere:
  - acting: ε-greedy over *legal* actions only;
  - target: Double DQN, i.e. argmax over legal next actions with the online
    network, evaluated with the target network.
* Episodes end by time-limit truncation only (no terminal states), so every
  transition bootstraps from its next state.
* Huber loss, Adam, gradient clipping, soft target updates (Polyak τ).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn

from corridor_ai.common.config import DQNConfig
from corridor_ai.marl.networks import MASK_LOGIT, _mlp


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, tuple(hidden)
        self.net = _mlp(obs_dim, self.hidden, n_actions, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def masked(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, q, torch.full_like(q, MASK_LOGIT))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.mask = np.zeros((capacity, n_actions), bool)
        self.next_mask = np.zeros((capacity, n_actions), bool)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, mask, action, reward, next_obs, next_mask) -> None:
        i = self.pos
        self.obs[i], self.mask[i], self.action[i] = obs, mask, action
        self.reward[i], self.next_obs[i], self.next_mask[i] = reward, next_obs, next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=n)
        return {
            "obs": self.obs[idx], "mask": self.mask[idx], "action": self.action[idx],
            "reward": self.reward[idx], "next_obs": self.next_obs[idx],
            "next_mask": self.next_mask[idx],
        }  # fmt: skip


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        greedy = masked(self.q(o), m).argmax(dim=-1).cpu().numpy()
        if epsilon <= 0:
            return greedy
        out = greedy.copy()
        for i in range(len(out)):
            if self.rng.random() < epsilon:
                out[i] = int(self.rng.choice(np.flatnonzero(masks[i])))
        return out

    def epsilon(self, step: int, total_steps: int) -> float:
        cfg = self.cfg
        horizon = max(1, int(cfg.eps_decay_fraction * total_steps))
        frac = min(step / horizon, 1.0)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    # ------------------------------------------------------------- update
    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        dev, cfg = self.device, self.cfg
        obs = torch.as_tensor(batch["obs"], device=dev)
        act = torch.as_tensor(batch["action"], device=dev)
        rew = torch.as_tensor(batch["reward"], device=dev)
        nxt = torch.as_tensor(batch["next_obs"], device=dev)
        nmask = torch.as_tensor(batch["next_mask"], device=dev)
        q = self.q(obs).gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():
            best = masked(self.q(nxt), nmask).argmax(dim=-1, keepdim=True)  # Double DQN
            target = rew + cfg.gamma * self.target(nxt).gather(1, best).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters(), strict=True):
                tp.mul_(1 - cfg.target_tau).add_(cfg.target_tau * p)
        return {"td_loss": float(loss.item()), "q_mean": float(q.mean().item())}

    # ---------------------------------------------------------- persistence
    def save(self, path, extra: dict | None = None) -> None:
        torch.save(
            {
                "algo": "dqn",
                "state_dict": self.q.state_dict(),
                "obs_dim": self.q.obs_dim,
                "n_actions": self.q.n_actions,
                "hidden": list(self.q.hidden),
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path, cfg: DQNConfig) -> tuple[DQNAgent, dict]:
        ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
        cfg = dataclasses.replace(cfg, hidden_sizes=tuple(ckpt["hidden"]))
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], cfg)
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        return agent, ckpt
