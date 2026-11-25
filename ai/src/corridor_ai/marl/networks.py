"""Actor-critic networks shared by all agents (parameter sharing).

The agent one-hot inside the observation lets the shared network specialise
per intersection. Illegal actions are masked by setting their logits to a large
negative number before the softmax, so their probability is effectively 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

MASK_LOGIT = -1e9


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    out = nn.Linear(last, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, tuple(hidden)
        self.actor = _mlp(obs_dim, self.hidden, n_actions, out_gain=0.01)
        self.critic = _mlp(obs_dim, self.hidden, 1, out_gain=1.0)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.actor(obs)
        logits = torch.where(mask, logits, torch.full_like(logits, MASK_LOGIT))
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
