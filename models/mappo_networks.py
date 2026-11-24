import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    """Her kavşağın kendi yerel gözlemiyle aksiyon (Keep=0, Switch=1) seçtiği ağ."""
    def __init__(self, obs_dim, hidden_dim=128, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs, action_mask=None):
        logits = self.net(obs)
        if action_mask is not None:
            # Maskelenen aksiyonun logit değerini -inf yaparak seçilmesini engelle
            logits = logits + (action_mask * -1e9)
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    """Eğitim esnasında tüm kavşakların birleşik durumundan V(S) kestiren merkezi ağ."""
    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)


class MAPPOModule(nn.Module):
    """3 kavşak için aktörleri ve merkezi eleştirmeni bir arada tutan üst modül."""
    def __init__(self, obs_dims: dict, act_dim=2):
        super().__init__()
        self.agents = list(obs_dims.keys())
        
        # Her kavşağın gözlem boyutu farklı olduğu için özel aktörler
        self.actors = nn.ModuleDict({
            agent: Actor(obs_dim=obs_dims[agent], act_dim=act_dim)
            for agent in self.agents
        })
        
        # Global State: Lozan (18) + Montrö (19) + 9 Eylül (20) = 57 boyut
        self.global_state_dim = sum(obs_dims.values())
        self.critic = CentralizedCritic(global_state_dim=self.global_state_dim)

    def get_global_state(self, obs_dict, device="cpu"):
        # Gözlemleri tek bir tensörde birleştir: [1, 57]
        tensors = [torch.as_tensor(obs_dict[agent], dtype=torch.float32, device=device) for agent in self.agents]
        return torch.cat(tensors, dim=-1)

    def get_actions_and_value(self, obs_dict, device="cpu"):
        actions = {}
        log_probs = {}
        
        # 1. Aktörler bağımsız aksiyon seçer
        for agent in self.agents:
            obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32, device=device)
            dist = self.actors[agent](obs_tensor)
            action = dist.sample()
            actions[agent] = action.item()
            log_probs[agent] = dist.log_prob(action)

        # 2. Merkezi eleştirmen global durumu değerlendirir
        global_state = self.get_global_state(obs_dict, device=device)
        value = self.critic(global_state)

        return actions, log_probs, value