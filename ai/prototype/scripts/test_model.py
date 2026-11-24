import sys
from pathlib import Path

# Make ai/prototype importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.mappo_networks import MAPPOModule

# Ortam testinden aldığımız net gözlem boyutları
obs_dims = {
    "lozan": 18,
    "montro": 19,
    "dokuz_eylul": 20
}

# Modeli oluştur
mappo = MAPPOModule(obs_dims=obs_dims)

# Örnek gözlem sözlüğü
dummy_obs = {
    "lozan": torch.zeros(18),
    "montro": torch.zeros(19),
    "dokuz_eylul": torch.zeros(20)
}

actions, log_probs, value = mappo.get_actions_and_value(dummy_obs)

print("Model Başarıyla Derlendi:")
print(f"  Üretilen Aksiyonlar: {actions}")
print(f"  Merkezi Değer Kestirimi V(S): {value.item():.4f}")
print(f"  Global State Boyutu: {mappo.global_state_dim}")