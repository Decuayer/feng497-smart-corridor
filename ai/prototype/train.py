import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

from envs.sumo_marl_env import SumoTTLEnv
from models.mappo_networks import MAPPOModule

# --- Hiperparametreler ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 4
BATCH_SIZE = 64
ROLLOUT_STEPS = 120       # Her güncelleme öncesi toplanacak adım sayısı
TOTAL_EPISODES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

from prototype_paths import CONFIG_PATH, MODEL_PATH, NET_FILE, ROUTE_FILE

obs_dims = {"lozan": 18, "montro": 19, "dokuz_eylul": 20}
agents = list(obs_dims.keys())


class RolloutBuffer:
    def __init__(self, steps):
        self.steps = steps
        self.reset()

    def reset(self):
        self.obs = {agent: [] for agent in agents}
        self.actions = {agent: [] for agent in agents}
        self.log_probs = {agent: [] for agent in agents}
        self.rewards = []
        self.global_states = []
        self.values = []
        self.dones = []

    def store(self, obs, acts, log_p, rew, g_state, val, done):
        for agent in agents:
            self.obs[agent].append(torch.as_tensor(obs[agent], dtype=torch.float32))
            self.actions[agent].append(torch.as_tensor(acts[agent], dtype=torch.long))
            self.log_probs[agent].append(log_p[agent])
        self.rewards.append(rew)
        self.global_states.append(g_state)
        self.values.append(val)
        self.dones.append(done)


def train():
    env = SumoTTLEnv(CONFIG_PATH, NET_FILE, ROUTE_FILE, use_gui=False, delta_time=5, min_green=10)
    model = MAPPOModule(obs_dims=obs_dims).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = RolloutBuffer(ROLLOUT_STEPS)

    for episode in range(1, TOTAL_EPISODES + 1):
        # SUMO başlat
        traci.start(["sumo", "-n", env.net_file, "-r", env.route_file, "--no-warnings", "true"])
        obs = {agent: env._get_observation(agent) for agent in agents}
        episode_reward = 0.0

        for step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                actions, log_probs, value = model.get_actions_and_value(obs, device=DEVICE)
                g_state = model.get_global_state(obs, device=DEVICE)

            next_obs, rewards, dones, _ = env.step(actions)
            step_reward = sum(rewards.values())
            episode_reward += step_reward

            buffer.store(obs, actions, log_probs, step_reward, g_state, value, dones["__all__"])
            obs = next_obs

            if dones["__all__"]:
                break

        traci.close()

        # --- GAE ve Hedef Değer Hesaplama ---
        with torch.no_grad():
            last_g_state = model.get_global_state(obs, device=DEVICE)
            next_val = model.critic(last_g_state).item()

        values = [v.item() for v in buffer.values] + [next_val]
        advantages = np.zeros(len(buffer.rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(buffer.rewards))):
            non_terminal = 1.0 - float(buffer.dones[t])
            delta = buffer.rewards[t] + GAMMA * values[t + 1] * non_terminal - values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae

        returns = torch.tensor(advantages + np.array(values[:-1]), dtype=torch.float32, device=DEVICE)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # --- PPO Güncellemesi ---
        g_states_b = torch.stack(buffer.global_states).squeeze(1).to(DEVICE)
        for _ in range(EPOCHS):
            # Kritik kaybı
            v_pred = model.critic(g_states_b).squeeze(-1)
            critic_loss = nn.MSELoss()(v_pred, returns)

            # Aktör kayıpları
            actor_loss = 0.0
            entropy_loss = 0.0
            for agent in agents:
                obs_b = torch.stack(buffer.obs[agent]).to(DEVICE)
                acts_b = torch.stack(buffer.actions[agent]).to(DEVICE)
                old_log_p = torch.stack(buffer.log_probs[agent]).to(DEVICE)

                dist = model.actors[agent](obs_b)
                new_log_p = dist.log_prob(acts_b)
                ratio = torch.exp(new_log_p - old_log_p)

                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_tensor
                actor_loss += -torch.min(surr1, surr2).mean()
                entropy_loss += dist.entropy().mean()

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        buffer.reset()
        print(f"Epizot [{episode}/{TOTAL_EPISODES}] | Toplam Ödül: {episode_reward:.4f} | Kayıp: {total_loss.item():.4f}")

    # Modeli kaydet
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Eğitim tamamlandı. Model '{MODEL_PATH}' dosyasına kaydedildi.")


if __name__ == "__main__":
    train()