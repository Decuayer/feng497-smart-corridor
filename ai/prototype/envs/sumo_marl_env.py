import json
import os
import sys
import numpy as np
import traci

class SumoTTLEnv:
    def __init__(self, config_path, net_file, route_file, use_gui=False, delta_time=5, min_green=10):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.agents = list(self.config.keys())
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.delta_time = delta_time
        self.min_green = min_green

        # Her kavşağın anlık faz takibi
        self.current_phase_idx = {agent: 0 for agent in self.agents}
        self.phase_timer = {agent: 0 for agent in self.agents}
        self.is_yellow = {agent: False for agent in self.agents}
        self.yellow_timer = {agent: 0 for agent in self.agents}
        self.target_green = {agent: 0 for agent in self.agents}

    def _get_observation(self, agent):
        cfg = self.config[agent]
        obs = []

        # 1 & 2. Şerit Kuyruk ve Bekleme Süreleri
        for lane in cfg["inc_lanes"]:
            queue_len = traci.lane.getLastStepHaltingNumber(lane)
            wait_time = traci.lane.getWaitingTime(lane)
            obs.append(queue_len / 20.0)       # Normalizasyon: maks 20 araç/şerit
            obs.append(min(wait_time / 300.0, 1.0))

        # 3. One-Hot Aktif Yeşil Faz Bilgisi
        num_greens = len(cfg["green_phases"])
        one_hot = [0.0] * num_greens
        curr_green = self.current_phase_idx[agent]
        if curr_green in cfg["green_phases"]:
            idx = cfg["green_phases"].index(curr_green)
            one_hot[idx] = 1.0
        obs.extend(one_hot)

        # 4. Faz Süresi ve Min-Green Emniyet Bayrağı
        obs.append(self.phase_timer[agent] / 50.0)
        obs.append(1.0 if self.phase_timer[agent] >= self.min_green else 0.0)

        return np.array(obs, dtype=np.float32)

    def _apply_action(self, agent, action):
        cfg = self.config[agent]
        tls_id = cfg["tls_id"]

        # Eğer zaten sarı ışıktaysa aksiyon alma, süresini tamamlamasını bekle
        if self.is_yellow[agent]:
            return

        # Action 1 (Switch) seçildiyse ve min_green dolduysa sarı ışığı başlat
        if action == 1 and self.phase_timer[agent] >= self.min_green:
            curr_phase_str = str(self.current_phase_idx[agent])
            yellow_idx = cfg["phase_map"][curr_phase_str]["yellow_phase"]
            self.target_green[agent] = cfg["phase_map"][curr_phase_str]["next_green"]
            
            # Sarı faza geç
            traci.trafficlight.setPhase(tls_id, yellow_idx)
            self.is_yellow[agent] = True
            self.yellow_timer[agent] = cfg["phase_map"][str(yellow_idx)]["duration"]
            self.phase_timer[agent] = 0

    def _compute_reward(self, agent):
        cfg = self.config[agent]
        total_wait = sum(traci.lane.getWaitingTime(lane) for lane in cfg["inc_lanes"])
        total_stops = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in cfg["inc_lanes"])
        
        # Ceza ölçekleme [-1, 0] bandı
        reward = -(0.7 * total_wait + 0.3 * (total_stops * 10.0)) / 1000.0
        return float(reward)

    def step(self, actions):
        # 1. Aksiyonları uygula
        for agent, action in actions.items():
            self._apply_action(agent, action)

        # 2. Simülasyonu delta_time saniye boyunca 1'er saniye ilerlet
        for _ in range(self.delta_time):
            for agent in self.agents:
                if self.is_yellow[agent]:
                    self.yellow_timer[agent] -= 1
                    if self.yellow_timer[agent] <= 0:
                        # Sarı bitti, hedeflenen yeşile geç
                        cfg = self.config[agent]
                        next_g = self.target_green[agent]
                        traci.trafficlight.setPhase(cfg["tls_id"], next_g)
                        self.current_phase_idx[agent] = next_g
                        self.is_yellow[agent] = False
                else:
                    self.phase_timer[agent] += 1

            traci.simulationStep()

        # 3. Yeni gözlem, ödül ve durumları topla
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        dones = {agent: traci.simulation.getMinExpectedNumber() <= 0 for agent in self.agents}
        dones["__all__"] = all(dones.values())

        return observations, rewards, dones, {}