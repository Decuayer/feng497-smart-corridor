import json
import time
import torch
import traci

from envs.sumo_marl_env import SumoTTLEnv
from models.mappo_networks import MAPPOModule

from prototype_paths import CONFIG_PATH, MODEL_PATH, NET_FILE, ROUTE_FILE
SCALE = 2.0

DELTA_TIME = 5
DEVICE = torch.device("cpu")  # Görsel izleme için CPU yeterlidir

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

agents = list(config.keys())
obs_dims = {"lozan": 18, "montro": 19, "dokuz_eylul": 20}

def main():
    env = SumoTTLEnv(CONFIG_PATH, NET_FILE, ROUTE_FILE, use_gui=True, delta_time=DELTA_TIME, min_green=10)
    
    # Modeli yükle
    model = MAPPOModule(obs_dims=obs_dims).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # GUI ve animasyon gecikmesi (100 ms) ile başlat
    traci.start(["sumo-gui", "-n", NET_FILE, "-r", ROUTE_FILE, "--scale", str(SCALE), "--start", "--delay", "100"])

    obs = {agent: env._get_observation(agent) for agent in agents}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            actions = {}
            with torch.no_grad():
                for agent in agents:
                    obs_tensor = torch.as_tensor(obs[agent], dtype=torch.float32, device=DEVICE)
                    logits = model.actors[agent].net(obs_tensor)
                    actions[agent] = torch.argmax(logits).item()

            # Aksiyonları uygula ve ortamı ilerlet
            for agent, act in actions.items():
                env._apply_action(agent, act)

            for _ in range(DELTA_TIME):
                for agent in agents:
                    if env.is_yellow[agent]:
                        env.yellow_timer[agent] -= 1
                        if env.yellow_timer[agent] <= 0:
                            next_g = env.target_green[agent]
                            traci.trafficlight.setPhase(config[agent]["tls_id"], next_g)
                            env.current_phase_idx[agent] = next_g
                            env.is_yellow[agent] = False
                    else:
                        env.phase_timer[agent] += 1

                traci.simulationStep()

            obs = {agent: env._get_observation(agent) for agent in agents}

    finally:
        traci.close()

if __name__ == "__main__":
    main()