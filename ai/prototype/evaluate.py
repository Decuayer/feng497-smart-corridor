# evaluate.py
import json
import numpy as np
import torch
import traci

from envs.sumo_marl_env import SumoTTLEnv
from models.mappo_networks import MAPPOModule

from prototype_paths import CONFIG_PATH, MODEL_PATH, NET_FILE, ROUTE_FILE
SCALE = 2.0

SIM_STEPS = 120  # Her adım 5 saniye -> toplam 600 saniye (10 dk simülasyon)
DELTA_TIME = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

agents = list(config.keys())
all_monitored_lanes = [lane for agent in agents for lane in config[agent]["inc_lanes"]]
obs_dims = {"lozan": 18, "montro": 19, "dokuz_eylul": 20}


def run_baseline():
    """SUMO'nun varsayılan sabit zamanlı (Fixed-Time) sinyal planını test eder."""
    traci.start([
        "sumo", "-n", NET_FILE, "-r", ROUTE_FILE, 
        "--scale", str(SCALE), 
        "--no-warnings", "true"
    ])
    
    total_waiting_time = 0.0
    total_halting_vehicles = 0
    step_count = 0

    for _ in range(SIM_STEPS):
        for _ in range(DELTA_TIME):
            traci.simulationStep()
            step_count += 1
            total_waiting_time += sum(traci.lane.getWaitingTime(lane) for lane in all_monitored_lanes)
            total_halting_vehicles += sum(traci.lane.getLastStepHaltingNumber(lane) for lane in all_monitored_lanes)

    completed_vehicles = traci.simulation.getArrivedNumber()
    traci.close()

    return {
        "avg_waiting_time": total_waiting_time / step_count,
        "avg_queue": total_halting_vehicles / step_count,
        "completed_trips": completed_vehicles
    }


def run_mappo():
    """Eğitilen MAPPO modelini çıkarım (inference) modunda test eder."""
    env = SumoTTLEnv(CONFIG_PATH, NET_FILE, ROUTE_FILE, use_gui=False, delta_time=DELTA_TIME, min_green=10)
    model = MAPPOModule(obs_dims=obs_dims).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    traci.start([
        "sumo", "-n", NET_FILE, "-r", ROUTE_FILE, 
        "--scale", str(SCALE), 
        "--no-warnings", "true"
    ])
    
    obs = {agent: env._get_observation(agent) for agent in agents}
    total_waiting_time = 0.0
    total_halting_vehicles = 0
    step_count = 0

    for _ in range(SIM_STEPS):
        actions = {}
        with torch.no_grad():
            for agent in agents:
                obs_tensor = torch.as_tensor(obs[agent], dtype=torch.float32, device=DEVICE)
                logits = model.actors[agent].net(obs_tensor)
                actions[agent] = torch.argmax(logits).item()  # Deterministik / En iyi aksiyon

        # Aksiyonları uygula ve delta_time kadar işlet
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
            step_count += 1
            total_waiting_time += sum(traci.lane.getWaitingTime(lane) for lane in all_monitored_lanes)
            total_halting_vehicles += sum(traci.lane.getLastStepHaltingNumber(lane) for lane in all_monitored_lanes)

        obs = {agent: env._get_observation(agent) for agent in agents}

    completed_vehicles = traci.simulation.getArrivedNumber()
    traci.close()

    return {
        "avg_waiting_time": total_waiting_time / step_count,
        "avg_queue": total_halting_vehicles / step_count,
        "completed_trips": completed_vehicles
    }


def main():
    print("1/2: Sabit Zamanlı (Baseline) Simülasyon Koşturuluyor...")
    baseline_results = run_baseline()

    print("2/2: MAPPO Modeli Simülasyonu Koşturuluyor...")
    mappo_results = run_mappo()

    # İyileşme Oranları Hesaplama
    wait_imp = ((baseline_results["avg_waiting_time"] - mappo_results["avg_waiting_time"]) / baseline_results["avg_waiting_time"]) * 100
    queue_imp = ((baseline_results["avg_queue"] - mappo_results["avg_queue"]) / baseline_results["avg_queue"]) * 100

    print("\n" + "=" * 55)
    print("         PERFORMANS KARŞILAŞTIRMA RAPORU")
    print("=" * 55)
    print(f"{'Metrik':<25} | {'Baseline':<12} | {'MAPPO':<12} | {'İyileşme':<8}")
    print("-" * 55)
    print(f"{'Ort. Bekleme Süresi (sn)':<25} | {baseline_results['avg_waiting_time']:<12.2f} | {mappo_results['avg_waiting_time']:<12.2f} | %{wait_imp:+.2f}")
    print(f"{'Ortalama Kuyruk (Araç)':<25} | {baseline_results['avg_queue']:<12.2f} | {mappo_results['avg_queue']:<12.2f} | %{queue_imp:+.2f}")
    print(f"{'Tamamlanan Rota (Adet)':<25} | {baseline_results['completed_trips']:<12} | {mappo_results['completed_trips']:<12} | -")
    print("=" * 55)


if __name__ == "__main__":
    main()