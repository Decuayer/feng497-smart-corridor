import os
import random
import traci
from envs.sumo_marl_env import SumoTTLEnv

# Dosya yolları
CONFIG_PATH = "configs/intersections_config.json"
NET_FILE = "networks/osm.net.xml"
ROUTE_FILE = "networks/osm.passenger.trips.xml"  # veya senaryonuzdaki .rou.xml dosyası

def run_test():
    # Ortamı GUI olmadan başlat (GUI görmek için use_gui=True yapabilirsiniz)
    env = SumoTTLEnv(
        config_path=CONFIG_PATH,
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        delta_time=5,
        min_green=10
    )

    # SUMO başlatma komutu
    sumo_binary = "sumo-gui" if env.use_gui else "sumo"
    traci.start([sumo_binary, "-n", env.net_file, "-r", env.route_file, "--no-warnings", "true"])

    print(f"Tanımlı Ajanlar: {env.agents}\n" + "-" * 50)

    try:
        for step_idx in range(1, 21):
            # Her ajan için rastgele aksiyon üret: 0 (Keep) veya 1 (Switch)
            actions = {agent: random.choice([0, 1]) for agent in env.agents}

            obs, rewards, dones, _ = env.step(actions)

            print(f"Adım {step_idx}:")
            for agent in env.agents:
                print(f"  [{agent}] Aksiyon: {actions[agent]} | Ödül: {rewards[agent]:.4f} | Gözlem Boyutu: {len(obs[agent])} | Aktif Faz: {env.current_phase_idx[agent]} (Sarı mı: {env.is_yellow[agent]})")

            if dones["__all__"]:
                print("Simülasyondaki tüm araçlar tamamlandı.")
                break

    finally:
        traci.close()
        print("-" * 50 + "\nTest başarıyla tamamlandı, TraCI kapatıldı.")

if __name__ == "__main__":
    run_test()