# Architecture overview

This page describes the system in two views:

1. the **functional view**: five layers and the data flowing between them;
2. the **deployment view**: the team's diagrams in [`diagrams/`](diagrams/).

## 1. Functional view (five layers)

```mermaid
flowchart TB
  subgraph L1["Layer 1 · Edge sensing (edge/)"]
    CAM["Camera 1920×1080"] --> PRE["Downscale 640×640"] --> DET["YOLOv8-nano<br/>TensorRT FP16/INT8"] --> ROI["ROI lane queue estimation<br/>(vehicles, metres)"]
  end
  subgraph L2["Layer 2 · Communication & V2I (hardware/)"]
    TX["Emergency vehicle transponder<br/>RF/LoRa 433/868 MHz, BLE 2.4 GHz<br/>HMAC/AES, ≥300 m LOS, under 50 ms"]
    ESP["ESP32 relay driver board"]
    IL["Hardware green-conflict interlock<br/>(under 35 ms)"]
    WD["1000 ms watchdog<br/>→ flashing yellow"]
    LAMP["220 V AC lamps (12 V DC logic)"]
    ESP --> IL --> LAMP
    WD --> LAMP
  end
  subgraph L3["Layer 3 · Digital twin (simulation/)"]
    OSM["OpenStreetMap"] -->|netconvert| NET["SUMO network"]
    IZUM["IZUM hourly counts"] --> OD["O/D matrices"] --> SC["Scenarios: rush hour,<br/>accident, emergency"]
    NET --> SC
  end
  subgraph L4["Layer 4 · AI & optimisation (ai/, analysis/)"]
    MARL["MARL PPO / DQN<br/>one agent per intersection"]
    EMG["Proactive emergency green wave<br/>(2 junctions ahead)"]
    FC["15-min density forecasting"]
    LLM["LLM decision support"]
    QT["Queueing M/M/1, M/M/c, M/G/1<br/>Webster baseline, BPR/DTA, CO₂"]
  end
  subgraph L5["Layer 5 · Backend & dashboard (backend/, dashboard/)"]
    BRK[("Mosquitto MQTT")]
    API["FastAPI REST + WebSocket<br/>BPR dynamic routing"]
    LOG[("System log store")]
    UI["Operator dashboard"]
  end

  ROI -->|queue_telemetry| BRK
  TX -->|emergency_request| BRK
  BRK <--> API
  API --> MARL
  API --> EMG
  SC <-->|TraCI| MARL
  SC <-->|TraCI| EMG
  MARL -->|signal_command| API
  EMG -->|signal_command| API
  API -->|signal_command| ESP
  API --> LOG -->|system_log_event| LLM
  FC --> LLM
  API <--> UI
  QT -.->|parameters, baselines| MARL
```

### Layer summaries

| Layer | Owner(s) | Inputs | Outputs |
|---|---|---|---|
| 1. Edge sensing | Nihat | Camera frames | `queue_telemetry` over MQTT (target latency under 85 ms) |
| 2. Communication / V2I | Tolga | Transponder frames, `signal_command` | Verified emergency requests (through the backend), lamp states. Hardware interlock under 35 ms, watchdog 1000 ms |
| 3. Digital twin | Öykü | OSM, IZUM counts | SUMO network, routes, scenarios, TraCI access |
| 4. AI & optimisation | Demir (`ai/`), Ayça (`analysis/`) | Telemetry, emergency requests, SUMO state, logs | `signal_command`, forecasts, operator briefings; analytical baselines |
| 5. Backend & dashboard | Ario | All MQTT traffic | REST/WebSocket APIs, routing, log store, dashboard |

### Safety and fallback behaviour

- Signal limits for every controller: **min green 10 s, max green 60 s, yellow 3 s, all-red 1 s**.
- The **hardware interlock** rejects conflicting greens regardless of the command it receives.
- The **watchdog** switches to flashing yellow if the controller stops responding for 1000 ms.
- If the **network or central link fails**, intersections fall back to local adaptive Webster mode.
- If the **MARL policy diverges**, actions are clamped to the agreed green-time bounds.

## 2. Deployment view (team diagrams)

| Diagram | Source |
|---|---|
| ![System architecture](diagrams/System-Arch.png) | [`System-Arch.drawio`](diagrams/System-Arch.drawio) |
| ![Web and mobile architecture](diagrams/Web-Mobile-Arch.png) | [`Web-Mobile-Arch.drawio`](diagrams/Web-Mobile-Arch.drawio) |

The diagrams (labels in Turkish) show four deployment tiers:

1. **Field and edge hardware**: ESP32 modules and the signal heads at each intersection.
2. **Distributed corridor servers**: regional servers (Alsancak, Bornova, Karşıyaka, Balçova, Buca) that send signal-phase commands to the ESP32 modules.
3. **Central server and analytics**: camera streams → YOLO/vision module → central optimisation and analytics server → corridor coordination. This tier issues macro distribution policies.
4. **User and management tier**: the web management panel (analytics, charts, model configuration) and the **mobile app** (map and dynamic routing, crowd-sourced GPS telemetry; owner TBD).

### Open questions on the architecture

- **Where vision inference runs.** The diagrams place the YOLO module on the central server; the project specification places YOLOv8-nano on edge devices (Jetson / Raspberry Pi). Which is the target deployment, or is the central one only for the IZUM stream prototype? (Nihat, Ario)
- **Regional servers vs. a single backend.** How do the regional corridor servers relate to the FastAPI backend? (Ario)
- **Mobile app** has no owner yet.
