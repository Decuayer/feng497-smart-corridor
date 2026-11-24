# edge/: Edge vision and queue estimation

- **Owner:** Nihat Efe Kargı (EEE)
- **Status:** in progress (one prototype)

## Purpose

On-device vehicle detection and lane queue estimation. Input: 1920×1080 camera frames, downscaled to 640×640. Model: YOLOv8-nano with TensorRT FP16/INT8 on NVIDIA Jetson / Raspberry Pi. Classes: car, light commercial, bus, ambulance, fire truck. ROI-based lane queues in vehicles and metres. Target latency under 85 ms. Output: JSON telemetry over MQTT.

## Scope

**Belongs here:** the IZUM camera data pipeline, synthetic image generation, dataset tooling, YOLOv8-nano training/export, TensorRT deployment, ROI calibration and queue estimation, the edge MQTT publisher.

**Does not belong here:** backend services (`backend/`), signal control (`ai/`), V2I radio (`hardware/`).

## Current and planned contents

```
edge/
├── prototypes/
│   ├── izum_live_detection.py     # live IZUM stream + YOLOv8n vehicle counting
│   └── requirements-optional.txt  # optional, AGPL-3.0 (ultralytics)
├── data_pipeline/     # camera data collection, synthetic images, labelling
├── detection/         # YOLOv8-nano training, export, TensorRT engines
├── queue_estimation/  # ROI definitions, queue length (veh, m), occupancy
├── deploy/            # Jetson / Raspberry Pi deployment
└── tests/
```

### `prototypes/izum_live_detection.py`

*Initial prototype contributed by Demir Cücü.* It reads an MJPEG stream, runs YOLOv8n (COCO classes car, motorcycle, bus, truck), and shows the counts in an OpenCV window.

```bash
pip install -r edge/prototypes/requirements-optional.txt
cp .env.example .env    # set IZUM_STREAM_URL (and optionally EDGE_YOLO_WEIGHTS)
set -a; source .env; set +a
python edge/prototypes/izum_live_detection.py
```

- **Check IZUM's terms of use** before using any camera stream. The script does not impersonate a browser. Extra headers are optional (`IZUM_STREAM_USER_AGENT`, `IZUM_STREAM_REFERER`, empty by default). TLS verification is on unless `IZUM_STREAM_VERIFY_TLS=false`.
- Weights (`yolov8n.pt`) are never committed. Ultralytics downloads them on first use, or set `EDGE_YOLO_WEIGHTS`.
- The inference device is hard-coded to `"mps"` (Apple Silicon). Change it for other platforms.
- It uses COCO classes, not the project classes, and does not publish `queue_telemetry` yet.

## Licensing

**Ultralytics YOLOv8 is AGPL-3.0.** It is an **optional** dependency of `edge/` only. Never vendor or bundle Ultralytics code or weights into this repository. Distributing a product that includes it requires AGPL-3.0 compliance or an Ultralytics enterprise licence.

## Interfaces

- **Produces:** [`queue_telemetry`](../shared/contracts/schemas/queue_telemetry.schema.json) on MQTT `izmir/corridor/v1/edge/{intersection_id}/queue` (DRAFT).
- **Open questions:** see [`docs/interfaces.md`](../docs/interfaces.md#open-questions-by-owner) (questions 4, 6 and 7), and where inference runs (edge vs. central; see [`docs/architecture/`](../docs/architecture/README.md#open-questions-on-the-architecture)).

## Setup, run, test

See above for the prototype. Test commands will be added with `edge/tests/`.
