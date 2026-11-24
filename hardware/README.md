# hardware/: V2I, ESP32 relay board and fail-safe

- **Owner:** Tolga Uçar (EEE)
- **Status:** planned

## Purpose

The physical layer of the corridor:

- **V2I:** RF/LoRa (433/868 MHz) and BLE (2.4 GHz) transponders on emergency vehicles, authenticated with HMAC/AES. Range ≥300 m line of sight, latency under 50 ms.
- **ESP32 solid-state relay driver board:** 12 V DC logic driving 220 V AC lamps.
- **Fail-safe:** hardware green-conflict interlock (under 35 ms) and a 1000 ms hardware watchdog that falls back to flashing yellow.

## Scope

**Belongs here:** ESP32 firmware, transponder firmware, V2I protocol and key handling (keys are **never** committed), PCB design files, interlock and watchdog design, HIL test benches.

**Does not belong here:** signal timing decisions (`ai/`), message routing (`backend/`).

## Planned sub-structure

```
hardware/
├── firmware/esp32/   # relay driver firmware (PlatformIO / Arduino / ESP-IDF)
├── v2i/              # transponder + receiver firmware, protocol spec, HMAC/AES
├── pcb/              # schematics, layouts, BOM (source files; large exports via releases)
├── failsafe/         # interlock logic, watchdog, conflict matrix per intersection
└── tests/            # HIL test procedures and results
```

## Interfaces

- **Consumes:** [`signal_command`](../shared/contracts/schemas/signal_command.schema.json) through the backend (MQTT `control/{intersection_id}/signal_command`, DRAFT). Commands are requests: the interlock and the watchdog are authoritative.
- **Produces:** raw V2I emergency frames to the backend, which verifies them and publishes [`emergency_request`](../shared/contracts/schemas/emergency_request.schema.json); lamp-state feedback (payload TBD).
- **Open questions:** phase plan ids shared with SUMO, raw V2I frame format, lamp-state feedback payload. See [`docs/interfaces.md`](../docs/interfaces.md#open-questions-by-owner).

## Setup, run, test

To be added (toolchain, flashing, HIL procedure). Never commit keys. Use `.env` or device provisioning.
