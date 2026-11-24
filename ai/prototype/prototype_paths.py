"""Filesystem paths used by the legacy MARL prototype.

Resolved relative to this file, so the prototype scripts can be run from any
working directory (e.g. `cd ai/prototype && python evaluate.py`).
"""

from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parent
AI_DIR = PROTO_DIR.parent
FIXTURE_DIR = AI_DIR / "sim_fixtures" / "osm_alsancak_v0"

CONFIG_PATH = str(PROTO_DIR / "configs" / "intersections_config.json")
# SUMO reads gzipped networks directly.
NET_FILE = str(FIXTURE_DIR / "osm.net.xml.gz")
ROUTE_FILE = str(FIXTURE_DIR / "osm.passenger.trips.xml")
# Model weights are never committed (ai/checkpoints/ is gitignored).
MODEL_PATH = str(AI_DIR / "checkpoints" / "mappo_corridor.pt")
# Generated plots go to ai/runs/ (gitignored).
RUNS_DIR = AI_DIR / "runs" / "prototype"
