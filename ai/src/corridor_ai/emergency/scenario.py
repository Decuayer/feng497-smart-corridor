"""Simulated emergency vehicles for the PLACEHOLDER fixtures.

``ev_count`` emergency vehicles are inserted on the fixture's KPI corridor route.
Their times lie between warm-up + ``ev_first_after_warmup_s`` and
end - ``ev_last_before_end_s``, evenly spaced with seeded jitter. The same seed
gives the same insertion times, so preemption and no-preemption runs are
paired. EVs **obey traffic signals** (no SUMO bluelight device; approved
ASSUMPTION) and have a higher desired speed (speedFactor 1.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from corridor_ai.common.config import ExperimentConfig
from corridor_ai.common.paths import CACHE_DIR
from corridor_ai.common.registry import Fixture


def ev_departures(cfg: ExperimentConfig, seed: int, end_s: int) -> list[tuple[float, str]]:
    em = cfg.emergency
    start = cfg.sim.warmup_s + em.ev_first_after_warmup_s
    stop = end_s - em.ev_last_before_end_s
    if em.ev_count <= 0 or stop <= start:
        return []
    rng = np.random.default_rng(seed + 7_000_000)
    slot = (stop - start) / em.ev_count
    out = []
    for i in range(em.ev_count):
        t = start + (i + 0.5) * slot + rng.uniform(-0.25, 0.25) * slot
        vtype = "firebrigade" if rng.random() < em.ev_firebrigade_share else "emergency"
        out.append((round(float(t), 1), vtype))
    return out


def write_ev_routes(fixture: Fixture, cfg: ExperimentConfig, seed: int, end_s: int) -> Path:
    if not fixture.corridor_edges:
        raise ValueError(f"fixture {fixture.name} has no corridor kpi_route")
    sf = cfg.emergency.ev_speed_factor
    path = (
        CACHE_DIR
        / "ev_routes"
        / fixture.name
        / f"seed{seed}_end{end_s}_n{cfg.emergency.ev_count}_sf{sf:g}.rou.xml"
    )
    if path.exists():
        return path
    edges = " ".join(fixture.corridor_edges)
    lines = [
        "<routes>",
        f'    <vType id="emergency" vClass="emergency" guiShape="emergency" speedFactor="{sf}" '
        'speedDev="0" color="1,1,1"/>',
        f'    <vType id="firebrigade" vClass="emergency" guiShape="firebrigade" speedFactor="{sf}" '
        'speedDev="0" color="1,0,0"/>',
        f'    <route id="ev_corridor" edges="{edges}"/>',
    ]
    for i, (t, vtype) in enumerate(ev_departures(cfg, seed, end_s)):
        lines.append(
            f'    <vehicle id="ev{i}" type="{vtype}" route="ev_corridor" depart="{t:.1f}" '
            'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", "utf-8")
    tmp.replace(path)
    return path
