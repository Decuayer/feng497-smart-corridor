"""Seeded traffic demand for the placeholder fixtures.

* ``toy_od``: Poisson arrivals on a fixed origin-destination table for the
  toy corridor (rates below are ASSUMPTIONS for a moderately loaded arterial
  at scale 1.0).
* ``file``: an existing route/trip file. The scale is applied with SUMO's
  ``--scale`` option, and randomness comes from SUMO's ``--seed``.

Generated files are cached under ai/runs/_cache/routes (gitignored) and named
by fixture, scale, seed and duration, so the same inputs always give the same
file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from corridor_ai.common.paths import CACHE_DIR
from corridor_ai.common.registry import Fixture

# veh/h at scale 1.0 (ASSUMPTION, placeholder demand)
TOY_ARTERIAL_THROUGH_VPH = 500.0  # W->E and E->W each
TOY_ARTERIAL_TO_SIDE_VPH = 15.0  # from each arterial end to each side exit
TOY_SIDE_TO_END_VPH = 40.0  # from each side entry to each arterial end
TOY_SIDE_CROSS_VPH = 40.0  # from each side entry straight across
N_TOY_JUNCTIONS = 6


@dataclass(frozen=True)
class Demand:
    route_file: Path
    sumo_scale: float  # passed as --scale (1.0 when the scale is already in the file)


def _toy_od_rates() -> list[tuple[str, str, float]]:
    rates = [("W_J1", "J6_E", TOY_ARTERIAL_THROUGH_VPH), ("E_J6", "J1_W", TOY_ARTERIAL_THROUGH_VPH)]
    for i in range(1, N_TOY_JUNCTIONS + 1):
        for side, other in (("N", "S"), ("S", "N")):
            exit_edge = f"J{i}_{side}{i}"
            rates.append(("W_J1", exit_edge, TOY_ARTERIAL_TO_SIDE_VPH))
            rates.append(("E_J6", exit_edge, TOY_ARTERIAL_TO_SIDE_VPH))
            entry = f"{side}{i}_J{i}"
            rates.append((entry, "J6_E", TOY_SIDE_TO_END_VPH))
            rates.append((entry, "J1_W", TOY_SIDE_TO_END_VPH))
            rates.append((entry, f"J{i}_{other}{i}", TOY_SIDE_CROSS_VPH))
    return rates


def write_toy_routes(path: Path, scale: float, seed: int, end_s: int) -> Path:
    rng = np.random.default_rng(seed)
    trips: list[tuple[float, str, str]] = []
    for frm, to, vph in _toy_od_rates():
        rate = vph * scale / 3600.0
        if rate <= 0:
            continue
        t = rng.exponential(1.0 / rate)
        while t < end_s:
            trips.append((round(float(t), 2), frm, to))
            t += rng.exponential(1.0 / rate)
    trips.sort()
    lines = [
        "<routes>",
        '    <vType id="passenger" vClass="passenger"/>',
    ]
    for n, (depart, frm, to) in enumerate(trips):
        lines.append(
            f'    <trip id="v{n}" type="passenger" depart="{depart:.2f}" from="{frm}" to="{to}" '
            'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def build_demand(fixture: Fixture, scale: float, seed: int, end_s: int) -> Demand:
    kind = fixture.demand.get("type", "file")
    if kind == "toy_od":
        out = CACHE_DIR / "routes" / fixture.name / f"scale{scale:g}_seed{seed}_end{end_s}.rou.xml"
        if not out.exists():
            write_toy_routes(out, scale, seed, end_s)
        return Demand(out, 1.0)
    if kind == "file":
        return Demand(Path(fixture.demand["path"]), scale)
    raise ValueError(f"unknown demand type '{kind}' for fixture {fixture.name}")
