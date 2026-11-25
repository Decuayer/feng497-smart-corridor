"""Seeded traffic demand for the placeholder fixtures.

* ``toy_od``: Poisson arrivals on a fixed origin-destination table for the
  toy corridor (rates below are ASSUMPTIONS for a moderately loaded arterial
  at scale 1.0).
* ``file``: an existing route/trip file. The scale is applied with SUMO's
  ``--scale`` option, and randomness comes from SUMO's ``--seed``.
* ``corridor_od`` (PLACEHOLDER, ai/ fixtures only): the fixture's background
  trips, thinned/duplicated by the scale with the seed, plus Poisson through
  traffic on every corridor route and cross traffic from every incoming
  approach of every agent light that is not on a corridor route. This makes
  sure demand reaches all six controlled intersections. All rates are
  ASSUMPTIONS and the scale is built into the file.

Generated files are cached under ai/runs/_cache/routes (gitignored) and named
by fixture, scale, seed and duration, so the same inputs always give the same
file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sumolib

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


def _poisson_times(rng: np.random.Generator, vph: float, end_s: int) -> list[float]:
    rate = vph / 3600.0
    if rate <= 0:
        return []
    out, t = [], rng.exponential(1.0 / rate)
    while t < end_s:
        out.append(round(float(t), 2))
        t += rng.exponential(1.0 / rate)
    return out


def corridor_cross_routes(fixture: Fixture) -> dict[str, list[tuple[str, str]]]:
    """For each agent light: [(approach edge, exit edge)] for approaches off the corridor."""
    net = sumolib.net.readNet(str(fixture.net), withPrograms=True)
    on_corridor = {e for edges in fixture.corridor_routes.values() for e in edges}
    out: dict[str, list[tuple[str, str]]] = {}
    for cid, tls in fixture.agents.items():
        pairs = sorted(
            {
                (i.getEdge().getID(), o.getEdge().getID())
                for i, o, _ in net.getTLS(tls).getConnections()
            }
        )
        out[cid] = [(a, b) for a, b in pairs if a not in on_corridor]
    return out


def write_corridor_routes(
    fixture: Fixture, path: Path, scale: float, seed: int, end_s: int
) -> Path:
    params = fixture.demand
    rng = np.random.default_rng(seed)
    entries: list[tuple[float, str]] = []

    # background trips: keep each trip floor(scale) times, plus once more with prob frac(scale)
    whole, frac = int(np.floor(scale)), scale - np.floor(scale)
    root = ET.parse(params["background"]).getroot()
    vtypes = [ET.tostring(v, encoding="unicode").strip() for v in root.iter("vType")]
    for trip in root.iter("trip"):
        depart = float(trip.get("depart", 0))
        if depart >= end_s:
            continue
        copies = whole + int(rng.random() < frac)
        for c in range(copies):
            attrs = dict(trip.attrib)
            attrs["id"] = f"bg_{attrs['id']}_{c}"
            attrs["depart"] = f"{depart:.2f}"
            text = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            entries.append((depart, f"    <trip {text}/>"))

    n = 0
    for name, vph in (params.get("through_vph") or {}).items():
        edges = " ".join(fixture.corridor_routes[name])
        for t in _poisson_times(rng, vph * scale, end_s):
            entries.append(
                (
                    t,
                    f'    <vehicle id="thr_{name}_{n}" depart="{t:.2f}" '
                    f'departLane="best" departSpeed="max"><route edges="{edges}"/></vehicle>',
                )
            )
            n += 1
    cross_vph = float(params.get("cross_vph_per_approach", 0.0))
    for cid, pairs in corridor_cross_routes(fixture).items():
        by_approach: dict[str, list[str]] = {}
        for a, b in pairs:
            by_approach.setdefault(a, []).append(b)
        for a, exits in by_approach.items():
            for b in exits:
                for t in _poisson_times(rng, cross_vph * scale / len(exits), end_s):
                    entries.append(
                        (
                            t,
                            f'    <vehicle id="crs_{cid}_{n}" depart="{t:.2f}" '
                            'departLane="best" departSpeed="max">'
                            f'<route edges="{a} {b}"/></vehicle>',
                        )
                    )
                    n += 1
    entries.sort(key=lambda x: x[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = ["    " + v for v in vtypes] + [e for _, e in entries]
    tmp.write_text("<routes>\n" + "\n".join(body) + "\n</routes>\n", "utf-8")
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
    if kind == "corridor_od":
        out = CACHE_DIR / "routes" / fixture.name / f"scale{scale:g}_seed{seed}_end{end_s}.rou.xml"
        if not out.exists():
            write_corridor_routes(fixture, out, scale, seed, end_s)
        return Demand(out, 1.0)
    raise ValueError(f"unknown demand type '{kind}' for fixture {fixture.name}")
