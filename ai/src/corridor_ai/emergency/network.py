"""Static network data for route look-ahead.

A *movement* is a pair (incoming edge, outgoing edge) through a signalised
junction. For each movement we store the controlling light, its link indices,
the approach lanes that feed it (where a queue in front of an emergency vehicle
builds up), and the green phases that serve it. Priority green 'G' phases
are preferred; permissive 'g' phases are used only if no 'G' phase exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sumolib

from corridor_ai.sim.tls_model import load_tls_models


@dataclass(frozen=True)
class Movement:
    tls_id: str
    in_edge: str
    out_edge: str
    link_indices: tuple[int, ...]
    in_lanes: tuple[str, ...]
    phases: tuple[int, ...]  # green phases serving the movement, best first


@dataclass(frozen=True)
class RoadNetwork:
    movements: dict[tuple[str, str], Movement]
    edge_length: dict[str, float]
    edge_speed: dict[str, float]

    def free_time(self, edge: str, speed_factor: float) -> float:
        return self.edge_length[edge] / max(self.edge_speed[edge] * speed_factor, 0.1)


@lru_cache(maxsize=4)
def load_road_network(net_file: str | Path) -> RoadNetwork:
    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    models = load_tls_models(net_file)
    grouped: dict[tuple[str, str], dict] = {}
    for tls in net.getTrafficLights():
        for in_lane, out_lane, idx in tls.getConnections():
            key = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
            g = grouped.setdefault(key, {"tls": tls.getID(), "idx": set(), "lanes": set()})
            g["idx"].add(idx)
            g["lanes"].add(in_lane.getID())
    movements = {}
    for key, g in grouped.items():
        model = models[g["tls"]]
        prio = [k for k, s in enumerate(model.green_states) if any(s[i] == "G" for i in g["idx"])]
        perm = [k for k, s in enumerate(model.green_states) if any(s[i] in "Gg" for i in g["idx"])]
        movements[key] = Movement(
            tls_id=g["tls"],
            in_edge=key[0],
            out_edge=key[1],
            link_indices=tuple(sorted(g["idx"])),
            in_lanes=tuple(sorted(g["lanes"])),
            phases=tuple(prio or perm),
        )
    return RoadNetwork(
        movements=movements,
        edge_length={e.getID(): e.getLength() for e in net.getEdges()},
        edge_speed={e.getID(): e.getSpeed() for e in net.getEdges()},
    )
