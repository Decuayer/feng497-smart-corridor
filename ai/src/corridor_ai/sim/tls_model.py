"""Static description of every traffic light in a SUMO network.

Built once from the .net.xml(.gz) with sumolib (pure Python, no SUMO process):

* ``green_states``: the allowed green signal states. These are the green phases of
  the network's own program, deduplicated in program order. Controllers can only
  ever request one of these, so they never invent a phase combination.
* ``planned_green_s``: the original duration of each green phase. Used for the
  fixed-time plan after clamping to the safety limits.
* ``lanes_by_phase``: incoming lanes that get green (G/g) in each green phase.
* ``foes``: pairs of link indices that conflict at the same junction, read from
  the junction right-of-way matrix (``<request foes=...>``). Used by the
  independent SafetyMonitor.

Phases containing yellow are transitions and are dropped. Phases without any
green or yellow are dropped as well: these are all-red clearances, or red-only
pedestrian phases. No pedestrians are simulated in these fixtures, so
(ASSUMPTION) red-only pedestrian phases are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sumolib

GREEN_CHARS = frozenset("Gg")
YELLOW_CHARS = frozenset("yY")


def is_green_state(state: str) -> bool:
    return any(c in GREEN_CHARS for c in state) and not any(c in YELLOW_CHARS for c in state)


@dataclass(frozen=True)
class TLSModel:
    tls_id: str
    n_links: int
    green_states: tuple[str, ...]
    planned_green_s: tuple[float, ...]
    link_in_lanes: tuple[tuple[str, ...], ...]  # per link index: incoming lane ids
    lanes_by_phase: tuple[tuple[str, ...], ...]
    incoming_lanes: tuple[str, ...]
    lane_lengths: dict[str, float]
    foes: frozenset[tuple[int, int]]  # (i, j) with i < j

    @property
    def n_phases(self) -> int:
        return len(self.green_states)

    def conflicting_greens(self, state: str) -> list[tuple[int, int]]:
        """Foe link pairs that both show priority green 'G' in `state`."""
        return [(i, j) for (i, j) in self.foes if state[i] == "G" and state[j] == "G"]


def _first_program(tls: sumolib.net.TLS):
    programs = tls.getPrograms()
    if not programs:
        return None
    return programs["0"] if "0" in programs else next(iter(programs.values()))


def _build_one(tls: sumolib.net.TLS) -> TLSModel:
    connections = tls.getConnections()  # [inLane, outLane, linkIndex]
    n_links = max(c[2] for c in connections) + 1
    link_in: list[set[str]] = [set() for _ in range(n_links)]
    # (node id, junction link index) per TLS link index, for the foe matrix
    junction_links: list[set[tuple[str, int]]] = [set() for _ in range(n_links)]
    nodes: dict[str, sumolib.net.node.Node] = {}
    lane_lengths: dict[str, float] = {}

    for in_lane, out_lane, idx in connections:
        link_in[idx].add(in_lane.getID())
        lane_lengths[in_lane.getID()] = in_lane.getLength()
        node = in_lane.getEdge().getToNode()
        nodes[node.getID()] = node
        for conn in in_lane.getOutgoing():
            if conn.getToLane() == out_lane and conn.getTLLinkIndex() == idx:
                jidx = node.getLinkIndex(conn)
                if jidx >= 0:
                    junction_links[idx].add((node.getID(), jidx))

    foes: set[tuple[int, int]] = set()
    for i in range(n_links):
        for j in range(i + 1, n_links):
            for node_i, ji in junction_links[i]:
                for node_j, jj in junction_links[j]:
                    if node_i != node_j:
                        continue
                    node = nodes[node_i]
                    try:
                        if node.areFoes(ji, jj) or node.areFoes(jj, ji):
                            foes.add((i, j))
                    except (KeyError, IndexError):  # node without right-of-way info
                        pass

    program = _first_program(tls)
    greens: list[str] = []
    durations: list[float] = []
    if program is not None:
        for phase in program.getPhases():
            state = phase.state
            if len(state) != n_links:
                state = state[:n_links].ljust(n_links, "r")
            if is_green_state(state) and state not in greens:
                greens.append(state)
                durations.append(float(phase.duration))
    if not greens:  # degenerate program: keep everything green that exists
        greens = ["G" * n_links]
        durations = [60.0]

    lanes_by_phase = tuple(
        tuple(
            sorted({lane for k, c in enumerate(state) if c in GREEN_CHARS for lane in link_in[k]})
        )
        for state in greens
    )
    incoming = tuple(sorted({lane for lanes in link_in for lane in lanes}))
    return TLSModel(
        tls_id=tls.getID(),
        n_links=n_links,
        green_states=tuple(greens),
        planned_green_s=tuple(durations),
        link_in_lanes=tuple(tuple(sorted(s)) for s in link_in),
        lanes_by_phase=lanes_by_phase,
        incoming_lanes=incoming,
        lane_lengths=lane_lengths,
        foes=frozenset(foes),
    )


@lru_cache(maxsize=8)
def load_tls_models(net_file: str | Path) -> dict[str, TLSModel]:
    """Return {tls_id: TLSModel} for every traffic light in the network."""
    net = sumolib.net.readNet(str(net_file), withPrograms=True, withFoes=True)
    return {tls.getID(): _build_one(tls) for tls in net.getTrafficLights()}
