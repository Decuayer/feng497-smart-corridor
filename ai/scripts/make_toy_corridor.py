#!/usr/bin/env python3
"""Generate the PLACEHOLDER toy corridor fixture (ai/sim_fixtures/toy_corridor).

Six signalised 4-arm junctions J1..J6 on a straight east-west arterial.
Spacing 500 m and 250 m end stubs give exactly 3.0 km from W to E, the length
of the emergency-KPI pilot corridor. The arterial has 2 lanes per direction
and the side streets have 1, all at 50 km/h. Traffic-light programs are
netconvert's static defaults; at runtime they are only used to derive the set
of allowed green states (see ai/docs/signal_control.md).

Usage: python ai/scripts/make_toy_corridor.py   (needs netconvert on PATH)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "sim_fixtures" / "toy_corridor"
N_JUNCTIONS = 6
SPACING_M = 500.0
STUB_M = 250.0
SIDE_M = 300.0
ARTERIAL_LANES = 2
SIDE_LANES = 1
SPEED_MPS = 13.89  # 50 km/h


def _nodes() -> list[str]:
    rows = [f'    <node id="W" x="{-STUB_M}" y="0" type="priority"/>']
    for i in range(1, N_JUNCTIONS + 1):
        x = (i - 1) * SPACING_M
        rows.append(f'    <node id="J{i}" x="{x}" y="0" type="traffic_light"/>')
        rows.append(f'    <node id="N{i}" x="{x}" y="{SIDE_M}" type="priority"/>')
        rows.append(f'    <node id="S{i}" x="{x}" y="{-SIDE_M}" type="priority"/>')
    end_x = (N_JUNCTIONS - 1) * SPACING_M + STUB_M
    rows.append(f'    <node id="E" x="{end_x}" y="0" type="priority"/>')
    return rows


def _edge(eid: str, frm: str, to: str, lanes: int) -> str:
    return (
        f'    <edge id="{eid}" from="{frm}" to="{to}" numLanes="{lanes}" '
        f'speed="{SPEED_MPS}" priority="{3 if lanes > 1 else 1}"/>'
    )


def _edges() -> list[str]:
    chain = ["W"] + [f"J{i}" for i in range(1, N_JUNCTIONS + 1)] + ["E"]
    rows = []
    for a, b in zip(chain, chain[1:], strict=False):
        rows.append(_edge(f"{a}_{b}", a, b, ARTERIAL_LANES))
        rows.append(_edge(f"{b}_{a}", b, a, ARTERIAL_LANES))
    for i in range(1, N_JUNCTIONS + 1):
        for side in ("N", "S"):
            rows.append(_edge(f"{side}{i}_J{i}", f"{side}{i}", f"J{i}", SIDE_LANES))
            rows.append(_edge(f"J{i}_{side}{i}", f"J{i}", f"{side}{i}", SIDE_LANES))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nod = OUT / "toy.nod.xml"
    edg = OUT / "toy.edg.xml"
    nod.write_text("<nodes>\n" + "\n".join(_nodes()) + "\n</nodes>\n", encoding="utf-8")
    edg.write_text("<edges>\n" + "\n".join(_edges()) + "\n</edges>\n", encoding="utf-8")
    subprocess.run(
        [
            "netconvert",
            "--node-files",
            str(nod),
            "--edge-files",
            str(edg),
            "--output-file",
            str(OUT / "toy.net.xml.gz"),
            "--tls.default-type",
            "static",
            "--no-turnarounds",
            "true",
            "--no-internal-links",
            "false",
            "--output.street-names",
            "false",
            "--precision",
            "2",
        ],
        check=True,
    )
    print(f"wrote {OUT / 'toy.net.xml.gz'}")


if __name__ == "__main__":
    main()
