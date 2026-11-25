"""Episode metrics from SUMO tripinfo output."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def parse_tripinfo(
    path: str | Path, min_depart_s: float = 0.0, exclude_vtypes: tuple[str, ...] = ()
) -> dict[str, float]:
    """Aggregate tripinfo for vehicles that departed at or after `min_depart_s`.

    Unfinished vehicles (written with --tripinfo-output.write-unfinished) are
    counted separately. Their durations are included in the travel-time and
    waiting means, so that gridlock cannot hide delay.
    CO2 is reported in g/km: sum(CO2_abs [mg]) / sum(routeLength [m]).
    """
    n = n_arrived = n_unfinished = 0
    dur = wait = loss = stops = route_len = co2_mg = 0.0
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag != "tripinfo":
            continue
        if float(el.get("depart", "-1")) < min_depart_s or el.get("vType") in exclude_vtypes:
            el.clear()
            continue
        n += 1
        if float(el.get("arrival", "-1")) >= 0:
            n_arrived += 1
        else:
            n_unfinished += 1
        dur += float(el.get("duration", 0))
        wait += float(el.get("waitingTime", 0))
        loss += float(el.get("timeLoss", 0))
        stops += float(el.get("waitingCount", 0))
        route_len += float(el.get("routeLength", 0))
        em = el.find("emissions")
        if em is not None:
            co2_mg += float(em.get("CO2_abs", 0))
        el.clear()
    safe = max(n, 1)
    return {
        "trips": float(n),
        "arrived": float(n_arrived),
        "unfinished": float(n_unfinished),
        "mean_travel_time_s": dur / safe,
        "mean_waiting_time_s": wait / safe,
        "mean_time_loss_s": loss / safe,
        "mean_stops": stops / safe,
        "co2_g_per_km": (co2_mg / route_len) if route_len > 0 else 0.0,
    }


def count_tripinfo_arrivals(path: str | Path) -> int:
    count = 0
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag == "tripinfo":
            if float(el.get("arrival", "-1")) >= 0:
                count += 1
            el.clear()
    return count


def tripinfo_records(path: str | Path, vtypes: tuple[str, ...]) -> list[dict[str, float | str]]:
    """Per-vehicle tripinfo rows for the given vehicle types (e.g. emergency vehicles)."""
    rows: list[dict[str, float | str]] = []
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag != "tripinfo":
            continue
        if el.get("vType") in vtypes:
            rows.append(
                {
                    "id": el.get("id", ""),
                    "vtype": el.get("vType", ""),
                    "depart_s": float(el.get("depart", 0)),
                    "duration_s": float(el.get("duration", 0)),
                    "depart_delay_s": float(el.get("departDelay", 0)),
                    "waiting_s": float(el.get("waitingTime", 0)),
                    "stops": float(el.get("waitingCount", 0)),
                    "route_length_m": float(el.get("routeLength", 0)),
                    "arrived": float(float(el.get("arrival", "-1")) >= 0),
                }
            )
        el.clear()
    return rows
