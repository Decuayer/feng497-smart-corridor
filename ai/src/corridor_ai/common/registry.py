"""Intersection registry (shared/contracts) and simulation fixtures (ai/configs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from corridor_ai.common.paths import AI_DIR, CONFIG_DIR, CONTRACTS_DIR, resolve


@dataclass(frozen=True)
class Intersection:
    id: str
    display_name: str
    sumo_tls_id: str | None


@dataclass(frozen=True)
class Fixture:
    name: str
    net: Path
    demand: dict[str, Any]
    agents: dict[str, str]  # contract id -> TLS id (insertion order = corridor order)
    description: str = ""
    corridor_routes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    kpi_route: str | None = None
    placeholder: bool = True

    @property
    def corridor_edges(self) -> tuple[str, ...]:
        """Edges of the emergency-KPI corridor route (empty if none defined)."""
        return self.corridor_routes.get(self.kpi_route or "", ())

    @property
    def disclaimer(self) -> str:
        return (
            f"Fixture: {self.name}. Relative improvement on a placeholder fixture, "
            "not the real digital twin."
        )


@lru_cache(maxsize=1)
def load_intersections() -> tuple[Intersection, ...]:
    data = yaml.safe_load((CONTRACTS_DIR / "intersections.yaml").read_text(encoding="utf-8"))
    return tuple(
        Intersection(
            id=i["id"],
            display_name=i["display_name"],
            sumo_tls_id=None if i.get("sumo_tls_id") in (None, "TBD") else i["sumo_tls_id"],
        )
        for i in data["intersections"]
    )


def intersection_ids() -> list[str]:
    return [i.id for i in load_intersections()]


def load_fixture(name: str, fixtures_file: Path | None = None) -> Fixture:
    data = yaml.safe_load((fixtures_file or CONFIG_DIR / "fixtures.yaml").read_text("utf-8"))
    if name not in data:
        raise KeyError(f"unknown fixture '{name}'; available: {sorted(data)}")
    raw = data[name]
    agents = {str(k): str(v) for k, v in raw["agents"].items()}
    known = set(intersection_ids())
    bad = set(agents) - known
    if bad:
        raise ValueError(f"fixture '{name}' uses unknown intersection ids {sorted(bad)}")
    demand = dict(raw.get("demand", {"type": "file"}))
    for key in ("path", "background"):
        if key in demand:
            demand[key] = str(resolve(demand[key], AI_DIR))
    corridor = raw.get("corridor") or {}
    if "file" in corridor:
        data_c = yaml.safe_load(resolve(corridor["file"], AI_DIR).read_text("utf-8"))
        routes = {k: tuple(v["edges"]) for k, v in data_c["routes"].items()}
    else:
        routes = {k: tuple(v) for k, v in (corridor.get("routes") or {}).items()}
    kpi_route = corridor.get("kpi_route")
    if kpi_route is not None and kpi_route not in routes:
        raise ValueError(f"fixture '{name}': kpi_route '{kpi_route}' not in corridor routes")
    return Fixture(
        name=name,
        net=resolve(raw["net"], AI_DIR),
        demand=demand,
        agents=agents,
        description=str(raw.get("description", "")).strip(),
        corridor_routes=routes,
        kpi_route=kpi_route,
        placeholder=bool(raw.get("placeholder", True)),
    )
