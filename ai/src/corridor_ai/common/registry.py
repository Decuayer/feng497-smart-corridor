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
    corridor_edges: tuple[str, ...] = field(default_factory=tuple)
    placeholder: bool = True

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
    if "path" in demand:
        demand["path"] = str(resolve(demand["path"], AI_DIR))
    return Fixture(
        name=name,
        net=resolve(raw["net"], AI_DIR),
        demand=demand,
        agents=agents,
        description=str(raw.get("description", "")).strip(),
        corridor_edges=tuple((raw.get("corridor") or {}).get("edges", ())),
        placeholder=bool(raw.get("placeholder", True)),
    )
