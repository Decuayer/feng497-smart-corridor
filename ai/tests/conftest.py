"""Shared pytest fixtures.

Integration tests (marked ``sumo``) need the SUMO binaries/libsumo from the pip
``eclipse-sumo`` package. CI runs ``-m "not sumo"``.
"""

from __future__ import annotations

import os

import pytest

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.tls_model import TLSModel


def _ensure_sumo_home() -> None:
    if os.environ.get("SUMO_HOME") and os.path.isdir(os.environ["SUMO_HOME"]):
        return
    try:
        import sumo  # type: ignore[import-not-found]

        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        pass


_ensure_sumo_home()


@pytest.fixture
def limits() -> SignalLimits:
    return SignalLimits()


def make_model(green_states, foes=(), tls_id="T") -> TLSModel:
    """Synthetic TLSModel for unit tests (one incoming lane per link)."""
    n = len(green_states[0])
    link_in = tuple((f"L{i}_0",) for i in range(n))
    lanes_by_phase = tuple(
        tuple(f"L{i}_0" for i, c in enumerate(s) if c in "Gg") for s in green_states
    )
    lanes = tuple(f"L{i}_0" for i in range(n))
    return TLSModel(
        tls_id=tls_id,
        n_links=n,
        green_states=tuple(green_states),
        planned_green_s=tuple(30.0 for _ in green_states),
        link_in_lanes=link_in,
        lanes_by_phase=lanes_by_phase,
        incoming_lanes=lanes,
        lane_lengths={ln: 75.0 for ln in lanes},  # capacity 10 vehicles
        foes=frozenset(foes),
    )


@pytest.fixture
def two_phase_model() -> TLSModel:
    # links 0,1 = north-south; links 2,3 = east-west; NS and EW conflict
    return make_model(["GGrr", "rrGG"], foes={(0, 2), (0, 3), (1, 2), (1, 3)})


@pytest.fixture
def three_phase_model() -> TLSModel:
    # links 2 (g, permissive) continues green in phases 0 and 1
    return make_model(["GGgrr", "rrGrr", "rrrGG"], foes={(0, 3), (1, 4), (2, 3)})
