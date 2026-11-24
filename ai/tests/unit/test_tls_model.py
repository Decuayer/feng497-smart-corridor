"""TLS model extraction from the fixture networks (sumolib only, no SUMO process)."""

import pytest

from corridor_ai.common.registry import load_fixture
from corridor_ai.sim.tls_model import is_green_state, load_tls_models


@pytest.fixture(scope="module")
def toy():
    return load_tls_models(load_fixture("toy_corridor").net)


@pytest.fixture(scope="module")
def osm():
    return load_tls_models(load_fixture("osm_alsancak_v0").net)


def test_toy_has_six_two_phase_lights(toy):
    assert sorted(toy) == [f"J{i}" for i in range(1, 7)]
    assert all(m.n_phases == 2 and m.foes for m in toy.values())


@pytest.mark.parametrize("fixture", ["toy", "osm"])
def test_original_green_states_have_no_conflicts(fixture, request):
    models = request.getfixturevalue(fixture)
    for m in models.values():
        for state in m.green_states:
            assert is_green_state(state)
            assert m.conflicting_greens(state) == [], (m.tls_id, state)


def test_union_of_phases_is_detected_as_conflict(toy):
    m = toy["J1"]
    union = "".join(
        "G" if "G" in (a, b) else c
        for a, b, c in zip(*m.green_states, m.green_states[0], strict=True)
    )
    assert m.conflicting_greens(union)


def test_osm_agents_exist_with_expected_phases(osm):
    fx = load_fixture("osm_alsancak_v0")
    assert len(osm) == 29
    phases = {cid: osm[tls].n_phases for cid, tls in fx.agents.items()}
    assert phases == {
        "alsancak_station": 2,
        "lozan": 4,
        "montro": 3,
        "basmane": 2,
        "cankaya": 2,
        "konak_pier": 4,
    }


def test_lanes_by_phase_subset_of_incoming(toy):
    for m in toy.values():
        for lanes in m.lanes_by_phase:
            assert lanes and set(lanes) <= set(m.incoming_lanes)
