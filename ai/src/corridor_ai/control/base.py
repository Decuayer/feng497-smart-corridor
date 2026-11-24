"""Controller interface.

A controller is called once per simulated second, before the second is
simulated. It returns signal requests ``{tls_id: target_green | None}`` for
the traffic lights it controls. Requests go to the PhaseMachine, which alone
enforces min/max green, yellow and all-red, so no controller can violate the
safety limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from corridor_ai.sim.core import CorridorSim


class Controller(Protocol):
    tls_ids: tuple[str, ...]

    def reset(self, sim: CorridorSim) -> None: ...

    def act(self, sim: CorridorSim) -> dict[str, int | None]: ...
