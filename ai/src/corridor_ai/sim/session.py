"""Thin wrapper around a SUMO process (libsumo in-process, or TraCI over a socket).

Only this module talks to libsumo/traci directly. Both backends expose the same
domain objects (``trafficlight``, ``lane``, ``vehicle``, ``simulation``), so the
rest of the code uses ``session.api.<domain>`` without caring which one is
active.

libsumo is much faster but allows only one simulation per process.
``sumo-gui`` requires TraCI.
"""

from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path
from typing import Any

_label_counter = itertools.count()
_LIBSUMO_ACTIVE = False


def sumo_binary(gui: bool = False) -> str:
    name = "sumo-gui" if gui else "sumo"
    home = os.environ.get("SUMO_HOME")
    if home:
        candidate = Path(home) / "bin" / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    try:  # pip eclipse-sumo package
        import sumo  # type: ignore[import-not-found]

        return str(Path(sumo.SUMO_HOME) / "bin" / name)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"cannot find '{name}'; install eclipse-sumo or set SUMO_HOME") from exc


class SumoSession:
    def __init__(self, backend: str = "libsumo", gui: bool = False) -> None:
        if gui and backend == "libsumo":
            backend = "traci"
        self.backend = backend
        self.gui = gui
        self.api: Any = None
        self._label: str | None = None

    def start(self, args: list[str]) -> None:
        global _LIBSUMO_ACTIVE
        cmd = [sumo_binary(self.gui), *args]
        if self.backend == "libsumo":
            import libsumo  # type: ignore[import-not-found]

            if _LIBSUMO_ACTIVE:
                raise RuntimeError("libsumo supports one simulation per process; close it first")
            libsumo.start(cmd)
            _LIBSUMO_ACTIVE = True
            self.api = libsumo
        else:
            import traci

            self._label = f"corridor_ai_{os.getpid()}_{next(_label_counter)}"
            traci.start(cmd, label=self._label)
            self.api = traci.getConnection(self._label)

    def step(self) -> None:
        self.api.simulationStep()

    @property
    def time(self) -> float:
        return float(self.api.simulation.getTime())

    def close(self) -> None:
        global _LIBSUMO_ACTIVE
        if self.api is None:
            return
        try:
            self.api.close()
        finally:
            if self.backend == "libsumo":
                _LIBSUMO_ACTIVE = False
            self.api = None
