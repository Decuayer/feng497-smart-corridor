"""Safety-constrained signal phase machine for one traffic light.

The machine is the *only* component that decides what a traffic light displays.
Controllers (fixed-time, Webster, MARL, emergency preemption) only call
:meth:`PhaseMachine.request` with a target green phase. Each simulated second
the machine returns the signal state to show, and the simulation core sends it
to SUMO with ``trafficlight.setRedYellowGreenState``. That call puts the light
on SUMO's "online" program, which never advances by itself. This removes the
prototype bug where SUMO's own program kept switching phases.

Timing rules (enforced here, checked independently by SafetyMonitor):

* a green phase is shown for at least ``min_green_s`` (10 s);
* a green phase is shown for at most ``max_green_s`` (60 s). At the limit the
  machine switches to the pending request, or else to the next phase in cyclic
  order;
* every change between green phases goes green -> yellow (exactly 3 s) ->
  all-red (exactly 1 s) -> new green;
* requests that arrive too early, or during a transition, are kept as pending
  and served as soon as it is legal.

A traffic light with a single green phase stays green (it has no conflicting
phase to serve) and is exempt from the max-green rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from corridor_ai.common.config import SignalLimits
from corridor_ai.sim.tls_model import GREEN_CHARS

GREEN, YELLOW, ALL_RED = "green", "yellow", "all_red"


def _keeps_green(f: str, t: str) -> bool:
    """A link keeps its green through a transition only if it is green in both
    phases and is not downgraded from protected 'G' to permissive 'g'."""
    return f in GREEN_CHARS and t in GREEN_CHARS and not (f == "G" and t == "g")


def transition_states(from_state: str, to_state: str, mode: str) -> tuple[str, str]:
    """(yellow state, all-red state) for a change from `from_state` to `to_state`.

    ``full``: every link that is green turns yellow, then every link is red.
    ``conflicting_only``: links that keep their green (see _keeps_green) stay
    green throughout. If no link would show yellow (nothing loses its green),
    the full sequence is used, so every change still has a visible 3 s yellow
    and 1 s all-red.
    """
    pairs = list(zip(from_state, to_state, strict=True))
    if mode == "conflicting_only":
        yellow = "".join(
            f if _keeps_green(f, t) else ("y" if f in GREEN_CHARS else "r") for f, t in pairs
        )
        if "y" in yellow:
            red = "".join(f if _keeps_green(f, t) else "r" for f, t in pairs)
            return yellow, red
    yellow = "".join("y" if f in GREEN_CHARS else "r" for f, _ in pairs)
    return yellow, "r" * len(from_state)


def yellow_state(from_state: str, to_state: str, mode: str) -> str:
    return transition_states(from_state, to_state, mode)[0]


def all_red_state(from_state: str, to_state: str, mode: str) -> str:
    return transition_states(from_state, to_state, mode)[1]


@dataclass
class _Transition:
    from_phase: int
    to_phase: int


class PhaseMachine:
    def __init__(
        self,
        tls_id: str,
        green_states: tuple[str, ...] | list[str],
        limits: SignalLimits,
        initial_phase: int = 0,
    ) -> None:
        if not green_states:
            raise ValueError(f"{tls_id}: at least one green state required")
        if len({len(s) for s in green_states}) != 1:
            raise ValueError(f"{tls_id}: green states differ in length")
        self.tls_id = tls_id
        self.green_states = tuple(green_states)
        self.limits = limits
        self.phase = initial_phase  # current green, or target green while in transition
        self.mode = GREEN
        self.t_in_mode = 0  # seconds already displayed in the current mode
        self.green_elapsed = 0  # seconds the current green has been displayed
        self.pending: int | None = None
        self._transition: _Transition | None = None
        # seconds since each phase was last displayed as green
        self.time_since_served = [0] * self.n_phases
        self.state = self.green_states[self.phase]
        self.switches = 0

    # ------------------------------------------------------------------ API
    @property
    def n_phases(self) -> int:
        return len(self.green_states)

    @property
    def in_transition(self) -> bool:
        return self.mode != GREEN

    @property
    def min_green_satisfied(self) -> bool:
        return self.mode == GREEN and self.green_elapsed >= self.limits.min_green_s

    def request(self, target: int | None) -> None:
        """Request green phase `target`; None (or the current phase) means keep."""
        if target is None:
            if self.mode == GREEN:
                self.pending = None
            return
        if not 0 <= target < self.n_phases:
            raise ValueError(f"{self.tls_id}: phase {target} out of range 0..{self.n_phases - 1}")
        if self.mode == GREEN and target == self.phase:
            self.pending = None
        else:
            self.pending = target

    def next_state(self) -> str:
        """Advance one second and return the state to display during that second."""
        lim = self.limits
        if self.mode == GREEN and self.n_phases > 1:
            wants_switch = self.pending is not None and self.pending != self.phase
            if wants_switch and self.green_elapsed >= lim.min_green_s:
                self._start_transition(self.pending)  # type: ignore[arg-type]
            elif self.green_elapsed >= lim.max_green_s:
                target = (
                    self.pending
                    if wants_switch
                    else (self.phase + 1) % self.n_phases  # ASSUMPTION: forced -> next phase
                )
                self._start_transition(target)
        elif self.mode == YELLOW and self.t_in_mode >= lim.yellow_s:
            self.mode, self.t_in_mode = ALL_RED, 0
        elif self.mode == ALL_RED and self.t_in_mode >= lim.all_red_s:
            assert self._transition is not None
            self.phase = self._transition.to_phase
            self._transition = None
            self.mode, self.t_in_mode, self.green_elapsed = GREEN, 0, 0
            if self.pending == self.phase:
                self.pending = None

        self.state = self._display()
        # account for the second that is about to be simulated
        self.t_in_mode += 1
        if self.mode == GREEN:
            self.green_elapsed += 1
        for k in range(self.n_phases):
            if self.mode == GREEN and k == self.phase:
                self.time_since_served[k] = 0
            else:
                self.time_since_served[k] += 1
        return self.state

    # ------------------------------------------------------------ internals
    def _start_transition(self, target: int) -> None:
        self._transition = _Transition(self.phase, target)
        self.phase = target
        self.mode, self.t_in_mode = YELLOW, 0
        self.switches += 1

    def _display(self) -> str:
        if self.mode == GREEN:
            return self.green_states[self.phase]
        assert self._transition is not None
        f = self.green_states[self._transition.from_phase]
        t = self.green_states[self._transition.to_phase]
        yellow, red = transition_states(f, t, self.limits.all_red_mode)
        return yellow if self.mode == YELLOW else red

    @property
    def current_green(self) -> int | None:
        """Index of the green currently displayed (None during a transition)."""
        return self.phase if self.mode == GREEN else None
