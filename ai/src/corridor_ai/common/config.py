"""Typed configuration loaded from YAML.

Every experiment parameter lives in ai/configs/*.yaml. Files are deep-merged in
order (base.yaml first), then CLI overrides are applied. Unknown keys raise an
error so typos cannot silently fall back to defaults.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from corridor_ai.common.paths import CONFIG_DIR, REPO_ROOT

# --------------------------------------------------------------------------
# Signal safety limits (project specification). These are the hard limits
# every controller must respect; configs may not relax them.
# --------------------------------------------------------------------------
SPEC_MIN_GREEN_S = 10
SPEC_MAX_GREEN_S = 60
SPEC_YELLOW_S = 3
SPEC_ALL_RED_S = 1


@dataclass(frozen=True)
class SignalLimits:
    min_green_s: int = SPEC_MIN_GREEN_S
    max_green_s: int = SPEC_MAX_GREEN_S
    yellow_s: int = SPEC_YELLOW_S
    all_red_s: int = SPEC_ALL_RED_S
    # "full": every link red during all-red (literal reading of the spec, default).
    # "conflicting_only": links green in both phases keep their green.
    all_red_mode: str = "full"

    def __post_init__(self) -> None:
        if self.min_green_s < SPEC_MIN_GREEN_S:
            raise ValueError(f"min_green_s must be >= {SPEC_MIN_GREEN_S}")
        if not (self.min_green_s <= self.max_green_s <= SPEC_MAX_GREEN_S):
            raise ValueError(f"need min_green_s <= max_green_s <= {SPEC_MAX_GREEN_S}")
        if self.yellow_s != SPEC_YELLOW_S:
            raise ValueError(f"yellow_s must be exactly {SPEC_YELLOW_S}")
        if self.all_red_s != SPEC_ALL_RED_S:
            raise ValueError(f"all_red_s must be exactly {SPEC_ALL_RED_S}")
        if self.all_red_mode not in ("full", "conflicting_only"):
            raise ValueError("all_red_mode must be 'full' or 'conflicting_only'")


@dataclass(frozen=True)
class SimConfig:
    backend: str = "libsumo"  # "libsumo" (fast, in-process) or "traci"
    gui: bool = False
    warmup_s: int = 300
    horizon_s: int = 3600  # controlled period after warm-up
    demand_scale: float = 1.0
    time_to_teleport_s: int = 300
    waiting_time_memory_s: int = 1000
    write_tripinfo: bool = True
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in ("libsumo", "traci"):
            raise ValueError("backend must be 'libsumo' or 'traci'")


@dataclass(frozen=True)
class RewardConfig:
    alpha: float = 1.0  # weight of cumulative waiting time W
    beta: float = 0.5  # weight of stops S
    stop_definition: str = "halted"  # "halted" | "stop_events" (open question)
    w_ref_s: float = 100.0  # normaliser for W (vehicle-seconds per lane)
    s_ref: float = 5.0  # normaliser for S (vehicles per lane)
    normalize_by_lanes: bool = True

    def __post_init__(self) -> None:
        if self.stop_definition not in ("halted", "stop_events"):
            raise ValueError("stop_definition must be 'halted' or 'stop_events'")


@dataclass(frozen=True)
class EnvConfig:
    decision_interval_s: int = 5
    k_max: int = 0  # 0 = derive from the fixture's agents
    phase_order: str = "free"  # "free" | "cyclic"
    max_red_s: int | None = 120  # starvation guard (ASSUMPTION); None disables
    vehicle_length_m: float = 7.5  # lane capacity = length / this
    wait_norm_s: float = 120.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.phase_order not in ("free", "cyclic"):
            raise ValueError("phase_order must be 'free' or 'cyclic'")
        if self.decision_interval_s < 1:
            raise ValueError("decision_interval_s must be >= 1")


@dataclass(frozen=True)
class PPOConfig:
    episodes: int = 20
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)
    device: str = "cpu"
    # demand scale sampled uniformly per training episode (domain randomisation)
    train_scale_min: float = 0.8
    train_scale_max: float = 1.3
    eval_every: int = 0  # 0 = no periodic greedy evaluation during training
    eval_seed: int = 999
    checkpoint_every: int = 10


@dataclass(frozen=True)
class EmergencyConfig:
    """Proactive emergency green corridor (TASK-302); see ai/docs/emergency_corridor.md."""

    enabled: bool = False  # False = observe emergency vehicles only (baseline runs)
    lookahead: int = 2  # pre-clear the current + next `lookahead` agent intersections
    startup_lost_s: float = 2.0  # queue-flush model: startup + n_queued * headway
    saturation_headway_s: float = 2.0
    margin_s: float = 3.0
    min_speed_frac: float = 0.6  # nominal ETA uses max(EMA speed, frac * free speed)
    reservice: bool = True  # leave a held green early if it would exceed max green
    # vehicles counted for the flush time: "vehicles_ahead" (all vehicles between the EV and
    # the stop line, per lane; default) or "halted" (only queued vehicles)
    flush_count: str = "vehicles_ahead"
    # hold budget: a pre-clear held this long without the EV passing is released
    # ("hold_timeout"); that EV/intersection may then only re-activate on its nominal
    # (observed-speed) ETA. Prevents starving cross streets for a blocked EV. ASSUMPTION.
    max_hold_s: int = 90
    ev_vtypes: tuple[str, ...] = ("emergency", "firebrigade")
    # simulated emergency vehicles (placeholder scenario; EVs obey signals)
    ev_speed_factor: float = 1.3
    ev_count: int = 10
    ev_first_after_warmup_s: int = 120
    ev_last_before_end_s: int = 900
    ev_firebrigade_share: float = 0.3
    # recovery measurement (vs the matched run without preemption)
    recovery_tolerance_frac: float = 0.1
    recovery_tolerance_abs: float = 2.0
    recovery_sustain_s: int = 60
    recovery_smooth_s: int = 30
    recovery_bound_s: int = 180

    def __post_init__(self) -> None:
        if self.lookahead < 0:
            raise ValueError("lookahead must be >= 0")
        if self.flush_count not in ("vehicles_ahead", "halted"):
            raise ValueError("flush_count must be 'vehicles_ahead' or 'halted'")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    fixture: str = "toy_corridor"
    seed: int = 0
    limits: SignalLimits = field(default_factory=SignalLimits)
    sim: SimConfig = field(default_factory=SimConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build(cls: type, data: dict[str, Any] | None, path: str = "") -> Any:
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown config key(s) at '{path or cls.__name__}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{path}.{name}".strip("."))
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_by_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _find(path: str | Path) -> Path:
    """Resolve a config path against the CWD first, then the repository root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_experiment_config(
    *paths: str | Path,
    overrides: dict[str, Any] | None = None,
    include_base: bool = True,
) -> ExperimentConfig:
    """Load base.yaml + the given files + dotted-key overrides into a typed config."""
    merged: dict[str, Any] = {}
    files = ([CONFIG_DIR / "base.yaml"] if include_base else []) + [_find(p) for p in paths]
    for p in files:
        merged = deep_merge(merged, load_yaml(p))
    for dotted, value in (overrides or {}).items():
        set_by_path(merged, dotted, value)
    return _build(ExperimentConfig, merged)


def config_to_dict(cfg: Any) -> Any:
    """Convert a (nested) dataclass config to plain YAML-serialisable data."""
    if is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, tuple):
        return [config_to_dict(v) for v in cfg]
    return cfg
