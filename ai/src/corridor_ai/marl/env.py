"""Multi-agent signal-control environment (PettingZoo ParallelEnv API).

One agent per controlled intersection (contract ids from
shared/contracts/intersections.yaml, mapped to SUMO lights by
ai/configs/fixtures.yaml). Every other traffic light in the network runs its
safety-compliant fixed-time plan.

Episode:
* reset: start SUMO (seeded), then run ``warmup_s`` under fixed-time for all lights;
* step: apply one action per agent, then simulate ``decision_interval_s`` seconds
  (default 5). All lights are advanced each second by their PhaseMachine;
* the episode is truncated (not terminated) after ``horizon_s`` controlled seconds.

``infos[agent]["action_mask"]`` holds the legal actions. An illegal action is
replaced by the first legal one and counted in ``infos[agent]["illegal_actions"]``.
At the end of an episode, ``infos[agent]["episode"]`` holds the SUMO metrics
(tripinfo, safety violations, drift).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from corridor_ai.common.config import ExperimentConfig
from corridor_ai.common.registry import Fixture
from corridor_ai.control.command_sink import SignalCommandRecorder
from corridor_ai.control.fixed_time import fixed_time_controller
from corridor_ai.marl.actions import action_mask, action_to_request
from corridor_ai.marl.observations import build_observation, observation_size
from corridor_ai.marl.reward import RewardAccumulator
from corridor_ai.sim.core import CorridorSim
from corridor_ai.sim.tls_model import load_tls_models


class CorridorParallelEnv(ParallelEnv):
    metadata = {"name": "corridor_signal_control_v0", "render_modes": []}

    def __init__(
        self,
        fixture: Fixture,
        cfg: ExperimentConfig,
        output_dir: Path | None = None,
        scale: float | None = None,
        command_log: Path | None = None,
        issuer: str = "marl_ppo",
    ) -> None:
        self.fixture = fixture
        self.command_log = command_log
        self.issuer = issuer
        self.recorder: SignalCommandRecorder | None = None
        self.cfg = cfg
        self.output_dir = output_dir
        self.scale = scale
        self.models = load_tls_models(fixture.net)
        self.agent_tls: dict[str, str] = dict(fixture.agents)
        missing = [t for t in self.agent_tls.values() if t not in self.models]
        if missing:
            raise ValueError(f"fixture {fixture.name}: traffic lights not in network: {missing}")
        self.possible_agents = list(self.agent_tls)
        self.agents: list[str] = []
        k_auto = max(self.models[t].n_phases for t in self.agent_tls.values())
        self.k_max = cfg.env.k_max or k_auto
        if self.k_max < k_auto:
            raise ValueError(f"k_max={self.k_max} < phases of an agent ({k_auto})")
        self.obs_dim = observation_size(self.k_max, len(self.possible_agents))
        self._obs_space = spaces.Box(0.0, 1.0, (self.obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(self.k_max + 1)
        self.sim: CorridorSim | None = None
        self._rewards: dict[str, RewardAccumulator] = {}
        self._illegal: dict[str, int] = {}
        self._background = None
        self._episode_seed = cfg.seed
        self.last_episode_metrics: dict[str, float] = {}

    # ----------------------------------------------------------- spaces
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    # ------------------------------------------------------------- core
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        if self.sim is not None and self.sim.started:
            self.sim.close()
        self._episode_seed = self.cfg.seed if seed is None else int(seed)
        scale = options.get("scale", self.scale)
        self.sim = CorridorSim(
            self.fixture, self.cfg, seed=self._episode_seed, output_dir=self.output_dir, scale=scale
        )
        self.sim.start()
        if self.command_log is not None:
            self.recorder = SignalCommandRecorder(
                self.agent_tls, self.issuer, "adaptive",
                valid_for_ms=self.cfg.env.decision_interval_s * 1000, path=self.command_log,
            )  # fmt: skip
        agent_tls = set(self.agent_tls.values())
        background = [t for t in self.models if t not in agent_tls]
        self._background = fixed_time_controller(self.models, self.cfg.limits, background)
        warmup_all = fixed_time_controller(self.models, self.cfg.limits)
        self.sim.run_controllers([warmup_all], self.cfg.sim.warmup_s)
        if self.recorder is not None:  # only commands issued by the policy are recorded
            self.recorder.sync(self.sim)
            self.sim.second_hooks.append(self.recorder)

        self.agents = list(self.possible_agents)
        self._illegal = {a: 0 for a in self.agents}
        self._rewards = {
            a: RewardAccumulator(self.models[t].incoming_lanes, self.cfg.env.reward)
            for a, t in self.agent_tls.items()
        }
        obs = self._observations()
        infos = {a: {"action_mask": self._mask(a)} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        assert self.sim is not None and self.agents, "call reset() first"
        requests: dict[str, int | None] = {}
        for agent in self.agents:
            tls = self.agent_tls[agent]
            mask = self._mask(agent)
            action = int(actions.get(agent, 0))
            if not (0 <= action < len(mask)) or not mask[action]:
                self._illegal[agent] += 1
                action = int(np.flatnonzero(mask)[0])
            requests[tls] = action_to_request(action, self.sim.machines[tls])

        for second in range(self.cfg.env.decision_interval_s):
            bg = self._background.act(self.sim) if self._background else {}
            self.sim.step_second({**bg, **(requests if second == 0 else {})})
            for acc in self._rewards.values():
                acc.observe_second(self.sim.api)
            if self.sim.done:
                break

        rewards, infos = {}, {}
        for agent in self.agents:
            r, rinfo = self._rewards[agent].pop_reward()
            rewards[agent] = r
            infos[agent] = {
                "action_mask": self._mask(agent),
                "illegal_actions": self._illegal[agent],
                **rinfo,
            }
        obs = self._observations()
        elapsed = self.sim.time_s - self.cfg.sim.warmup_s
        truncated = self.sim.done or elapsed >= self.cfg.sim.horizon_s
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        if truncated:
            self.last_episode_metrics = self._finish()
            for agent in self.agents:
                infos[agent]["episode"] = self.last_episode_metrics
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def action_masks(self) -> dict[str, np.ndarray]:
        return {a: self._mask(a) for a in self.agents}

    def close(self) -> None:
        if self.sim is not None and self.sim.started:
            self.sim.close()

    # --------------------------------------------------------- internals
    def _mask(self, agent: str) -> np.ndarray:
        assert self.sim is not None
        machine = self.sim.machines[self.agent_tls[agent]]
        return action_mask(machine, self.k_max, self.cfg.limits, self.cfg.env)

    def _observations(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        out = {}
        n = len(self.possible_agents)
        for idx, agent in enumerate(self.possible_agents):
            tls = self.agent_tls[agent]
            model = self.models[tls]
            stats = {
                ln: (
                    self.sim.lane_halting(ln),
                    self.sim.lane_waiting(ln),
                    self.sim.lane_occupancy(ln),
                )
                for ln in model.incoming_lanes
            }
            out[agent] = build_observation(
                model, self.sim.machines[tls], stats, idx, n, self.k_max,
                self.cfg.limits, self.cfg.env,
            )  # fmt: skip
        return out

    def _finish(self) -> dict[str, float]:
        assert self.sim is not None
        metrics = self.sim.close(min_depart_s=self.cfg.sim.warmup_s)
        metrics["illegal_actions"] = float(sum(self._illegal.values()))
        metrics["seed"] = float(self._episode_seed)
        return metrics
