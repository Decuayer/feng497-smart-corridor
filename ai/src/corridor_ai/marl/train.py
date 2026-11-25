"""Train IPPO-PS or Double DQN on a fixture and log to TensorBoard.

Examples (one line per experiment, see ai/README.md):

    python -m corridor_ai.marl.train --config ai/configs/marl_ppo.yaml \
        --fixture toy_corridor --seed 0 --episodes 20
    python -m corridor_ai.marl.train --config ai/configs/marl_dqn.yaml \
        --fixture toy_corridor --seed 0

Outputs:
    ai/runs/<name>/<fixture>/<run>/         TensorBoard, config.yaml, episodes.csv
    ai/checkpoints/<name>/<fixture>/<run>/  model_epXXXX.pt, model_final.pt

Episode e of training seed k uses SUMO/route seed 10000*(k+1) + e. This never
overlaps the evaluation seeds (1000-1009), the debugging seeds (2000-2019) or
the Webster calibration seeds.

Transitions taken while an agent's light is under emergency preemption
(``infos[agent]["preempted"]``) are excluded from the PPO loss and are not
stored by DQN. They only occur when training with ``emergency.enabled``.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from corridor_ai.common.config import ExperimentConfig, config_to_dict, load_experiment_config
from corridor_ai.common.paths import CHECKPOINT_DIR, RUNS_DIR
from corridor_ai.common.registry import load_fixture
from corridor_ai.common.seeding import set_global_seed
from corridor_ai.marl.agents.dqn import DQNAgent, ReplayBuffer
from corridor_ai.marl.agents.ppo import PPOAgent, Trajectory
from corridor_ai.marl.env import CorridorParallelEnv

SIM_TAGS = ("mean_travel_time_s", "mean_waiting_time_s", "mean_stops", "co2_g_per_km",
            "arrived", "unfinished", "teleports")  # fmt: skip


def train_episode_seed(train_seed: int, episode: int) -> int:
    return 10000 * (train_seed + 1) + episode


# ----------------------------------------------------------------- episodes
def run_episode(
    env: CorridorParallelEnv,
    agent: PPOAgent,
    seed: int,
    scale: float | None,
    deterministic: bool = False,
    collect: bool = True,
) -> tuple[dict[str, Trajectory], dict[str, float], dict[str, float]]:
    """One PPO episode. Returns trajectories, per-agent returns and SUMO metrics."""
    obs, infos = env.reset(seed=seed, options={"scale": scale} if scale is not None else None)
    names = list(env.agents)
    trajs = {a: Trajectory() for a in names}
    returns = {a: 0.0 for a in names}
    while env.agents:
        o = np.stack([obs[a] for a in names])
        m = np.stack([infos[a]["action_mask"] for a in names])
        preempted = [bool(infos[a].get("preempted", False)) for a in names]
        acts, logps, vals = agent.act(o, m, deterministic=deterministic)
        obs, rewards, _, _, infos = env.step({a: int(acts[i]) for i, a in enumerate(names)})
        for i, a in enumerate(names):
            returns[a] += rewards[a]
            if collect:
                tr = trajs[a]
                tr.obs.append(o[i])
                tr.masks.append(m[i])
                tr.actions.append(int(acts[i]))
                tr.logps.append(float(logps[i]))
                tr.values.append(float(vals[i]))
                tr.rewards.append(float(rewards[a]))
                tr.include.append(not preempted[i])
    if collect:  # truncation: bootstrap from the value of the final observation
        last = agent.value(np.stack([obs[a] for a in names]))
        for i, a in enumerate(names):
            trajs[a].last_value = float(last[i])
    return trajs, returns, dict(env.last_episode_metrics)


def run_dqn_episode(
    env: CorridorParallelEnv,
    agent: DQNAgent,
    seed: int,
    scale: float | None,
    buffer: ReplayBuffer | None = None,
    step0: int = 0,
    total_steps: int = 1,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
    """One DQN episode. With a buffer it explores and trains; without, it acts greedily."""
    obs, infos = env.reset(seed=seed, options={"scale": scale} if scale is not None else None)
    names = list(env.agents)
    returns = {a: 0.0 for a in names}
    cfg = agent.cfg
    step, losses, qs, eps = step0, [], [], 0.0
    while env.agents:
        o = np.stack([obs[a] for a in names])
        m = np.stack([infos[a]["action_mask"] for a in names])
        preempted = [bool(infos[a].get("preempted", False)) for a in names]
        eps = agent.epsilon(step, total_steps) if buffer is not None else 0.0
        acts = agent.act(o, m, eps)
        obs, rewards, _, _, infos = env.step({a: int(acts[i]) for i, a in enumerate(names)})
        for i, a in enumerate(names):
            returns[a] += rewards[a]
            if buffer is not None and not preempted[i]:
                next_mask = infos[a]["action_mask"] if a in infos else m[i]
                buffer.add(o[i], m[i], acts[i], rewards[a], obs[a], next_mask)
        step += 1
        if (
            buffer is not None
            and buffer.size >= cfg.learning_starts
            and step % cfg.train_every == 0
        ):
            for _ in range(cfg.gradient_steps):
                s = agent.update(buffer.sample(cfg.batch_size))
                losses.append(s["td_loss"])
                qs.append(s["q_mean"])
    stats = {
        "td_loss": float(np.mean(losses)) if losses else float("nan"),
        "q_mean": float(np.mean(qs)) if qs else float("nan"),
        "epsilon": float(eps),
        "updates": float(len(losses)),
    }
    return returns, dict(env.last_episode_metrics), stats, step


# ------------------------------------------------------------------ logging
class EpisodeLogger:
    def __init__(self, run_dir: Path, disclaimer: str) -> None:
        self.writer = SummaryWriter(log_dir=str(run_dir))
        self.writer.add_text("fixture", disclaimer)
        self.csv_path = run_dir / "episodes.csv"
        self.fields: list[str] | None = None

    def log(self, ep, scale, returns, metrics, stats, groups, wall) -> float:
        w = self.writer
        mean_ret = float(np.mean(list(returns.values())))
        w.add_scalar("train/episode_return_mean", mean_ret, ep)
        for a, r in returns.items():
            w.add_scalar(f"train/episode_return/{a}", r, ep)
        w.add_scalar("train/demand_scale", scale, ep)
        for group, keys in groups.items():
            for k in keys:
                if k in stats and np.isfinite(stats[k]):
                    w.add_scalar(f"{group}/{k}", stats[k], ep)
        for k in SIM_TAGS:
            if k in metrics:
                w.add_scalar(f"sim/{k}", metrics[k], ep)
        w.add_scalar("safety/violations", metrics.get("safety_violations", 0.0), ep)
        w.add_scalar("safety/drift_mismatches", metrics.get("drift_mismatches", 0.0), ep)
        w.add_scalar("safety/illegal_actions", metrics.get("illegal_actions", 0.0), ep)
        w.add_scalar("time/episode_wall_s", wall, ep)
        row = {"episode": ep, "scale": round(scale, 3), "return_mean": round(mean_ret, 4),
               **{k: round(v, 5) for k, v in stats.items()},
               **{f"sim_{k}": round(v, 4) for k, v in metrics.items()},
               "wall_s": round(wall, 2)}  # fmt: skip
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            dw = csv.DictWriter(fh, fieldnames=self.fields or list(row))
            if self.fields is None:
                self.fields = list(row)
                dw.writeheader()
            dw.writerow(row)
        return mean_ret

    def log_eval(self, ep, returns, metrics) -> None:
        self.writer.add_scalar(
            "eval/episode_return_mean", float(np.mean(list(returns.values()))), ep
        )
        for k in ("mean_travel_time_s", "mean_waiting_time_s", "co2_g_per_km"):
            if k in metrics:
                self.writer.add_scalar(f"eval/{k}", metrics[k], ep)

    def close(self) -> None:
        self.writer.close()


def _setup(cfg: ExperimentConfig, run_name: str | None):
    set_global_seed(cfg.seed)
    fixture = load_fixture(cfg.fixture)
    run_id = run_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{cfg.seed}"
    run_dir = RUNS_DIR / cfg.name / fixture.name / run_id
    ckpt_dir = CHECKPOINT_DIR / cfg.name / fixture.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshot = config_to_dict(cfg) | {"fixture_disclaimer": fixture.disclaimer}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), "utf-8")
    env = CorridorParallelEnv(fixture, cfg, output_dir=run_dir / "sumo")
    extra = {"fixture": fixture.name, "agents": list(env.possible_agents), "k_max": env.k_max}
    return fixture, run_dir, ckpt_dir, env, extra


def _finish(agent, ckpt_dir: Path, run_dir: Path, extra: dict, episodes: int) -> dict[str, Path]:
    final = ckpt_dir / "model_final.pt"
    agent.save(final, extra | {"episode": episodes})
    (run_dir / "checkpoint.txt").write_text(str(final) + "\n", "utf-8")
    print(f"TensorBoard: tensorboard --logdir {run_dir}")
    print(f"Checkpoint:  {final}")
    return {"run_dir": run_dir, "checkpoint": final}


# ------------------------------------------------------------------- PPO
def train_ppo(cfg: ExperimentConfig, run_name: str | None = None) -> dict[str, Path]:
    fixture, run_dir, ckpt_dir, env, extra = _setup(cfg, run_name)
    pc = cfg.ppo
    agent = PPOAgent(env.obs_dim, env.k_max + 1, pc)
    logger = EpisodeLogger(run_dir, fixture.disclaimer)
    rng = np.random.default_rng(cfg.seed)
    groups = {"loss": ("policy_loss", "value_loss", "entropy"),
              "ppo": ("approx_kl", "clip_frac", "explained_variance")}  # fmt: skip
    try:
        for ep in range(1, pc.episodes + 1):
            t0 = time.time()
            scale = float(rng.uniform(pc.train_scale_min, pc.train_scale_max))
            trajs, returns, metrics = run_episode(
                env, agent, train_episode_seed(cfg.seed, ep), scale
            )
            stats = agent.update(list(trajs.values()))
            wall = time.time() - t0
            mean_ret = logger.log(ep, scale, returns, metrics, stats, groups, wall)
            print(f"[ppo ep {ep:4d}/{pc.episodes}] scale={scale:.2f} return={mean_ret:8.3f} "
                  f"wait={metrics.get('mean_waiting_time_s', float('nan')):6.1f}s "
                  f"viol={int(metrics.get('safety_violations', 0))} "
                  f"kl={stats['approx_kl']:.4f} ({wall:.1f}s)", flush=True)  # fmt: skip
            if pc.eval_every and ep % pc.eval_every == 0:
                _, eret, em = run_episode(env, agent, pc.eval_seed, 1.0, True, collect=False)
                logger.log_eval(ep, eret, em)
            if pc.checkpoint_every and ep % pc.checkpoint_every == 0:
                agent.save(ckpt_dir / f"model_ep{ep:04d}.pt", extra | {"episode": ep})
    finally:
        env.close()
        logger.close()
    return _finish(agent, ckpt_dir, run_dir, extra, pc.episodes)


# ------------------------------------------------------------------- DQN
def train_dqn(cfg: ExperimentConfig, run_name: str | None = None) -> dict[str, Path]:
    fixture, run_dir, ckpt_dir, env, extra = _setup(cfg, run_name)
    dc = cfg.dqn
    agent = DQNAgent(env.obs_dim, env.k_max + 1, dc, seed=cfg.seed)
    buffer = ReplayBuffer(dc.buffer_size, env.obs_dim, env.k_max + 1, seed=cfg.seed)
    logger = EpisodeLogger(run_dir, fixture.disclaimer)
    rng = np.random.default_rng(cfg.seed)
    steps_per_ep = cfg.sim.horizon_s // cfg.env.decision_interval_s
    total = dc.episodes * steps_per_ep
    groups = {"loss": ("td_loss",), "dqn": ("q_mean", "epsilon", "updates")}
    step = 0
    try:
        for ep in range(1, dc.episodes + 1):
            t0 = time.time()
            scale = float(rng.uniform(dc.train_scale_min, dc.train_scale_max))
            returns, metrics, stats, step = run_dqn_episode(
                env, agent, train_episode_seed(cfg.seed, ep), scale, buffer, step, total
            )
            stats["buffer"] = float(buffer.size)
            wall = time.time() - t0
            mean_ret = logger.log(ep, scale, returns, metrics, stats, groups, wall)
            print(f"[dqn ep {ep:4d}/{dc.episodes}] scale={scale:.2f} return={mean_ret:8.3f} "
                  f"wait={metrics.get('mean_waiting_time_s', float('nan')):6.1f}s "
                  f"viol={int(metrics.get('safety_violations', 0))} eps={stats['epsilon']:.2f} "
                  f"td={stats['td_loss']:.4f} ({wall:.1f}s)", flush=True)  # fmt: skip
            if dc.eval_every and ep % dc.eval_every == 0:
                eret, em, _, _ = run_dqn_episode(env, agent, dc.eval_seed, 1.0)
                logger.log_eval(ep, eret, em)
            if dc.checkpoint_every and ep % dc.checkpoint_every == 0:
                agent.save(ckpt_dir / f"model_ep{ep:04d}.pt", extra | {"episode": ep})
    finally:
        env.close()
        logger.close()
    return _finish(agent, ckpt_dir, run_dir, extra, dc.episodes)


def train(cfg: ExperimentConfig, run_name: str | None = None) -> dict[str, Path]:
    if cfg.algo == "ppo":
        return train_ppo(cfg, run_name)
    if cfg.algo == "dqn":
        return train_dqn(cfg, run_name)
    raise ValueError(f"unknown algo '{cfg.algo}'")


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="ai/configs/marl_ppo.yaml")
    p.add_argument("--fixture")
    p.add_argument("--seed", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--run-name")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted overrides, e.g. env.reward.beta=0.2")  # fmt: skip
    args = p.parse_args(argv)
    overrides = _parse_overrides(args.set)
    if args.fixture:
        overrides["fixture"] = args.fixture
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_experiment_config(args.config, overrides=overrides)
    if args.episodes is not None:
        key = "ppo.episodes" if cfg.algo == "ppo" else "dqn.episodes"
        cfg = load_experiment_config(args.config, overrides=overrides | {key: args.episodes})
    train(cfg, args.run_name)


if __name__ == "__main__":
    main()
