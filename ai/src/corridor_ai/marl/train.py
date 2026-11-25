"""Train parameter-shared IPPO on a fixture and log to TensorBoard.

Example (one line per experiment, see ai/README.md):

    python -m corridor_ai.marl.train --config ai/configs/marl_ppo.yaml \
        --fixture toy_corridor --seed 0 --episodes 20

Outputs:
    ai/runs/<name>/<fixture>/<timestamp>_seed<k>/         TensorBoard, config.yaml, episodes.csv
    ai/checkpoints/<name>/<fixture>/<timestamp>_seed<k>/  model_epXXXX.pt, model_final.pt

Episode e of training seed k uses SUMO/route seed 10000*(k+1) + e. This never
overlaps the evaluation seeds (1000-1009) or the Webster calibration seeds.
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
from corridor_ai.marl.agents.ppo import PPOAgent, Trajectory
from corridor_ai.marl.env import CorridorParallelEnv


def train_episode_seed(train_seed: int, episode: int) -> int:
    return 10000 * (train_seed + 1) + episode


def run_episode(
    env: CorridorParallelEnv,
    agent: PPOAgent,
    seed: int,
    scale: float | None,
    deterministic: bool = False,
    collect: bool = True,
) -> tuple[dict[str, Trajectory], dict[str, float], dict[str, float]]:
    obs, infos = env.reset(seed=seed, options={"scale": scale} if scale is not None else None)
    names = list(env.agents)
    trajs = {a: Trajectory() for a in names}
    returns = {a: 0.0 for a in names}
    while env.agents:
        o = np.stack([obs[a] for a in names])
        m = np.stack([infos[a]["action_mask"] for a in names])
        acts, logps, vals = agent.act(o, m, deterministic=deterministic)
        obs, rewards, _, truncs, infos = env.step({a: int(acts[i]) for i, a in enumerate(names)})
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
    if collect:  # truncation: bootstrap from the value of the final observation
        last = agent.value(np.stack([obs[a] for a in names]))
        for i, a in enumerate(names):
            trajs[a].last_value = float(last[i])
    return trajs, returns, dict(env.last_episode_metrics)


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def train(cfg: ExperimentConfig, run_name: str | None = None) -> dict[str, Path]:
    set_global_seed(cfg.seed)
    fixture = load_fixture(cfg.fixture)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = run_name or f"{stamp}_seed{cfg.seed}"
    run_dir = RUNS_DIR / cfg.name / fixture.name / run_id
    ckpt_dir = CHECKPOINT_DIR / cfg.name / fixture.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshot = config_to_dict(cfg) | {"fixture_disclaimer": fixture.disclaimer}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), "utf-8")

    env = CorridorParallelEnv(fixture, cfg, output_dir=run_dir / "sumo")
    agent = PPOAgent(env.obs_dim, env.k_max + 1, cfg.ppo)
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text("fixture", fixture.disclaimer)
    rng = np.random.default_rng(cfg.seed)
    csv_path = run_dir / "episodes.csv"
    fields: list[str] | None = None
    extra = {"fixture": fixture.name, "agents": list(env.possible_agents), "k_max": env.k_max}

    try:
        for ep in range(1, cfg.ppo.episodes + 1):
            t0 = time.time()
            scale = float(rng.uniform(cfg.ppo.train_scale_min, cfg.ppo.train_scale_max))
            trajs, returns, metrics = run_episode(
                env, agent, train_episode_seed(cfg.seed, ep), scale
            )
            stats = agent.update(list(trajs.values()))
            wall = time.time() - t0

            mean_ret = float(np.mean(list(returns.values())))
            writer.add_scalar("train/episode_return_mean", mean_ret, ep)
            for a, r in returns.items():
                writer.add_scalar(f"train/episode_return/{a}", r, ep)
            writer.add_scalar("train/demand_scale", scale, ep)
            for k in ("policy_loss", "value_loss", "entropy"):
                writer.add_scalar(f"loss/{k}", stats[k], ep)
            for k in ("approx_kl", "clip_frac", "explained_variance"):
                writer.add_scalar(f"ppo/{k}", stats[k], ep)
            for k in ("mean_travel_time_s", "mean_waiting_time_s", "mean_stops", "co2_g_per_km",
                      "arrived", "unfinished", "teleports"):  # fmt: skip
                if k in metrics:
                    writer.add_scalar(f"sim/{k}", metrics[k], ep)
            writer.add_scalar("safety/violations", metrics.get("safety_violations", 0.0), ep)
            writer.add_scalar("safety/drift_mismatches", metrics.get("drift_mismatches", 0.0), ep)
            writer.add_scalar("safety/illegal_actions", metrics.get("illegal_actions", 0.0), ep)
            writer.add_scalar("time/episode_wall_s", wall, ep)

            row = {"episode": ep, "scale": round(scale, 3), "return_mean": round(mean_ret, 4),
                   **{k: round(v, 5) for k, v in stats.items()},
                   **{f"sim_{k}": round(v, 4) for k, v in metrics.items()},
                   "wall_s": round(wall, 2)}  # fmt: skip
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields or list(row))
                if fields is None:
                    fields = list(row)
                    w.writeheader()
                w.writerow(row)
            print(
                f"[ep {ep:4d}/{cfg.ppo.episodes}] scale={scale:.2f} return={mean_ret:8.3f} "
                f"wait={metrics.get('mean_waiting_time_s', float('nan')):6.1f}s "
                f"viol={int(metrics.get('safety_violations', 0))} "
                f"kl={stats['approx_kl']:.4f} ({wall:.1f}s)",
                flush=True,
            )

            if cfg.ppo.eval_every and ep % cfg.ppo.eval_every == 0:
                _, eret, em = run_episode(env, agent, cfg.ppo.eval_seed, 1.0, True, collect=False)
                writer.add_scalar(
                    "eval/episode_return_mean", float(np.mean(list(eret.values()))), ep
                )
                for k in ("mean_travel_time_s", "mean_waiting_time_s", "co2_g_per_km"):
                    if k in em:
                        writer.add_scalar(f"eval/{k}", em[k], ep)
            if cfg.ppo.checkpoint_every and ep % cfg.ppo.checkpoint_every == 0:
                agent.save(ckpt_dir / f"model_ep{ep:04d}.pt", extra | {"episode": ep})
    finally:
        env.close()
        writer.close()
    final = ckpt_dir / "model_final.pt"
    agent.save(final, extra | {"episode": cfg.ppo.episodes})
    (run_dir / "checkpoint.txt").write_text(str(final) + "\n", "utf-8")
    print(f"TensorBoard: tensorboard --logdir {run_dir}")
    print(f"Checkpoint:  {final}")
    return {"run_dir": run_dir, "checkpoint": final}


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
    if args.episodes is not None:
        overrides["ppo.episodes"] = args.episodes
    cfg = load_experiment_config(args.config, overrides=overrides)
    train(cfg, args.run_name)


if __name__ == "__main__":
    main()
