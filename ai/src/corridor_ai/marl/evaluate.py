"""Compare controllers on identical scenarios and seeds.

    python -m corridor_ai.marl.evaluate --fixture toy_corridor \
        --controllers fixed_time webster ppo --checkpoint <path/to/model_final.pt> \
        --seeds 1000 1001 1002 --scales 1.0

Controllers:
* ``fixed_time``  original phase order, greens clamped to [10, 60] s (control/fixed_time.py)
* ``webster``     Webster cycle/splits from flows measured on seed + offset (control/webster.py)
* ``ppo``         trained IPPO policy, greedy (argmax over legal actions)
* ``actuated``    REFERENCE ONLY: the network's own SUMO programs, no safety
                  enforcement (yellow 6 s etc.); safety violations are reported

Every run: same warm-up (fixed-time for warmup_s), same horizon, same SUMO
seed and demand. Metrics cover vehicles that departed after the warm-up.
Background lights (not agents) always run fixed-time.

Outputs in ai/runs/eval/<fixture>/<timestamp>/: results.csv (one row per run)
and summary.md (mean ± 95 % CI per controller and scale, relative to
fixed-time). Both carry the placeholder-fixture disclaimer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sstats

from corridor_ai.common.config import ExperimentConfig, load_experiment_config
from corridor_ai.common.paths import RUNS_DIR
from corridor_ai.common.registry import Fixture, load_fixture
from corridor_ai.control.fixed_time import fixed_time_controller, fixed_time_plan
from corridor_ai.control.webster import WebsterParams, measure_lane_flows, webster_controller
from corridor_ai.sim.core import CorridorSim

METRICS = (
    "mean_travel_time_s",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "mean_stops",
    "co2_g_per_km",
    "arrived",
    "unfinished",
    "teleports",
    "safety_violations",
    "drift_mismatches",
)
SUMMARY_METRICS = (
    "mean_travel_time_s", "mean_waiting_time_s", "mean_stops", "co2_g_per_km",
    "arrived", "safety_violations",
)  # fmt: skip


def run_plan_controller(
    fixture: Fixture,
    cfg: ExperimentConfig,
    controller: str,
    seed: int,
    scale: float,
    out: Path,
    plan_log: dict | None = None,
) -> dict[str, float]:
    """Run fixed_time / webster / actuated for one seed and scale."""
    if controller == "actuated":
        sim = CorridorSim(fixture, cfg, seed, output_dir=out, scale=scale, controlled=False)
        sim.start()
        while not sim.done:
            sim.step_second()
        return sim.close(min_depart_s=cfg.sim.warmup_s)

    agents = list(fixture.agents.values())
    details = None
    if controller == "webster":  # calibrate BEFORE starting the evaluation sim (libsumo: 1 sim)
        params = WebsterParams.from_config()
        flows = measure_lane_flows(fixture, cfg, seed + params.calibration_seed_offset, scale)
    sim = CorridorSim(fixture, cfg, seed, output_dir=out, scale=scale)
    sim.start()
    background = [t for t in sim.models if t not in set(agents)]
    sim.run_controllers([fixed_time_controller(sim.models, cfg.limits)], cfg.sim.warmup_s)
    if controller == "fixed_time":
        main = fixed_time_controller(sim.models, cfg.limits, agents)
        details = {t: fixed_time_plan(sim.models[t], cfg.limits).as_dict() for t in agents}
    elif controller == "webster":
        main, details = webster_controller(sim.models, flows, cfg.limits, params, agents)
    else:
        raise ValueError(f"unknown plan controller '{controller}'")
    if plan_log is not None and details is not None:
        plan_log[f"{controller}/scale{scale:g}/seed{seed}"] = details
    bg = fixed_time_controller(sim.models, cfg.limits, background)
    while not sim.done:
        sim.run_controllers([main, bg], 1)
    return sim.close(min_depart_s=cfg.sim.warmup_s)


def run_ppo(
    fixture: Fixture, cfg: ExperimentConfig, checkpoint: Path, seed: int, scale: float, out: Path
) -> dict[str, float]:
    from corridor_ai.marl.agents.ppo import PPOAgent
    from corridor_ai.marl.env import CorridorParallelEnv
    from corridor_ai.marl.train import run_episode

    env = CorridorParallelEnv(fixture, cfg, output_dir=out)
    agent, _ = PPOAgent.load(checkpoint, cfg.ppo)
    if agent.net.obs_dim != env.obs_dim:
        raise ValueError(f"checkpoint obs_dim {agent.net.obs_dim} != env obs_dim {env.obs_dim}")
    _, _, metrics = run_episode(env, agent, seed, scale, deterministic=True, collect=False)
    env.close()
    return metrics


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return float(sstats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / math.sqrt(n))


def write_summary(rows: list[dict], fixture: Fixture, cfg: ExperimentConfig, path: Path) -> str:
    lines = [
        f"# Controller comparison: {fixture.name}",
        "",
        f"> **{fixture.disclaimer}**",
        "",
        f"Horizon {cfg.sim.horizon_s} s after a {cfg.sim.warmup_s} s fixed-time warm-up; "
        "metrics over vehicles departing after the warm-up (unfinished vehicles included in "
        "means). Values: mean ± 95 % CI over seeds; Δ = relative change vs fixed-time "
        "(negative = better for times, stops and CO₂).",
        "",
    ]
    for scale in sorted({r["scale"] for r in rows}):
        sub = [r for r in rows if r["scale"] == scale]
        controllers = list(dict.fromkeys(r["controller"] for r in sub))
        seeds = sorted({r["seed"] for r in sub})
        lines += [f"## Demand scale {scale:g} (seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)})", ""]
        header = "| Controller | " + " | ".join(SUMMARY_METRICS) + " |"
        lines += [header, "|" + "---|" * (len(SUMMARY_METRICS) + 1)]
        base = {m: np.mean([r[m] for r in sub if r["controller"] == "fixed_time"] or [np.nan])
                for m in SUMMARY_METRICS}  # fmt: skip
        for c in controllers:
            cells = []
            for m in SUMMARY_METRICS:
                vals = [r[m] for r in sub if r["controller"] == c]
                mean = float(np.mean(vals))
                cell = f"{mean:.2f} ± {_ci95(vals):.2f}"
                if (
                    c != "fixed_time"
                    and base[m]
                    and not np.isnan(base[m])
                    and m not in ("safety_violations",)
                ):
                    cell += f" (Δ {100 * (mean - base[m]) / base[m]:+.1f} %)"
                cells.append(cell)
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        "`actuated` = the network's own SUMO programs; reference only, not safety-compliant."
    )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def evaluate(
    cfg: ExperimentConfig,
    controllers: list[str],
    seeds: list[int],
    scales: list[float],
    checkpoint: Path | None,
    out_dir: Path | None = None,
) -> Path:
    fixture = load_fixture(cfg.fixture)
    out_dir = out_dir or RUNS_DIR / "eval" / fixture.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    plan_log: dict = {}
    for scale in scales:
        for seed in seeds:
            for c in controllers:
                sumo_out = out_dir / "sumo" / c / f"scale{scale:g}"
                if c == "ppo":
                    if checkpoint is None:
                        raise ValueError("--checkpoint is required for controller 'ppo'")
                    m = run_ppo(fixture, cfg, checkpoint, seed, scale, sumo_out)
                else:
                    m = run_plan_controller(fixture, cfg, c, seed, scale, sumo_out, plan_log)
                row = {"fixture": fixture.name, "controller": c, "scale": scale, "seed": seed}
                row |= {k: float(m.get(k, float("nan"))) for k in METRICS}
                rows.append(row)
                print(
                    f"{c:<11} scale={scale:<4g} seed={seed} "
                    f"travel={row['mean_travel_time_s']:7.1f}s "
                    f"wait={row['mean_waiting_time_s']:6.1f}s "
                    f"viol={int(row['safety_violations'])} drift={row['drift_mismatches']:.0f}",
                    flush=True,
                )
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {fixture.disclaimer}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "plans.json").write_text(json.dumps(plan_log, indent=1), "utf-8")
    (out_dir / "eval_config.yaml").write_text(
        yaml.safe_dump({"controllers": controllers, "seeds": seeds, "scales": scales,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "horizon_s": cfg.sim.horizon_s, "warmup_s": cfg.sim.warmup_s}),
        "utf-8",
    )  # fmt: skip
    print(write_summary(rows, fixture, cfg, out_dir / "summary.md"))
    print(f"results: {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="ai/configs/marl_ppo.yaml")
    p.add_argument("--fixture", default="toy_corridor")
    p.add_argument("--controllers", nargs="+", default=["fixed_time", "webster"])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1010)))
    p.add_argument("--scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--horizon", type=int, help="override sim.horizon_s")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    overrides = {"fixture": args.fixture}
    if args.horizon:
        overrides["sim.horizon_s"] = args.horizon
    cfg = load_experiment_config(args.config, overrides=overrides)
    evaluate(cfg, args.controllers, args.seeds, args.scales, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
