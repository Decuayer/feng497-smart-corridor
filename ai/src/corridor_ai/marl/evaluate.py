"""Compare controllers on identical scenarios and seeds.

    python -m corridor_ai.marl.evaluate --fixture toy_corridor \
        --controllers fixed_time webster --policy ippo_s0=<ckpt> ippo_s1=<ckpt> dqn_s0=<ckpt> \
        --group ippo=ippo_s0,ippo_s1,ippo_s2 \
        --seeds 1000 1001 1002 --scales 1.0

Controllers:
* ``fixed_time``  original phase order, greens clamped to [10, 60] s (control/fixed_time.py)
* ``webster``     Webster cycle/splits from flows measured on seed + offset (control/webster.py)
* ``--policy LABEL=PATH``  trained policies (IPPO or DQN, read from the checkpoint),
                  greedy (argmax over legal actions); ``--checkpoint`` = label ``ppo``
* ``--group LABEL=a,b,c``   aggregate row: per evaluation seed, the mean over the
                  member policies (e.g. the three IPPO training seeds)
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


def checkpoint_algo(checkpoint: Path) -> str:
    import torch

    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("algo", "ppo")


def run_policy(
    fixture: Fixture, cfg: ExperimentConfig, checkpoint: Path, seed: int, scale: float, out: Path
) -> dict[str, float]:
    """Greedy evaluation of a trained IPPO or DQN checkpoint."""
    from corridor_ai.marl.agents.dqn import DQNAgent
    from corridor_ai.marl.agents.ppo import PPOAgent
    from corridor_ai.marl.env import CorridorParallelEnv
    from corridor_ai.marl.train import run_dqn_episode, run_episode

    env = CorridorParallelEnv(fixture, cfg, output_dir=out)
    try:
        if checkpoint_algo(checkpoint) == "dqn":
            agent, _ = DQNAgent.load(checkpoint, cfg.dqn)
            dim = agent.q.obs_dim
        else:
            agent, _ = PPOAgent.load(checkpoint, cfg.ppo)
            dim = agent.net.obs_dim
        if dim != env.obs_dim:
            raise ValueError(f"checkpoint obs_dim {dim} != env obs_dim {env.obs_dim}")
        if isinstance(agent, DQNAgent):
            _, metrics, _, _ = run_dqn_episode(env, agent, seed, scale)
        else:
            _, _, metrics = run_episode(env, agent, seed, scale, deterministic=True, collect=False)
    finally:
        env.close()
    return metrics


run_ppo = run_policy  # backwards compatibility


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return float(sstats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / math.sqrt(n))


def significance_lines(rows: list[dict]) -> list[str]:
    """Paired Wilcoxon signed-rank tests (per evaluation seed) vs fixed-time and Webster."""
    out = ["## Paired comparison vs baselines (Wilcoxon signed-rank over seeds)", "",
           "| Scale | Controller | vs | Δ waiting | p (waiting) | Δ travel | p (travel) |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for scale in sorted({r["scale"] for r in rows}):
        sub = [r for r in rows if r["scale"] == scale]
        names = list(dict.fromkeys(r["controller"] for r in sub))
        for base in ("fixed_time", "webster"):
            if base not in names:
                continue
            b = {r["seed"]: r for r in sub if r["controller"] == base}
            for c in names:
                if c in ("fixed_time", "webster", "actuated"):
                    continue
                x = {r["seed"]: r for r in sub if r["controller"] == c}
                seeds = sorted(set(b) & set(x))
                cells = []
                for m in ("mean_waiting_time_s", "mean_travel_time_s"):
                    bv = np.array([b[s][m] for s in seeds])
                    xv = np.array([x[s][m] for s in seeds])
                    d = 100 * (xv.mean() - bv.mean()) / bv.mean()
                    try:
                        pv = sstats.wilcoxon(xv, bv).pvalue if len(seeds) > 1 else float("nan")
                    except ValueError:
                        pv = float("nan")
                    cells += [f"{d:+.1f} %", f"{pv:.4f}"]
                out.append(f"| {scale:g} | {c} | {base} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def add_group_rows(rows: list[dict], groups: dict[str, list[str]]) -> list[dict]:
    """Per (scale, seed): mean of the member controllers' metrics."""
    extra = []
    for label, members in groups.items():
        for scale in sorted({r["scale"] for r in rows}):
            for seed in sorted({r["seed"] for r in rows}):
                mem = [r for r in rows if r["scale"] == scale and r["seed"] == seed
                       and r["controller"] in members]  # fmt: skip
                if len(mem) != len(members):
                    continue
                row = {"fixture": mem[0]["fixture"], "controller": label, "scale": scale,
                       "seed": seed}  # fmt: skip
                row |= {k: float(np.mean([m[k] for m in mem])) for k in METRICS}
                extra.append(row)
    return rows + extra


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
    lines += significance_lines(rows)
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
    checkpoint: Path | None = None,
    out_dir: Path | None = None,
    policies: dict[str, Path] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> Path:
    fixture = load_fixture(cfg.fixture)
    out_dir = out_dir or RUNS_DIR / "eval" / fixture.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    plan_log: dict = {}
    policies = dict(policies or {})
    if checkpoint is not None:
        policies.setdefault("ppo", checkpoint)
    controllers = [c for c in controllers if c != "ppo" or "ppo" not in policies]
    names = controllers + [p for p in policies if p not in controllers]
    for scale in scales:
        for seed in seeds:
            for c in names:
                sumo_out = out_dir / "sumo" / c / f"scale{scale:g}"
                if c in policies:
                    m = run_policy(fixture, cfg, policies[c], seed, scale, sumo_out)
                elif c == "ppo":
                    raise ValueError("--checkpoint (or --policy ppo=...) is required for 'ppo'")
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
    rows = add_group_rows(rows, groups or {})
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {fixture.disclaimer}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "plans.json").write_text(json.dumps(plan_log, indent=1), "utf-8")
    (out_dir / "eval_config.yaml").write_text(
        yaml.safe_dump({"controllers": controllers, "seeds": seeds, "scales": scales,
                        "policies": {k: str(v) for k, v in policies.items()},
                        "groups": groups or {},
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
    p.add_argument("--checkpoint", type=Path, help="IPPO checkpoint evaluated as label 'ppo'")
    p.add_argument("--policy", nargs="*", default=[], metavar="LABEL=PATH")
    p.add_argument("--group", nargs="*", default=[], metavar="LABEL=a,b,c")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1010)))
    p.add_argument("--scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--horizon", type=int, help="override sim.horizon_s")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    overrides = {"fixture": args.fixture}
    if args.horizon:
        overrides["sim.horizon_s"] = args.horizon
    cfg = load_experiment_config(args.config, overrides=overrides)
    policies = {k: Path(v) for k, _, v in (x.partition("=") for x in args.policy)}
    groups = {k: v.split(",") for k, _, v in (x.partition("=") for x in args.group)}
    evaluate(cfg, args.controllers, args.seeds, args.scales, args.checkpoint, args.out,
             policies, groups)  # fmt: skip


if __name__ == "__main__":
    main()
