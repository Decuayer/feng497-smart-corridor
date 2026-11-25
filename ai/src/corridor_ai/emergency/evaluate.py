"""TASK-302 acceptance: emergency-vehicle (EV) corridor transit with vs without preemption.

    python -m corridor_ai.emergency.evaluate --config ai/configs/emergency.yaml \
        --fixture toy_corridor --controllers fixed_time ppo \
        --checkpoint <model_final.pt> --seeds 1000 1001 ... --scales 1.0 1.5

For every (controller, scale, seed), two paired runs:
* **baseline** (ASSUMPTION, approved): same scenario, same seed, same EV
  insertion times, **no preemption**;
* **preemption**: the PreemptionManager overrides the base controller.

Reported per controller and scale:
* EV corridor transit time (tripinfo duration of EVs on the KPI corridor route)
  as median and mean. Improvement = 1 - median(pre) / median(base). Acceptance
  is >= 50 %. Also a paired Wilcoxon test on the per-seed mean EV transit;
* recovery time after each release vs the matched baseline run
  (emergency/recovery.py), with the share within ``recovery_bound_s``;
* safety: monitor violations and drift in **all** runs (must be 0);
* side effects: non-EV mean travel and waiting time, with vs without preemption.

Outputs (ai/runs/emergency/<fixture>/<name>/): summary.md, ev_trips.csv,
recoveries.csv, runs.csv, and for the first seed of each controller the
schema-valid logs: system_log_events.jsonl, emergency_requests.jsonl and
signal_commands.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sstats

from corridor_ai.common.config import ExperimentConfig, load_experiment_config
from corridor_ai.common.paths import RUNS_DIR
from corridor_ai.common.registry import Fixture, load_fixture
from corridor_ai.control.command_sink import SignalCommandRecorder
from corridor_ai.control.fixed_time import fixed_time_controller
from corridor_ai.emergency.preemption import PreemptionManager
from corridor_ai.emergency.recovery import QueueRecorder, recoveries
from corridor_ai.emergency.scenario import write_ev_routes
from corridor_ai.sim.core import CorridorSim
from corridor_ai.sim.metrics import parse_tripinfo, tripinfo_records


def _collect(sim_metrics, sim, manager, recorder, cfg, controller, preempt, seed, scale):
    ev = tripinfo_records(sim.tripinfo_path, cfg.emergency.ev_vtypes) if sim.tripinfo_path else []
    non_ev = parse_tripinfo(sim.tripinfo_path, cfg.sim.warmup_s, cfg.emergency.ev_vtypes)
    times, series = recorder.arrays()
    return {
        "controller": controller, "preempt": preempt, "seed": seed, "scale": scale,
        "ev": ev, "non_ev": non_ev, "times": times, "series": series,
        "activations": list(manager.activations), "releases": list(manager.releases),
        "events": list(manager.events), "requests": list(manager.request_messages),
        "safety_violations": sim_metrics["safety_violations"],
        "violations_by_rule": sim.monitor.counts_by_rule(),
        "drift_mismatches": sim_metrics["drift_mismatches"],
        "reservices": sum(r.get("reservices", 0) for r in manager.releases),
    }  # fmt: skip


def run_fixed_time(fixture, cfg, seed, scale, preempt, out_dir, command_log=None) -> dict:
    end_s = cfg.sim.warmup_s + cfg.sim.horizon_s
    ev_file = write_ev_routes(fixture, cfg, seed, end_s)
    sim = CorridorSim(fixture, cfg, seed, output_dir=out_dir, scale=scale,
                      extra_route_files=[ev_file])  # fmt: skip
    sim.start()
    manager = PreemptionManager(fixture, cfg, enabled=preempt)
    recorder = QueueRecorder(fixture.agents, start_s=cfg.sim.warmup_s)
    sim.second_hooks += [manager.update, recorder]
    if command_log is not None:
        sim.second_hooks.append(
            SignalCommandRecorder(fixture.agents, "fixed_time", "fixed_time", path=command_log,
                                  mode_override=manager.command_mode)  # fmt: skip
        )
    base = fixed_time_controller(sim.models, cfg.limits)
    try:
        while not sim.done:
            requests = base.act(sim)
            if preempt:
                requests.update(manager.requests(sim))
            sim.step_second(requests)
    finally:
        metrics = sim.close(min_depart_s=cfg.sim.warmup_s)
    return _collect(metrics, sim, manager, recorder, cfg, "fixed_time", preempt, seed, scale)


def run_ppo(fixture, cfg, checkpoint, seed, scale, preempt, out_dir, command_log=None) -> dict:
    from corridor_ai.marl.agents.ppo import PPOAgent
    from corridor_ai.marl.env import CorridorParallelEnv

    env = CorridorParallelEnv(fixture, cfg, output_dir=out_dir, command_log=command_log,
                              ev_scenario=True, preempt=preempt)  # fmt: skip
    agent, _ = PPOAgent.load(checkpoint, cfg.ppo)
    if agent.net.obs_dim != env.obs_dim:
        raise ValueError(f"checkpoint obs_dim {agent.net.obs_dim} != env obs_dim {env.obs_dim}")
    obs, infos = env.reset(seed=seed, options={"scale": scale})
    recorder = QueueRecorder(fixture.agents, start_s=cfg.sim.warmup_s)
    env.sim.second_hooks.append(recorder)
    sim, manager = env.sim, env.manager
    names = list(env.agents)
    while env.agents:
        o = np.stack([obs[a] for a in names])
        m = np.stack([infos[a]["action_mask"] for a in names])
        acts, _, _ = agent.act(o, m, deterministic=True)
        obs, _, _, _, infos = env.step({a: int(acts[i]) for i, a in enumerate(names)})
    return _collect(env.last_episode_metrics, sim, manager, recorder, cfg, "ppo", preempt, seed,
                    scale)  # fmt: skip


# ----------------------------------------------------------------- summarise
def _pct(base: float, new: float) -> float:
    return 100.0 * (1.0 - new / base) if base else float("nan")


def summarise(runs: list[dict], fixture: Fixture, cfg: ExperimentConfig, out: Path) -> str:
    em = cfg.emergency
    ev_rows, rec_rows, run_rows = [], [], []
    lines = [
        f"# TASK-302 emergency green corridor: {fixture.name}",
        "",
        f"> **{fixture.disclaimer}**",
        "",
        f"KPI route `{fixture.kpi_route}` ({len(fixture.corridor_edges)} edges). "
        f"{em.ev_count} EVs per run (speedFactor {em.ev_speed_factor}, EVs obey signals), "
        f"horizon {cfg.sim.horizon_s} s after a {cfg.sim.warmup_s} s warm-up. "
        f"Baseline = same scenario, same seeds, same EV insertion times, no preemption "
        "(ASSUMPTION). Look-ahead: current + next "
        f"{em.lookahead} intersections. Improvement = 1 − median(pre)/median(base).",
        "",
        "| Controller | Scale | Seeds | EVs | EV transit base: median / mean (s) | "
        "EV transit preempt: median / mean (s) | Improvement (median) | Improvement (mean) | "
        "Wilcoxon p | ≥ 50 %? | Recovery median / p90 / max (s) | Recovered ≤ "
        f"{em.recovery_bound_s} s | Censored | Non-EV travel Δ | Non-EV waiting Δ | "
        "Safety violations | Drift |",
        "|" + "---|" * 17,
    ]
    keys = sorted(
        {(r["controller"], r["scale"]) for r in runs}, key=lambda k: (k[0] != "fixed_time", k)
    )
    for controller, scale in keys:
        sub = [r for r in runs if r["controller"] == controller and r["scale"] == scale]
        seeds = sorted({r["seed"] for r in sub})
        base_ev, pre_ev, per_seed = [], [], []
        recs, nonev_b, nonev_p = [], [], []
        viol = sum(r["safety_violations"] for r in sub)
        drift = sum(r["drift_mismatches"] for r in sub)
        for seed in seeds:
            b = next(r for r in sub if r["seed"] == seed and not r["preempt"])
            p = next(r for r in sub if r["seed"] == seed and r["preempt"])
            bt = [e["duration_s"] for e in b["ev"]]
            pt = [e["duration_s"] for e in p["ev"]]
            base_ev += bt
            pre_ev += pt
            if bt and pt:
                per_seed.append((np.mean(bt), np.mean(pt)))
            for which, r in (("base", b), ("preempt", p)):
                for e in r["ev"]:
                    ev_rows.append({"controller": controller, "scale": scale, "seed": seed,
                                    "run": which, **e})  # fmt: skip
            for rr in recoveries(p["times"], p["series"], b["series"], p["releases"],
                                 p["activations"], em):  # fmt: skip
                recs.append(rr)
                rec_rows.append({"controller": controller, "scale": scale, "seed": seed, **rr})
            nonev_b.append(b["non_ev"])
            nonev_p.append(p["non_ev"])
            for r in (b, p):
                run_rows.append({
                    "controller": controller, "scale": scale, "seed": seed, "preempt": r["preempt"],
                    "evs": len(r["ev"]), "evs_arrived": int(sum(e["arrived"] for e in r["ev"])),
                    "ev_transit_mean_s": (
                        round(float(np.mean([e["duration_s"] for e in r["ev"]])), 1)
                        if r["ev"] else None
                    ),
                    "preemptions": len(r["activations"]), "reservices": r["reservices"],
                    "non_ev_travel_s": round(r["non_ev"]["mean_travel_time_s"], 1),
                    "non_ev_waiting_s": round(r["non_ev"]["mean_waiting_time_s"], 1),
                    "safety_violations": r["safety_violations"], "drift": r["drift_mismatches"],
                })  # fmt: skip
        med_b, med_p = float(np.median(base_ev)), float(np.median(pre_ev))
        mean_b, mean_p = float(np.mean(base_ev)), float(np.mean(pre_ev))
        imp_med, imp_mean = _pct(med_b, med_p), _pct(mean_b, mean_p)
        try:
            pval = sstats.wilcoxon([a for a, _ in per_seed], [b for _, b in per_seed]).pvalue
        except ValueError:
            pval = float("nan")
        done = [r["recovery_s"] for r in recs if r["recovery_s"] is not None]
        within = sum(1 for x in done if x <= em.recovery_bound_s)
        rec_txt = (f"{np.median(done):.0f} / {np.percentile(done, 90):.0f} / {max(done):.0f}"
                   if done else "n/a")  # fmt: skip
        nb = np.mean([x["mean_travel_time_s"] for x in nonev_b])
        np_ = np.mean([x["mean_travel_time_s"] for x in nonev_p])
        wb = np.mean([x["mean_waiting_time_s"] for x in nonev_b])
        wp = np.mean([x["mean_waiting_time_s"] for x in nonev_p])
        lines.append(
            f"| {controller} | {scale:g} | {len(seeds)} | {len(base_ev)} / {len(pre_ev)} | "
            f"{med_b:.1f} / {mean_b:.1f} | {med_p:.1f} / {mean_p:.1f} | **{imp_med:.1f} %** | "
            f"{imp_mean:.1f} % | {pval:.4f} | {'**yes**' if imp_med >= 50 else '**no**'} | "
            f"{rec_txt} | {within}/{len(recs)} ({100 * within / max(len(recs), 1):.0f} %) | "
            f"{len(recs) - len(done)} | {100 * (np_ - nb) / nb:+.1f} % | "
            f"{100 * (wp - wb) / wb:+.1f} % | {int(viol)} | {int(drift)} |"
        )
    lines += [
        "",
        "Recovery = time after a release until the intersection's queue (30 s moving average) "
        f"is at most max({100 * em.recovery_tolerance_frac:.0f} %, "
        f"{em.recovery_tolerance_abs:g} veh) above the matched no-preemption run for "
        f"{em.recovery_sustain_s} s. Censored = the next preemption at that intersection or the "
        "end of the run came first. Non-EV Δ = change of non-emergency traffic, preemption vs "
        "baseline (positive = slower). Safety violations and drift are summed over all runs "
        "(baseline and preemption).",
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, "utf-8")
    for name, rows in (("ev_trips.csv", ev_rows), ("recoveries.csv", rec_rows),
                       ("runs.csv", run_rows)):  # fmt: skip
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                fh.write(f"# {fixture.disclaimer}\n")
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return text


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="ai/configs/emergency.yaml")
    p.add_argument("--fixture", default="toy_corridor")
    p.add_argument("--controllers", nargs="+", default=["fixed_time"])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1010)))
    p.add_argument("--scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = p.parse_args(argv)
    cfg = load_experiment_config(args.config, overrides={"fixture": args.fixture})
    fixture = load_fixture(args.fixture)
    out = RUNS_DIR / "emergency" / fixture.name / args.name
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for controller in args.controllers:
        for scale in args.scales:
            for i, seed in enumerate(args.seeds):
                for preempt in (False, True):
                    tag = f"{controller}_scale{scale:g}_seed{seed}_{'pre' if preempt else 'base'}"
                    logs = out / "logs" / tag if (i == 0 and preempt) else None
                    cmd = logs / "signal_commands.jsonl" if logs else None
                    sumo_out = out / "sumo" / tag
                    if controller == "fixed_time":
                        r = run_fixed_time(fixture, cfg, seed, scale, preempt, sumo_out, cmd)
                    elif controller == "ppo":
                        if args.checkpoint is None:
                            raise ValueError("--checkpoint is required for controller 'ppo'")
                        r = run_ppo(fixture, cfg, args.checkpoint, seed, scale, preempt, sumo_out,
                                    cmd)  # fmt: skip
                    else:
                        raise ValueError(controller)
                    if logs:
                        logs.mkdir(parents=True, exist_ok=True)
                        outputs = {"system_log_events.jsonl": r["events"],
                                   "emergency_requests.jsonl": r["requests"]}  # fmt: skip
                        for fname, items in outputs.items():
                            (logs / fname).write_text(
                                "".join(json.dumps(x) + "\n" for x in items), "utf-8"
                            )
                    evs = [e["duration_s"] for e in r["ev"]]
                    print(f"{tag:<40} EVs {len(evs):2d} mean transit "
                          f"{np.mean(evs) if evs else float('nan'):6.1f} s  preemptions "
                          f"{len(r['activations']):3d}  viol {int(r['safety_violations'])}  "
                          f"drift {int(r['drift_mismatches'])}", flush=True)  # fmt: skip
                    runs.append(r)
    print(summarise(runs, fixture, cfg, out))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
