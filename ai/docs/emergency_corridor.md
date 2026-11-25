# Proactive emergency green corridor (TASK-302)

*Module:* `ai/src/corridor_ai/emergency/` · *Owner:* Demir Cücü

When an emergency vehicle (EV; ambulance or fire truck) requests priority, the system pre-clears not only the intersection the EV is approaching, but also **the next two intersections on its route**. This forms a green wave that flushes queues ahead of the vehicle ("zipper" clearance). After the EV has passed, control returns to the normal controller (fixed-time, Webster or MARL) without violating any signal safety limit.

The algorithm sits on top of the safety-constrained signal layer (`signal_control.md`). It only *requests* phases. The PhaseMachine still enforces min green 10 s, max green 60 s, yellow 3 s and all-red 1 s, and the independent SafetyMonitor checks every run.

## 1. Inputs

| Source | How | Contract |
|---|---|---|
| Simulation | SUMO vehicles of vType `emergency` (ambulance) or `firebrigade` (fire truck), detected on departure | `emergency_request` built with `source: simulation`, `auth.method: simulation` |
| V2I → backend → ai | `PreemptionManager.submit_message(msg)`. The message must validate against the schema and be verified by the backend (`auth.verified = true`); unverified requests are rejected and logged. In simulation the message must name the vehicle (`sim.sumo_vehicle_id`) | `emergency_request` (DRAFT) |

`request_type: cancel` ends the preemption for that request.

## 2. Algorithm

```mermaid
flowchart LR
  A[EV detected /<br/>request accepted] --> B[Route look-ahead:<br/>ordered agent intersections<br/>+ serving phase per movement]
  B --> C[Every second: distance and ETA<br/>to each upcoming stop line]
  C --> D{For current + next 2:<br/>ETA_opt ≤ T_flush + T_switch + margin?}
  D -- yes --> E[Activate preemption<br/>(arbitration: priority, then ETA)]
  E --> F[Request serving phase every second<br/>PhaseMachine enforces limits;<br/>reservice if hold would exceed 60 s]
  F --> G{EV passed the light?}
  G -- yes --> H[Release → base controller resumes<br/>from the current safe state]
  D -- no --> C
```

### 2.1 Route look-ahead (`emergency/network.py`)

The EV's route (a list of edges) is scanned for **movements** (incoming edge → outgoing edge) through *agent* intersections. For each movement we know:
- the controlling light and its link indices;
- the approach lanes;
- the **serving phases**: green phases in which the movement has priority green `G` (permissive `g` only if no `G` phase exists).

Only the first **1 + lookahead = 3** unpassed agent intersections are candidates (`emergency.lookahead: 2`).

### 2.2 ETA per intersection

With d = distance along the route to the stop line, and EV desired speed = speed limit × speedFactor (1.3):

$$\text{ETA}_\text{opt} = \sum_{\text{edges to stop line}} \frac{\ell_e}{v_e^{\max}\cdot 1.3}, \qquad
\text{ETA}_\text{nom} = \max\!\Big(\text{ETA}_\text{opt},\ \frac{d}{\max(\bar v_\text{EMA},\ 0.6\, v^{\max}\cdot 1.3)}\Big)$$

The optimistic ETA is used to decide *when to start* clearing, so it errs towards starting early. The nominal ETA ranks competing EVs and decides how long a green will be held.

### 2.3 Queue-flush lead time and activation

A light is pre-cleared as soon as

$$\text{ETA}_\text{opt} \le T_\text{flush} + T_\text{switch} + \text{margin}, \qquad
T_\text{flush} = t_\text{startup} + n\cdot h_\text{sat}\;(\text{0 if } n = 0)$$

| Parameter | Value | Status |
|---|---|---|
| n | Largest number of vehicles **between the EV and the stop line on one approach lane** (`flush_count: vehicles_ahead`). The moving platoon in front of the EV must clear as well as the standing queue. `halted` counts only the queue | — |
| Start-up lost time t_startup | 2 s | ASSUMPTION |
| Saturation headway h_sat | 2 s/veh | ASSUMPTION |
| Margin | 3 s | ASSUMPTION |
| T_switch | Time until a serving phase can show green **under the safety limits**: 0 if it is already green; otherwise remaining min green + 3 s yellow + 1 s all-red, or the remainder of a running transition (`time_to_green`) | — |

Because downstream intersections have longer ETAs, they are triggered when *their own* flush time requires it. The result is a green wave that flushes intersection *k+2* early enough for the platoon released at *k* and *k+1*.

**Design choice from diagnosis.** A first version counted only halted vehicles. On the debugging seeds (2000–2001, never used for acceptance), most remaining EV stops (53 of 86) happened with the light already **green**, inside a queue that had not finished discharging. Counting the whole platoon ahead of the EV removed most of them. That raised the EV improvement on the debugging seeds from 44 → 49 % (scale 1.0) and 47 → 52 % (scale 1.5).

### 2.4 Arbitration over the base controller

- There is at most one active preemption per light. Candidates from all EVs are ranked by (priority, nominal ETA).
- A running preemption is displaced only by a **strictly higher** priority, so two equal-priority EVs never make a light flip back and forth.
- For a preempted light, the preemption request **overrides** the base controller every second:
  - fixed-time and Webster plans are simply overridden;
  - in the MARL environment, the agent's action mask becomes KEEP-only and `infos[agent]["preempted"] = True`. TASK-301b excludes these transitions from the PPO update.
- Lights that are not on the EV's route keep their normal control.
- Only the six pilot (agent) intersections are preempted. On the OSM corridor the route also crosses four background lights. They keep their fixed-time plan in both runs (with and without preemption) and limit the achievable improvement there.

### 2.5 Phase scheduling under the safety limits

- The serving phase is requested every second. The PhaseMachine never truncates a green below 10 s and always inserts 3 s yellow + 1 s all-red.
- **Max green (60 s)** still applies while preempted. If holding the serving green would exceed 60 s before the EV arrives, the light **re-services** the other phases (min green) and comes back. This happens only if even the optimistic ETA leaves time for that (`min green + 2·(yellow + all-red) + margin + T_flush`), so the EV does not meet the re-serviced phase. Otherwise the machine's forced switch at 60 s still applies and the serving phase is requested again right afterwards.

### 2.6 Hold budget

A pre-clear that has been held for `max_hold_s` = 90 s (ASSUMPTION) without the EV passing is released with reason `hold_timeout`. That EV/intersection pair may then only re-activate when its **nominal** ETA (observed speed) meets the trigger, not its optimistic one.

**Why.** On the OSM placeholder corridor the KPI route crosses four non-preempted background lights. EVs can be stuck there or in congestion while downstream agent lights keep holding the corridor green on the optimistic ETA. Holds of 100–206 s were observed. Cross streets then starved, spillback blocked the corridor, and the network gridlocked. On 20 OSM debugging seeds (2000–2019) the budget removed one of the two preemption-induced gridlocks. A third gridlocking seed also gridlocks without any preemption, i.e. the placeholder scenario itself is overloaded under fixed-time. On the toy corridor the budget changed nothing. See *Limitations*.

### 2.7 Release, handback and recovery

- **Release.** When the EV's route index moves past the approach edge (it has crossed the stop line), the preemption ends and a `preemption_ended` event is logged.
- **Handback.** The base controller resumes **from the current phase-machine state**. There is no forced switch and no reset. Min green, yellow and all-red stay enforced, so the handback can never violate a limit. The first phase change after a release is emitted as a `signal_command` with `mode: recovery`.
- **Recovery, online indicator.** A `recovery_completed` event is logged when the intersection's total queue is back to its level at activation, within tolerance.
- **Recovery, acceptance metric (offline).** The per-second intersection queue is compared with the **matched run without preemption** (same seed, demand, controller and EVs). Recovery time τ is the smallest delay after release such that, for 60 consecutive seconds, Q_pre − Q_base ≤ max(10 % · Q_base, 2 veh). Both series are smoothed with a 30 s moving average. The test is **one-sided**: an intersection that ends up better off than without preemption counts as recovered. If the next preemption at the same intersection, or the end of the run, comes first, the release is *censored*. The bound for "bounded recovery" is 180 s (ASSUMPTION).

## 3. Outputs

| Output | Contract |
|---|---|
| One message per accepted EV | `emergency_request` (schema-validated) |
| Events `emergency_request_received`, `preemption_started` (ETA, vehicles to clear, look-ahead rank), `preemption_ended` (duration, reason, reservices), `recovery_completed`; `synthetic: true` | `system_log_event` (feeds TASK-303) |
| Phase changes during preemption: `issuer: emergency_preemption`, `mode: preemption`, `emergency_request_id`; the first change after release: `mode: recovery` | `signal_command` |

## 4. Simulated scenario (placeholder)

| Item | Value | Status |
|---|---|---|
| EVs per run | 10 on the fixture's KPI corridor route, evenly spaced between warm-up + 120 s and end − 900 s with seeded jitter; 30 % fire trucks | — |
| Pairing | The same seed gives identical EV times and background demand, with and without preemption | — |
| EV behaviour | **Obeys signals** (no SUMO `bluelight` device, so other drivers do not yield); speedFactor 1.3 | Approved ASSUMPTION; the bluelight sensitivity run is a NICE item |
| `toy_corridor` | `eastbound` route W → E, 3.0 km, 6 agent lights | — |
| `osm_corridor` | Corridor-focused demand (background trips + corridor through traffic + cross traffic at every agent light, so every controlled light sees demand); KPI route `konak_to_alsancak`, 2.97 km, 6 agent lights + 4 background lights, generated by `ai/scripts/build_osm_corridor.py` | — |

## 5. Acceptance measurement

Run with `python -m corridor_ai.emergency.evaluate` (commands in `ai/README.md`).

- **Baseline** (ASSUMPTION, approved): same scenario, same seeds, same EV insertion times, **no preemption**. Reported separately for the fixed-time and the MARL controller.
- **EV transit time** = tripinfo `duration` of each EV over the full corridor route.
- **Improvement** = 1 − median(T_preempt) / median(T_baseline). The mean-based value and a paired Wilcoxon test on per-seed means are also reported.
- **Target:** ≥ 50 %.
- **Safety:** monitor violations and drift summed over all runs (target 0).
- **Recovery:** distribution of τ, share ≤ 180 s, censored count.
- **Side effects:** change in non-EV mean travel time and waiting time.

Results: see `ai/README.md` (section *302*). Every table carries the placeholder-fixture disclaimer.

## 6. Limitations

- **Gridlock risk with fixed-time base control on the OSM placeholder.** Even with the hold budget, 1 of 20 OSM debugging seeds still gridlocked when preemption was combined with the *fixed-time* base controller, while the same seed did not gridlock without preemption. With the adaptive (MARL) base controller, no gridlock was observed. A spillback-aware activation rule (do not hold the corridor green when the EV's exit or the cross streets are jammed) is the proposed next step. The OSM scenario is an uncalibrated placeholder, and its fixed-time baseline is itself close to saturation.

- EVs do not get a rescue lane, so after pre-clearing they still follow slower cars at about 50 km/h. On the 3 km toy corridor this alone costs about 50 s compared with free flow (~170 s), which caps the achievable improvement.
- Non-agent lights on the route are not preempted.
- Flush model parameters (2 s start-up, 2 s/veh) are ASSUMPTIONS until calibrated on the real network and with `analysis/` (Ayça).
- The MARL policies used here are short "smoke" policies from 301a, and on the OSM corridor a 40-episode policy. Full training is TASK-301b.
