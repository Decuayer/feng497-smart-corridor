# MARL formulation: state, action, reward and learning algorithm

*Module:* `ai/src/corridor_ai/marl/` · *Owner:* Demir Cücü · *Task:* TASK-301 · Implementation status: 301a (environment, IPPO smoke training)

## 1. Multi-agent setting

- **Agents** $i \in \mathcal{N} = \{$`alsancak_station`, `lozan`, `montro`, `basmane`, `cankaya`, `konak_pier`$\}$: one per signalised pilot intersection (ids from `shared/contracts/intersections.yaml`). `ai/configs/fixtures.yaml` maps each id to a SUMO traffic light of the fixture. Every other light in the network runs its fixed-time plan (see `signal_control.md`).
- **Decision epochs.** The simulation runs in 1 s steps. Agents act every $\Delta = 5$ s (`env.decision_interval_s`), simultaneously (PettingZoo `ParallelEnv`).
- **Episode.** 300 s warm-up under fixed-time for all lights (not part of the episode), then a horizon of $H$ controlled seconds: 3600 s by default, 1800 s in the 301a smoke configuration, so $T = H/\Delta$ decisions. The episode is **truncated**, not terminated, at the horizon, so the value of the final state is bootstrapped.
- Each agent $i$ controls a light with $K_i$ allowed green phases. $K_{\max} = \max_i K_i$: 2 on the toy corridor, 4 on `osm_alsancak_v0`.

The problem is a decentralised partially observable Markov game. Each agent sees only local measurements of its own intersection, while traffic couples neighbouring intersections.

## 2. Observation (state) $o_{i,t} \in [0,1]^{5K_{\max}+3+N}$

Features are aggregated **per green phase** $k$ over $\mathcal{L}_{i,k}$, the incoming lanes that get green in phase $k$. This gives every intersection the same input size, and feature $k$ lines up with action "switch to phase $k$".

| Block | Size | Definition | Normalisation |
|---|---|---|---|
| queue | $K_{\max}$ | $q_{i,k} = \sum_{l\in\mathcal{L}_{i,k}} h_l$ (halted vehicles, $v<0.1$ m/s) | $\div \sum_{l} c_l$, with lane capacity $c_l = \text{length}_l / 7.5$ m |
| waiting | $K_{\max}$ | $w_{i,k} = \sum_{l} W_l$ (cumulative waiting time of vehicles on $l$) | $\div (\sum_l c_l \cdot 120\text{ s})$ |
| occupancy | $K_{\max}$ | mean $\rho_l$ over $\mathcal{L}_{i,k}$ | already in $[0,1]$ |
| active phase | $K_{\max}$ | one-hot of the green shown (all 0 during yellow / all-red) | — |
| timing | 3 | $t^{\text{green}}/60$, $\mathbb{1}[t^{\text{green}} \ge 10]$, $\mathbb{1}[\text{in transition}]$ | — |
| starvation | $K_{\max}$ | seconds since phase $k$ was last green | $\div 120$ s |
| agent id | $N$ | one-hot index of the agent (needed for parameter sharing) | — |

Every entry is clipped to $[0,1]$. Padded phases ($k \ge K_i$) are 0. Dimensions: 19 on the toy corridor ($K_{\max}=2$), 29 on the OSM fixture ($K_{\max}=4$).

This covers the project's lane-level metrics: queue length $q_l$ (vehicles; metres = vehicles · 7.5 m), stopped vehicles, cumulative waiting time $W_l$, occupancy $\rho_l$ and elapsed phase time $t_{\text{phase}}$.

## 3. Action space and action masking

$a_{i,t} \in \mathcal{A} = \{0, 1, \dots, K_{\max}\}$:

- $a = 0$: **KEEP**. Extend the current green by $\Delta$.
- $a = k \ge 1$: **SWITCH** to green phase $k-1$. The phase machine inserts yellow (3 s) and all-red (1 s).

In the task's terms, *extend* = KEEP and *shorten* = SWITCH before the nominal split. With `phase_order: free` (default) any phase may follow any other. `cyclic` allows only the next phase in program order.

**Mask** $m_{i,t} \in \{0,1\}^{K_{\max}+1}$, with $t^g$ = green elapsed:

$$
m(0) = \begin{cases} 0 & t^g + \Delta > g_{\max} \\ 1 & \text{otherwise}\end{cases}
\qquad
m(k{+}1) = \mathbb{1}\big[t^g \ge g_{\min}\big]\cdot\mathbb{1}\big[k < K_i,\; k \neq k^{\text{cur}}\big]\cdot \mathbb{1}\big[\text{cyclic} \Rightarrow k = k^{\text{cur}}{+}1 \bmod K_i\big]
$$

plus three special cases:

- during yellow or all-red only $a=0$ is legal (a no-op);
- a light with $K_i = 1$ only has $a=0$;
- **starvation guard** (**ASSUMPTION**, `env.max_red_s` = 120 s): once $t^g \ge g_{\min}$, if some phase has waited longer than 120 s, the only legal action is switching to the most-starved phase.

At least one action is always legal (property-tested). Masked actions get logit $-10^9$, so their probability is effectively 0. The phase machine enforces the same rules independently, so even an unmasked or diverging policy cannot violate the limits. This implements the risk-plan requirement "if the MARL policy diverges, actions are constrained to agreed green-time bounds".

## 4. Reward

From the proposal: $R_t = -(\alpha \sum_l W(l,t) + \beta \sum_l S(l,t))$. Implemented per agent and averaged over the $\Delta$ seconds of the decision interval:

$$
r_{i,t} = -\left(\alpha\,\frac{\overline{W}_{i,t}}{W_{\text{ref}}} + \beta\,\frac{\overline{S}_{i,t}}{S_{\text{ref}}}\right),\qquad
\overline{W}_{i,t} = \frac{1}{\Delta\,|\mathcal{L}_i|}\sum_{\tau=1}^{\Delta}\sum_{l\in\mathcal{L}_i} W(l,\tau),\quad
\overline{S}_{i,t} = \frac{1}{\Delta\,|\mathcal{L}_i|}\sum_{\tau=1}^{\Delta}\sum_{l\in\mathcal{L}_i} S(l,\tau)
$$

| Symbol | Meaning | Default |
|---|---|---|
| $W(l,\tau)$ | Sum over vehicles on lane $l$ of their current waiting time (SUMO `lane.getWaitingTime`; time below 0.1 m/s since the vehicle last moved) | — |
| $S(l,\tau)$ | `halted`: number of halted vehicles. `stop_events`: number of vehicles that went from moving to halted during second $\tau$ | `halted` (decided). `stop_events` is reported as an extra metric through tripinfo `waitingCount` |
| $\alpha, \beta$ | weights | 1.0, 0.5 |
| $W_{\text{ref}}, S_{\text{ref}}$ | normalisers | 100 veh·s, 5 veh per lane |
| $\lvert\mathcal{L}_i\rvert$ | number of incoming lanes (`normalize_by_lanes: true`) | — |

All of these are in `env.reward` in `ai/configs/base.yaml`. Each agent receives its **own** reward. The legacy prototype gave every agent the same team sum.

## 5. Objective and algorithm

Each agent maximises the discounted return $G_{i,t} = \sum_{u\ge 0}\gamma^u r_{i,t+u}$ with $\gamma = 0.99$ per decision. With $\Delta = 5$ s the effective horizon is about 100 decisions, roughly 8 minutes.

### 5.1 Independent PPO with parameter sharing (IPPO-PS), chosen

- One actor $\pi_\theta(a\mid o, m)$ and one critic $V_\phi(o)$ are **shared by all agents**. The agent one-hot in $o$ lets the shared network specialise per intersection. The critic is local: it sees only $o_{i,t}$.
- Every agent's experience is a separate trajectory with its own advantages:

$$\hat A_{i,t} = \sum_{u\ge0} (\gamma\lambda)^u \delta_{i,t+u},\qquad \delta_{i,t} = r_{i,t} + \gamma V_\phi(o_{i,t+1}) - V_\phi(o_{i,t}),\quad \lambda=0.95$$

  with bootstrap $V_\phi(o_{i,T})$ at truncation.
- PPO-clip loss, over minibatches drawn from all agents:

$$\mathcal{L}(\theta,\phi) = -\mathbb{E}\Big[\min\big(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t, 1\pm\epsilon)\hat A_t\big)\Big] + c_v\,\mathbb{E}\big[(V_\phi(o_t) - \hat G_t)^2\big]/2 - c_e\,\mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot\mid o_t, m_t))\big]$$

  with $\rho_t = \pi_\theta(a_t\mid o_t,m_t)/\pi_{\theta_\text{old}}(a_t\mid o_t,m_t)$, $\epsilon=0.2$, $c_v=0.5$, $c_e=0.01$. Advantages are normalised per minibatch and gradient norms clipped at 0.5.
- Networks: two-layer MLPs of 128 tanh units for each of actor and critic, orthogonal initialisation. Adam, learning rate $3\cdot10^{-4}$, 4 epochs per episode, minibatch 256. All values in `ai/configs/marl_ppo.yaml`.

**Why IPPO-PS rather than MAPPO as the main method.** Independent PPO is simple and stable, and it scales to heterogeneous intersections once observations are phase-aggregated. It is also reported to be competitive with centralised-critic MAPPO on many cooperative benchmarks. Execution must be decentralised anyway (one controller per intersection, with fallback to local control if the link fails), and IPPO's decentralised critic needs no global state at training time either. A MAPPO variant (same actor, critic on the concatenated observations of all agents) is planned as a **NICE** ablation in 301b.

### 5.2 DQN (TASK-301b)

Parameter-shared independent Double DQN on the same observations, masks and rewards. Illegal actions get $Q=-\infty$ in both the greedy action and the target's argmax, and exploration is ε-greedy over legal actions.

## 6. Training protocol and reproducibility

- **Domain randomisation.** Each training episode samples a demand scale uniformly from [0.8, 1.3] (`ppo.train_scale_*`).
- **Seeds.** One experiment seed $k$ seeds Python, NumPy and PyTorch. Episode $e$ uses SUMO/route seed $10000(k+1)+e$. Evaluation seeds are 1000–1009, and Webster calibration seeds are the evaluation seed + 100000. The three sets never overlap.
- **Logging.** TensorBoard under `ai/runs/<name>/<fixture>/<run>/`:
  - training: `train/episode_return_mean`, `train/episode_return/<agent>`, `train/demand_scale`, `loss/{policy_loss,value_loss,entropy}`, `ppo/{approx_kl,clip_frac,explained_variance}`;
  - simulation: `sim/{mean_travel_time_s,mean_waiting_time_s,mean_stops,co2_g_per_km,arrived,unfinished,teleports}`;
  - safety: `safety/{violations,drift_mismatches,illegal_actions}`;
  - evaluation: a deterministic episode on seed 999 every `ppo.eval_every` episodes, under `eval/*`.

  Each run also writes `config.yaml` (with the fixture disclaimer) and `episodes.csv`.
- **Evaluation** (`corridor_ai.marl.evaluate`). Same scenario, same seeds and same warm-up for every controller (fixed-time, Webster, PPO greedy, and actuated as reference). Metrics cover vehicles departing after the warm-up; unfinished vehicles are *included* so gridlock cannot hide delay. Results are reported as mean ± 95 % CI. Every table carries the disclaimer *"relative improvement on a placeholder fixture, not the real digital twin"*.

## 7. Open questions

1. `S` definition: `halted` is the training default (decided). Should the course report also include a `stop_events`-trained variant?
2. Reward weights α/β and the normalisers are first guesses. A sensitivity sweep is planned for 301b.
3. The starvation guard of 120 s is an ASSUMPTION; operators may want a different value.
4. Phase-order constraints on the ESP32 may force `cyclic` (request to Tolga).
