# HDTN-SCN — Complete RL Implementation (MAPPO + GNN + Masking + MPC-Residual)

This document specifies the **complete reinforcement-learning system** that replaces the hand-designed DT-migration heuristic in HDTN-SCN. It is the concrete, buildable version of the algorithm chosen earlier:

> **CTDE multi-agent actor–critic — MAPPO** — with a **GNN observation encoder**, **action masking** as a safety shield, and a **model-based (MPC) lookahead** exploiting the known orbital dynamics.

## What the RL system replaces (scope, exact)

It replaces **only the migration *decision***: the paper's *DT Migration* logic and the entire *Influence of Latency Threshold* sweep (Fig. 5). Concretely it subsumes:

- the fixed **latency-threshold trigger** (learns *when* to migrate),
- the **nearest-GS destination rule** (learns *where*),
- **load-awareness** and **regular/pre-planned migration** (folded into the learned, dynamics-aware policy).

It does **not** touch Module 1 (physics/graph), the migration *mechanics* (`MigrationExecutor`, addressing tables, seamless duplication), Pillar B slicing, or fault *detection*. In code: the system is a drop-in for `module2_dt_control/policies/rl.py :: RLMigrationPolicy.decide()` and removes the threshold check in `module3_simulation/simulator.py`.

## The four RL modules (map)

```
module1_environment  ─┐  (the physics world — unchanged)
module2_dt_control   ─┤  (DT plane; policy plug-point — unchanged except rl.py)
module3_simulation   ─┘  (benchmarks + eval harness — reused for evaluation)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ MODULE 4  module4_marl_env        Dec-POMDP wrapper: obs, global state,     │
│                                   action masking, multi-objective reward,   │
│                                   trace logging.        (the RL "contract") │
├───────────────────────────────────────────────────────────────────────────┤
│ MODULE 5  module5_representation  GNN topology encoder + temporal encoder + │
│                                   MPC/orbital lookahead feature engine.     │
│                                   (turns raw obs → learnable embeddings)    │
├───────────────────────────────────────────────────────────────────────────┤
│ MODULE 6  module6_mappo           CTDE core: shared actor (residual-over-   │
│                                   MPC), centralized GNN critic, GAE rollout │
│                                   buffer, masked PPO loss, offline warm-start│
├───────────────────────────────────────────────────────────────────────────┤
│ MODULE 7  module7_train_eval_gui  training orchestration + curriculum +     │
│                                   evaluation vs benchmarks + REAL-TIME GUI  │
└───────────────────────────────────────────────────────────────────────────┘
```

Dependency direction: **7 → 6 → 5 → 4 → (1,2,3)**. Each module is independently testable with a smoke test, mirroring Modules 1–3.

**Novelty threaded through all four (so this is a contribution, not a bolt-on):**
1. **Residual-over-MPC actor** — the policy learns a *correction* to the analytic optimal-nearest-GS/MPC action, not a policy from scratch. Exploits the ~90% known dynamics; the network only learns the stochastic residual (faults, congestion, inter-agent contention).
2. **Contention-aware centralized critic** — the critic is a GNN over the **bipartite DT↔GS assignment graph**, so it sees load coupling explicitly (what QMIX/VDN cannot represent).
3. **Differentiable safety shield** — action masking on infeasible/overloaded GS, applied to logits; the learned policy *cannot* emit an unsafe action, and a Lagrangian on capacity anneals soft→hard.
4. **Event-triggered "stay" as a first-class action** — replaces the fixed threshold; the agent learns its own operating point per satellite per step.
5. **Inductive GNN → zero-shot con1→con2 transfer** as a headline evaluation, not just a training curve.
6. **AoI/sync-coupled reward** — migration cost and Age-of-Information enter the objective, connecting migration to the paper's synchronization pain.

---

# MODULE 4 — `module4_marl_env` (the Dec-POMDP contract)

**Responsibility.** Turn `DTControlPlane` (Module 2) into a standards-compliant multi-agent environment with the exact observation / global-state / action / reward / mask surfaces MAPPO needs. This is the single source of truth for the RL problem definition.

**File tree**
```
module4_marl_env/
├── __init__.py
├── marl_env.py          # HDTNParallelEnv (PettingZoo ParallelEnv API)
├── obs_builder.py       # per-agent local obs + centralized global state
├── action_space.py      # discrete {stay} ∪ 𝒢, index<->GS mapping
├── masking.py           # feasibility mask (capacity, reachability, self)
├── reward.py            # multi-objective reward + Lagrangian terms
├── trace_logger.py      # (s,a,r,s',mask) logging for offline warm-start
└── smoke_test.py
```

**Agents.** One agent per edge-DT: `agent_id = sat_id`. `N = 66` (con1) / `264` (con2). Homogeneous → **parameter sharing** across agents (one network, agent-id/embedding as input).

**Observation (per agent, feeds the actor).** Built by `obs_builder.local_obs(sat_id, obs)` from Module 1's `Observation`:
- current PS–DT latency, one-step predicted latency (`Observation.predicted_latency`),
- candidate latency to each reachable GS (vector over 𝒢, `-1`/masked where invisible),
- current host id (one-hot / embedding), time-since-last-migration (AoI proxy),
- per-candidate GS normalized load, GS distance rank,
- **MPC lookahead feature** (from Module 5): predicted latency-if-stay vs latency-if-move over horizon H.

**Global state (centralized, feeds the critic only — CTDE).** `obs_builder.global_state(obs)`:
- the full bipartite **DT↔GS assignment** (who is where), all GS loads/capacities, all current latencies, the constellation graph snapshot handle. This is available *only at the NCC* — exactly the paper's central-DT — so CTDE respects the architecture.

**Action.** Discrete `a ∈ {0=stay, 1..|𝒢|}` → GS index via `action_space.decode`. `stay` is first-class (event-triggered control; replaces the threshold).

**Mask.** `masking.feasible(sat_id, obs) → bool[|𝒢|+1]`: `stay` always feasible; GS feasible iff visible/reachable *and* `load < capacity` *and* `≠ current host`. The shield guarantees no unsafe action is ever executed, regardless of policy output.

**Reward (multi-objective, per step, shared-team + per-agent shaping).**
```
r_t = − w_lat · mean_latency
      − w_mig · n_migrations
      − w_load · load_variance(GS)
      − w_aoi · mean_age_of_information
      − β_t · Σ overload_violation        # Lagrangian, annealed
```
`w_*` are config; `β_t` is the dual variable updated by Module 6 (safe-RL). Defaults + all weights live in `reward.py` and are logged. Team reward = global; add small per-agent potential-based shaping (its own latency delta) to reduce credit-assignment variance.

**API (PettingZoo ParallelEnv).**
```python
class HDTNParallelEnv:
    def reset(self, seed=None): -> obs_dict, info      # obs_dict[agent] incl. "action_mask"
    def step(self, actions): -> obs, rewards, term, trunc, info
    def state(self): -> np.ndarray                      # centralized global state
    agents: list[str]; possible_agents: list[str]
```
Wraps `DTControlPlane`: `reset()` calls `cp.reset()`; `step(actions)` decodes → `cp.apply_actions()` → `env.step({})` to advance time → rebuilds obs. Reuses Module 2's executor so migration mechanics/addressing stay identical.

**Integration.** Depends on Modules 1+2 only. `trace_logger` runs the **greedy** policy through this same env to produce the offline dataset Module 6 warm-starts from — guaranteeing identical (obs, mask, action) formats between behavior policy and learner.

**Checkpoint.** Random-policy rollout runs; masks never allow overloaded/invisible GS; reward decomposes into logged components; PettingZoo API conformance test passes.

---

# MODULE 5 — `module5_representation` (encoders + physics lookahead)

**Responsibility.** Map raw observations/global state into embeddings, and manufacture the **MPC lookahead** feature that makes the whole system sample-efficient. This is the "why RL will beat a flat-vector agent" module.

**File tree**
```
module5_representation/
├── __init__.py
├── graph_builder.py     # NetworkGraph.G -> torch_geometric HeteroData
├── gnn_encoder.py       # heterogeneous GNN (sat/GS/NCC nodes; ISL/GSL/terr edges)
├── temporal_encoder.py  # per-agent GRU/TCN over history (traffic, latency, status)
├── mpc_lookahead.py     # analytic H-step roll-forward -> per-action latency preview
├── backbone.py          # fuse GNN + temporal + MPC -> actor/critic feature tensors
└── smoke_test.py
```

**GNN encoder (`gnn_encoder.py`).** Heterogeneous message passing (GraphSAGE/GAT-style) over `NetworkGraph.G`: node types {sat, gs, ncc}, edge types {isl, gsl, terrestrial} with **edge weight = propagation delay**. Outputs a per-node embedding. **Inductive** (aggregates neighborhoods, no fixed node count) → a model trained on con1 runs unchanged on con2. This is the transfer story.

**MPC lookahead (`mpc_lookahead.py`) — the novel core.** Because orbital motion is analytic (`constellation.positions(t+kΔ)`), for each agent roll the world forward `H` steps and compute, per candidate action, the **predicted cumulative latency + migration cost**:
```python
def preview(sat_id, env, H) -> dict[action -> predicted_cost]:
    # deterministic roll-forward using known dynamics; no learning here.
```
Two uses: (a) a **feature** appended to the observation (the agent sees "what the physics says"); (b) the **anchor action** for the residual actor (Module 6). This is what lets a small policy learn only the stochastic residual (contention, faults, congestion) instead of re-deriving orbital mechanics.

**Temporal encoder.** GRU/TCN over each agent's recent history → forecasts latent traffic/latency trend feeding `predicted_latency` with a *learned* multi-step signal (upgrades Module 1's 1-step analytic lookahead).

**Backbone.** `backbone.actor_features(obs)` and `backbone.critic_features(global_state)` fuse (GNN node embedding for this agent) ⊕ (temporal state) ⊕ (MPC preview) ⊕ (raw scalar obs). Critic additionally consumes the bipartite DT↔GS graph embedding (contention-aware critic).

**Integration.** Pure functions of Module 4's obs/state + Module 1's live graph; returns tensors to Module 6. Torch / torch-geometric only here.

**Checkpoint.** Same encoder weights accept con1 (66) and con2 (264) tensors without shape errors (inductive test); MPC preview for `stay` matches Module 1's true next-step latency within float tolerance.

---

# MODULE 6 — `module6_mappo` (CTDE learner)

**Responsibility.** The MAPPO algorithm: decentralized shared actor + centralized critic, masked action distribution, residual-over-MPC parameterization, GAE rollout buffer, PPO-clip update, Lagrangian safety, and offline behavior-cloning warm-start.

**File tree**
```
module6_mappo/
├── __init__.py
├── actor.py             # shared policy head: residual over MPC anchor, masked logits
├── critic.py            # centralized value net over global (contention) graph state
├── rollout_buffer.py    # multi-agent trajectories, GAE-λ, advantage normalization
├── mappo_learner.py     # PPO-clip loss, entropy, value loss, KL early-stop
├── lagrangian.py        # dual variable β_t update for capacity constraint (safe RL)
├── warm_start.py        # BC / offline pretrain from greedy traces (Module 4 logger)
├── mappo_policy.py       # MAPPOPolicy(MigrationPolicy) — inference adapter for Module 2
└── smoke_test.py
```

**Actor (`actor.py`) — residual + masked.** Shared across agents (parameter sharing). Given `backbone.actor_features`:
```
anchor  = one-hot(MPC.best_action)          # physics prior
logits  = anchor_bias · anchor + MLP(features)   # learn a correction
logits  = logits.masked_fill(~mask, -inf)   # differentiable safety shield
π(a|o)  = softmax(logits)
```
So the policy defaults to the physics-optimal action and learns *when to deviate*. Discrete Categorical; entropy bonus for exploration.

**Critic (`critic.py`) — centralized, contention-aware.** GNN over the bipartite DT↔GS assignment graph (from Module 5) + global scalars → scalar `V(state)`. Sees all agents' placements and loads → captures the coupling MAPPO needs and QMIX/VDN cannot. Used only at training time.

**Rollout & update.** `rollout_buffer` stores per-agent `(obs, mask, action, logp, reward, value)`; compute **GAE-λ** advantages on the team return, normalize. `mappo_learner` does the standard **PPO-clip** actor loss + clipped value loss + entropy, with **KL early-stopping** and minibatch epochs. Shared actor → all agents contribute gradients to one network.

**Safe RL (`lagrangian.py`).** Dual ascent on `β_t` from the observed overload-violation rate; starts soft (learning explores), anneals toward hard as violations must → 0. Complements the hard mask (mask = never execute unsafe; Lagrangian = discourage getting *close* to capacity).

**Warm-start (`warm_start.py`).** Behavior-clone the actor from greedy traces (Module 4 `trace_logger`), optional CQL-style conservative value pretrain for the critic. Removes cold-start thrash; gives a *safe fallback equal to greedy* on step 0.

**Inference adapter (`mappo_policy.py`).** `MAPPOPolicy(MigrationPolicy)` implements `.decide(dt, obs, t)`: run actor forward on `backbone.actor_features`, apply mask, argmax/sample → GS id or `None`. **This is the drop-in for `policies/rl.py`** — Modules 1/2/3 need zero changes; the simulator runs it exactly like the greedy policy.

**Integration.** Consumes Module 5 features + Module 4 rollouts; emits a policy Module 2 can execute and Module 3 can evaluate.

**Checkpoint.** On a tiny 5-agent env, MAPPO return climbs above random and reaches ≥ greedy within a short budget; KL stays bounded; masked actions never sampled; warm-started actor ≈ greedy at iteration 0.

---

# MODULE 7 — `module7_train_eval_gui` (orchestration, evaluation, real-time GUI)

**Responsibility.** Drive training with config + curriculum, evaluate against the paper's benchmarks (reusing Module 3), and provide the **real-time GUI** to watch learning and convergence live.

**File tree**
```
module7_train_eval_gui/
├── __init__.py
├── config/
│   ├── mappo_con1.yaml       # lr, clip, γ, λ, H, reward weights, net sizes
│   └── curriculum.yaml       # con1 -> con2 schedule
├── train.py                  # main training loop (env↔buffer↔learner), checkpoints
├── curriculum.py             # constellation growth + transfer-eval hooks
├── evaluate.py               # RL vs greedy/B1/B2/B3 on Module 3 protocol
├── experiments/
│   └── fig6_rl_vs_greedy.py  # overlay learned policy on Fig.4/Fig.5 axes
├── logging/
│   ├── metrics.py            # scalars: return, latency, migrations, entropy, KL, β
│   └── event_bus.py          # streams live state to the GUI (queue / websocket)
├── gui/
│   ├── app.py                # Plotly Dash dashboard (real-time)
│   ├── panels.py             # reward curve, latency dist, GS-load heatmap, KL/entropy
│   └── constellation_view.py # live world map: sats, GS, migration events animated
└── smoke_test.py
```

**Training (`train.py`).** Standard MAPPO loop: collect `T` steps across all agents into `rollout_buffer` → compute GAE → `mappo_learner.update()` → repeat. Checkpoints, resumable, seeded. Config-driven so no code edits to sweep.

**Curriculum (`curriculum.py`).** Train on con1 (66 sats); periodically **evaluate zero-shot on con2 (264 sats)** using the inductive GNN — the headline generalization result. Optionally fine-tune on con2.

**Evaluation (`evaluate.py`, `fig6_*`).** Wrap the trained `MAPPOPolicy` and run it through **Module 3's existing harness** against greedy/B1/B2/B3 on the exact Fig.4/Fig.5 protocol. Report: mean & p95 latency, migration frequency, **load variance**, AoI, reliability-violation rate, and **con1→con2 transfer**. The win must show on the *coupled* metrics (load/reliability/migration cost), since greedy already near-ties on pure latency — this is stated up front so the result is honest.

**Real-time GUI (`gui/`) — the end-goal deliverable.**
- **Stack:** **Plotly Dash** (interactive, live-updating via `dcc.Interval` + a `event_bus` queue the trainer writes to). Alternative: Streamlit for speed, or TensorBoard for scalars only. Dash chosen for the custom constellation map.
- **Live panels:**
  1. **Convergence** — episode return, actor/critic loss, **policy entropy**, **KL divergence**, Lagrangian `β_t` (all streaming).
  2. **Latency** — live mean & p95 PS–DT latency vs the greedy baseline line (is RL beating it *now*?).
  3. **GS load heatmap** — per-ground-station load over time (watch the agent learn load balancing).
  4. **Migration rate** — migrations/step vs greedy (watch it learn to migrate *less* for the same latency).
  5. **Constellation view** (`constellation_view.py`) — world map with satellites, ground stations, and **animated migration events** (arc from old→new GS), colored by decision confidence. This is the "see the RL think" panel.
  6. **Transfer panel** — con1 training vs con2 zero-shot eval side by side.
- **Wiring:** trainer pushes a compact state dict to `event_bus` every K steps; the Dash app polls the bus and redraws. No coupling into the hot training loop (GUI is a consumer, never blocks the learner). Runnable headless (train) or with `--gui` (launch dashboard).

**Checkpoint.** `train.py --config mappo_con1 --steps small` produces rising return + checkpoints; `evaluate.py` reproduces Fig.4/Fig.5 axes with the RL curve overlaid; `gui/app.py` renders and updates live from a running trainer.

---

## Build order & phase gates

1. **M4** first — lock the Dec-POMDP contract; validate with a random policy + the greedy trace logger. *Gate: PettingZoo-conformant, masks safe, reward decomposes.*
2. **M5** — encoders + MPC preview; validate inductive shapes + MPC-vs-truth. *Gate: con1/con2 shape-agnostic; MPC `stay` matches truth.*
3. **M6** — MAPPO on a tiny env; warm-start = greedy at step 0, then improves. *Gate: return ≥ greedy on toy; KL bounded; no masked actions sampled.*
4. **M7** — full training on con1, evaluate vs benchmarks, con1→con2 transfer, then GUI last. *Gate: RL beats greedy on coupled metrics; GUI streams live.*

## Honesty gates (do not skip)

- **Greedy is near-optimal for pure latency.** The RL win must appear on **load variance + migration cost + reliability + transfer**, or the result is null. Reward weights in M4 make this explicit and logged.
- **MPC is the real baseline.** If model-free ever beats residual-over-MPC only marginally, report that — the physics prior may be most of the value.
- **Sim fidelity caveat.** Latency is propagation-only today; congestion/queuing (a Module-5 neural surrogate) is needed before strong latency-reduction claims. State this in every result.
- **Report transfer, not just training curves.** con1-overfit that fails con2 is a negative result and must be reported as such.

## Minimal dependency addendum

```bash
pip install torch torch-geometric gymnasium pettingzoo dash plotly
# (existing: numpy scipy pandas matplotlib networkx pyyaml)
```
