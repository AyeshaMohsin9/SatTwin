# SatTwin

An open-source research repository for **AI-driven Digital Twins of Satellite–Terrestrial (Non-Terrestrial) Networks**. It starts from the HDTN-SCN architecture (Zhou et al., *IEEE ComMag* 2023) and pushes it into a **learning-native** digital-twin system: every control decision, every model inside the twin, and the synchronization loop itself become learnable.

## Research Areas

Digital Twin Networks · Satellite Communications · 6G / NTN · Reinforcement Learning for Wireless · Graph & Sequence Models · Resource Allocation · Network Synchronization · Semantic Communications

## Repository Structure

```
docs/          Papers (PDF) + one-paragraph summaries in ideas.md
ideas.md       Structured summaries of the source papers
code/
  HDTN-SCN/    Reproduction of the HDTN-SCN paper as a 3-module simulator (see below)
    module1_environment/   physical + network substrate (Gym-style env)  ← the RL "world"
    module2_dt_control/    DT control plane; pluggable MigrationPolicy    ← RL plugs in here
    module3_simulation/    simulator, 4 benchmarks, Fig.4/Fig.5 experiments
README.md      This file — the end-to-end ML/RL enhancement roadmap
```

The simulator already exposes exactly the surfaces a learning system needs:
- `module1_environment/environment.py` — `reset() / observe(t) / step(actions) → (obs, reward, info)`, reward `= −(mean_latency + migration_cost·migrations)`.
- `module1_environment/observation.py` — `Observation.feature_vector(dt_id)` (current/predicted/candidate latencies + normalized GS loads).
- `module2_dt_control/migration.py` — `MigrationPolicy.decide(dt, obs, t)` interface + policy-agnostic `MigrationExecutor`.
- `module2_dt_control/policies/{greedy,rl}.py` — the greedy baseline and the RL stub sharing one interface.

**Everything below is written as concrete plug points into those files.** The design rule is: *a new learning method should be a new policy/model class, not a rewrite of the environment.*

---

# 1. Reframe HDTN-SCN as a learning problem

The paper's three pillars are hand-designed heuristics. Each is really a **sequential decision / prediction problem** under a partially observed, fast-varying, deterministic-plus-stochastic environment. That is the opening for RL and ML.

### 1.1 The migration MDP (single-DT view)

For one edge-DT mirroring satellite `k`:

- **State** `s_t` = `Observation.feature_vector(k)` — current host, current PS–DT latency, candidate latency to every GS, one-step predicted latency, per-GS normalized load. (Already produced by Module 1.)
- **Action** `a_t ∈ {stay} ∪ 𝒢` — keep the DT or migrate it to ground station `g` (discrete, `|𝒢|+1` actions). Maps 1:1 to `RLMigrationPolicy._decode`.
- **Reward** `r_t = −(latency_k + c·migrated_k + λ·overload_penalty + μ·reliability_violation)` — extends the reward already in `environment.step`.
- **Transition** — satellite motion is **deterministic and known** (orbital mechanics in `constellation.py`); only faults/traffic are stochastic. → this is a *known-dynamics* MDP, which unlocks model-based RL / MPC (§3.1).

### 1.2 Why it is really a Dec-POMDP (multi-DT view)

`N` edge-DTs share finite GS capacity (`GroundStation.capacity`). One DT's migration changes the load another DT observes → the agents are **coupled**, each sees only local state → **decentralized partially-observable MDP (Dec-POMDP)**. This is the technically correct framing and it maps *perfectly* onto the paper's hierarchy:

- **Edge-DT = decentralized actor** (executes with local observation, low latency).
- **Central-DT / NCC = centralized critic** (trains with global state).

That is exactly **Centralized Training, Decentralized Execution (CTDE)** — the hierarchy in the paper *is* the RL training topology. This alignment (§4) is the strongest conceptual contribution SatTwin can make.

---

# 2. Where each technique plugs in (the map)

| Paper pillar / component | Code location | Heuristic today | Learning upgrade | Family |
|---|---|---|---|---|
| **A. DT migration** | `policies/rl.py` | greedy nearest-GS | value/policy RL, then MARL | RL / MARL |
| **A. Migration timing** | `simulator.py` loop | fixed latency threshold | learned/event-triggered timing | RL / bandit |
| **A. Fault → irregular migration** | `control_plane.inject_*` | manual injection | anomaly detection triggers it | Self-supervised / Bayesian |
| **B. QoS slicing** | `slicing.py` | static type→slice map | slice admission + bandwidth split | Contextual bandit / DRL |
| **B. Case-study routing (K strategies)** | (Module 2, new) | heuristic K paths | GNN + RL route generation | GNN-RL |
| **C. Addressing** | `addressing.py` | reactive table update | predictive prefetch/caching of mappings | Sequence model |
| **DT fidelity (the model itself)** | `digital_twin.model_state` | scalar fields | neural surrogate of channel/latency/state | GNN / Neural-ODE / Transformer |
| **Synchronization loop** | `environment.observe` predict | 1-step lookahead | what/when-to-sync, semantic compression | RL + AoI + info theory |
| **State representation** | `observation.feature_vector` | flat vector | graph/temporal encoder | GNN / Transformer |

The rest of the document expands each row into a concrete method with an implementation entry point.

---

# 3. Technique catalog (dense)

## 3.1 Migration policy — the flagship RL target (Pillar A)

Build these as new classes in `module2_dt_control/policies/`, all implementing `MigrationPolicy.decide`:

1. **Value-based DRL — `DQNPolicy`.** Double-DQN / Dueling-DQN over the `feature_vector`. Action space = `{stay} ∪ 𝒢`. Cheap first win; the RL stub's `_decode` already accepts a discrete GS index.
2. **Policy-gradient — `PPOPolicy`.** PPO / SAC-discrete for stable on-policy learning; better with the load-coupled dynamics. Recommended default learner.
3. **Model-based RL / MPC — `MPCPolicy`.** Exploit *known orbital dynamics*: roll `constellation.positions(t+kΔ)` forward `H` steps, plan the migration sequence minimizing predicted latency + migration cost (receding horizon). Often beats model-free here because 90% of the dynamics is analytic — only faults/traffic need learning. **Highest expected value-per-effort.**
4. **Multi-agent RL — `QMIXPolicy` / `MADDPGPolicy`.** The correct Dec-POMDP solution. Per-DT actors, a centralized mixing critic at the NCC (CTDE). Learns load-aware coordination the greedy policy cannot (e.g. two DTs avoiding the same GS). Value-decomposition (QMIX/VDN) or MADDPG.
5. **Graph RL — `GNNPolicy` encoder.** Replace the flat `feature_vector` with a GNN over the live `NetworkGraph.G` (sats + GS + NCC, edges = ISL/GSL/terrestrial). Inductive → generalizes across con1↔con2 without retraining. Encoder feeds any of 1–4.
6. **Safe / constrained RL.** GS capacity and reliability thresholds are **hard constraints**. Use CPO / Lagrangian-PPO or action masking (mask overloaded GS — the greedy policy already does this; keep it as a safety shield around the learned policy).
7. **Offline RL bootstrap — `CQLPolicy`.** Log greedy-policy traces from Module 3 runs, pretrain with Conservative Q-Learning, then fine-tune online. De-risks cold-start and gives a safe fallback.
8. **Multi-objective RL.** Latency vs migration overhead vs load balance vs energy are competing. Learn a Pareto front (envelope Q-learning) or condition the policy on a preference vector so operators trade off at inference.

**Reward shaping already supported:** extend `environment.step`'s reward with load-variance and overload/reliability penalties — the `info` dict already carries `latencies`, `n_migrations`, per-GS loads.

## 3.2 The digital twin *as a learned model* (DT fidelity)

The paper says the DT "builds AI models from current and historical data." Right now `EdgeDT.model_state` holds scalar fields. Make the twin *predictive*:

- **Neural latency/channel surrogate.** Train a network `f(topology, positions, traffic) → PS–DT latency / link quality` to replace or accelerate the Dijkstra path cost, and to model effects the geometric simulator omits (queuing, congestion around NCC — the very thing that makes Benchmark-3 slow). Plug behind `NetworkGraph.ps_dt_latency` as an optional learned backend.
- **Temporal state models.** LSTM / Temporal-Conv / **Transformer** over each satellite's history to forecast device status, traffic, and QoE → feeds `Observation.predicted_latency` with a *learned* multi-step forecast instead of the current 1-step analytic lookahead.
- **Neural ODE / physics-informed NN.** Continuous-time twin of orbital + environmental state; respects known physics while learning residual perturbations (drag, solar pressure) the analytic propagator ignores. This is the honest "high-fidelity twin" upgrade.
- **Graph representation learning.** GNN embeddings of the constellation topology; shared by the migration policy, routing, and slicing (one encoder, many heads).
- **Generative what-if (central-DT).** VAE / diffusion model to synthesize plausible future topologies and traffic for the central-DT's parallel evaluation (the paper's "duplicate K central-DTs, deduce N snapshots" workflow becomes *learned* scenario generation).
- **Uncertainty-aware twins.** Bayesian / ensemble / conformal prediction so the DT knows *when it is stale*. High predictive uncertainty → trigger a synchronization or an irregular migration. Directly powers §3.4.

## 3.3 Slicing & resource allocation (Pillar B) → learning

- **Slice admission & bandwidth split — `DRLSlicer`.** Replace the static `DATA_TYPE_SLICE` map with a policy that allocates bandwidth across low-latency / high-bw / best-effort under load, maximizing served QoS. Contextual bandit (fast, low-risk) → DRL (full sequential).
- **Joint slicing × migration.** These interact (a migration changes which slice can meet its SLA). A hierarchical or multi-head policy that co-decides is the "pervasive AI" the paper defers.
- **Routing strategy generation (case study).** The paper evaluates *K* routing strategies in parallel central-DTs. Replace hand-picked strategies with a **GNN-RL router** that generates candidates; the central-DT's parallel snapshot evaluation becomes the RL environment rollout. Age-of-Information / deadline-aware objectives.

## 3.4 Synchronization loop → learned, semantic, event-triggered

The paper's core pain is **synchronization latency**. Make synchronization itself intelligent:

- **When to sync (event-triggered RL).** Don't sync on a fixed period; sync when predicted DT–PS divergence (from §3.2 uncertainty) exceeds a learned bound. Minimizes **Age of Information** subject to a bandwidth budget — an RL / restless-bandit problem.
- **What to sync (semantic / importance sampling).** Transmit only state that changes the downstream *decision*, not all telemetry. Ties directly to the repo's "semantic communications" theme. Learned importance weights + autoencoder compression on the PS→DT link.
- **How to sync (slice-aware).** Route sync flows through §3.3's learned slices; co-optimize with the low-latency/best-effort split.

## 3.5 Fault handling → anomaly detection (irregular migration, Pillar A)

`control_plane.inject_gs_failure / inject_sat_failure` are currently manual. Learn to *detect* faults from telemetry streams:

- Self-supervised anomaly detection (autoencoder reconstruction error, forecasting residuals) on satellite/GS state → automatically triggers irregular migration.
- Predictive maintenance: forecast link degradation before failure → *pre-emptive* migration (cheaper than reactive).

## 3.6 Representation & generalization (cross-cutting)

- **GNN state encoder** (shared, §3.1.5) — inductive across constellation sizes.
- **Meta-RL / fast adaptation** (MAML, RL²) — adapt in a few steps to a new constellation, new GS layout, or a degraded topology.
- **Curriculum & transfer** — train on con1 (66 sats), transfer to con2 (264 sats). The simulator supports both configs today; this is a ready-made curriculum.
- **Federated learning across ground stations** — each GS trains on *local* edge-DT data and shares only model updates. This is not a bolt-on: it is the *native* privacy/bandwidth-respecting fit for the paper's distributed edge-DT design (data stays at the ground station, exactly as the paper argues for reducing terrestrial overhead).

---

# 4. The elegant alignment: Hierarchical RL over the DT hierarchy

The paper's edge/central split and RL's CTDE / feudal hierarchy are the **same structure**. Exploit it explicitly:

```
            ┌──────────────────── NCC / Central-DTs ────────────────────┐
            │  HIGH-LEVEL POLICY (feudal manager)                        │
            │  goals: global load balance, slice budgets, migration      │
            │  quotas per region;  CENTRALIZED CRITIC for all edge actors│
            └───────────────▲───────────────────────────┬───────────────┘
                            │ global state (training)    │ sub-goals / budgets
            ┌───────────────┴───────────────────────────▼───────────────┐
            │  LOW-LEVEL POLICIES (edge-DTs, one per GS region)          │
            │  execute migration + local sync with LOCAL observation     │
            │  decentralized, low-latency, act every step                │
            └────────────────────────────────────────────────────────────┘
```

- **Feudal / hierarchical RL (FuN, option-critic):** central-DT sets sub-goals (e.g. "keep region load < X", "prioritize low-latency slice"); edge-DTs achieve them locally. Matches "central-DTs for global optimization, edge-DTs for real-time local operations" verbatim.
- **CTDE:** train the edge actors with a central critic that sees global state (available only at the NCC, exactly as in the architecture). At execution, edge-DTs need only local observation → respects the paper's low-latency edge requirement.

This is where SatTwin stops being "HDTN-SCN + an RL block" and becomes a *coherent learning-native redesign* whose training structure is dictated by the physical architecture.

---

# 5. End-to-end learning system

```
        ┌─────────────────────── CENTRAL-DT (NCC) ───────────────────────────┐
        │  • Generative what-if world model (VAE/diffusion) — §3.2            │
        │  • High-level HRL manager + centralized critic (CTDE) — §4          │
        │  • Global routing (GNN-RL) & slice budgets — §3.3                   │
        └───────▲──────────────────────────────────────────────┬────────────┘
   global state │ (training only)                    sub-goals   │ budgets
        ┌───────┴──────────────────────────────────────────────▼────────────┐
        │  EDGE-DTs (ground stations) — decentralized actors                  │
        │  • GNN state encoder over live topology — §3.1.5                    │
        │  • Migration policy: MPC + MARL, safety-shielded — §3.1             │
        │  • Neural surrogate + temporal/uncertainty twin — §3.2              │
        │  • Event-triggered, semantic synchronization — §3.4                 │
        │  • Anomaly detector → irregular migration — §3.5                    │
        └───────▲──────────────────────────────────────────────┬────────────┘
   telemetry    │ (semantic, importance-sampled)     migration/ │ sync actions
        ┌───────┴──────────────────────────────────────────────▼────────────┐
        │  PHYSICAL SYSTEM (Module 1 env: LEO constellation, GSL/ISL, faults) │
        └──────────────────────────────────────────────────────────────────┘

Cross-cutting: federated learning across GS (§3.6) · meta-RL adaptation ·
               multi-objective preferences · offline-RL bootstrap from greedy traces
```

---

# 6. Phased roadmap (crawl → walk → run)

**Phase 0 — instrumentation (done / near-done).** Gym-style env, greedy baseline, Fig.4/Fig.5 reproduction, `RLMigrationPolicy` interface. *Add:* a thin `gymnasium.Env` wrapper around `DTControlPlane` and a trace logger for offline RL.

**Phase 1 — single-agent migration RL.** `DQNPolicy` then `PPOPolicy` vs greedy on con1. Bootstrap with offline RL (§3.1.7) from greedy traces. **Success = lower mean PS–DT latency at equal-or-fewer migrations than greedy** on Fig.4/Fig.5 protocol.

**Phase 2 — model-based & graph.** `MPCPolicy` (known dynamics) and the `GNNPolicy` encoder. Expect the biggest single jump here (dynamics are mostly analytic).

**Phase 3 — multi-agent (CTDE).** `QMIXPolicy` / `MADDPGPolicy`; demonstrate learned load-aware coordination that beats independent learners under tight GS capacity.

**Phase 4 — hierarchical + joint control.** Feudal manager at the central-DT; co-optimize migration × slicing × sync (§4, §3.3, §3.4).

**Phase 5 — predictive twin + semantic sync.** Neural/Neural-ODE surrogate, uncertainty-triggered synchronization, semantic compression (§3.2, §3.4).

**Phase 6 — generalization.** Meta-RL / curriculum con1→con2, federated training across GS (§3.6).

---

# 7. Evaluation protocol

**Baselines (already implemented in Module 3):** greedy nearest-GS (HDTN-SCN), no-migration (B1), centralized-ISL (B2), centralized-terrestrial (B3).

**Metrics:** mean & p95 PS–DT latency (`RunResult`), migration frequency, GS load variance / overload count, Age-of-Information, sync bandwidth consumed, reliability-SLA violation rate, and constraint-violation count for safe RL.

**Protocol:** reuse `fig4_latency_vs_time.py` and `fig5_freq_vs_threshold.py`; add a `fig6_rl_vs_greedy.py` that overlays learned policies on the same axes. Train on con1, **report zero-shot transfer to con2**.

**Mandatory ablations:** flat vector vs GNN encoder · model-free vs MPC vs hybrid · independent learners vs CTDE · with/without safety shield · online-only vs offline-bootstrapped · reactive vs uncertainty-triggered sync.

---

# 8. Pitfalls & honesty checklist

- **Don't out-engineer the physics.** Orbital motion is analytic; a model-free agent that ignores this is wasteful. MPC/model-based should be the strong baseline the fancy methods must beat.
- **The greedy policy is strong.** With well-distributed ground stations it is near-optimal for pure latency (see the simulator's own results). RL must win on the *coupled* objectives (load, reliability, sync cost), not just latency — design rewards accordingly.
- **Reward hacking.** Migration cost must be real, or the agent thrashes. Keep the safety shield (action masking on overloaded GS) around every learned policy.
- **Sim fidelity gates the science.** Current latency = propagation only. Congestion/queuing (the §3.2 surrogate) must be added before latency-reduction claims are credible — the paper's Benchmark-3 gap is a *terrestrial congestion* effect the geometric model only approximates.
- **Evaluate transfer, not just training curves.** A policy overfit to con1 that fails on con2 is a negative result; report it.

---

## Objective

SatTwin turns HDTN-SCN from a set of hand-designed heuristics into a **learning-native digital-twin network** where the migration controller, the slice/resource allocator, the synchronization loop, and the twin's own predictive models are all learned — with the RL training topology dictated by the physical edge/central hierarchy. The three-module simulator in `code/HDTN-SCN/` is the environment; this README is the research program built on top of it.
