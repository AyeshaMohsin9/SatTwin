# HDTN-SCN — Modular Implementation Guide

A reference implementation plan for reproducing the paper:

> **A Hierarchical Digital Twin Network for Satellite Communication Networks (HDTN-SCN)**
> Y. Zhou, R. Zhang, J. Liu, T. Huang, Q. Tang, F. R. Yu — *IEEE Communications Magazine*, Nov. 2023.
> DOI: 10.1109/MCOM.001.2200864

This guide turns the magazine article (which is architectural, not algorithmic) into a concrete, runnable digital-twin simulation, organized into **three independent modules**. It reconstructs the paper's two headline results:

1. **PS–DT latency vs. time** — HDTN-SCN ≈ **24 ms** average vs. Benchmark-1 (no migration) ≈ 92 ms, Benchmark-2 (centralized, ISL) ≈ 90 ms, Benchmark-3 (centralized, terrestrial offload) ≈ 200 ms (Fig. 4).
2. **Migration frequency & average latency vs. latency threshold** — as threshold rises 0 → 200 ms, migration frequency falls and latency rises (Fig. 5).

Where the paper leaves a quantity unspecified (it is a magazine article), sensible defaults are given and flagged as **[assumption]** so you can tune them.

---

## The three-module design

The plan is split into three modules with **clean, one-directional dependencies** (Module 3 → Module 2 → Module 1). The split is chosen so that the **migration decision is isolated in one place** and can later be replaced by a learned (RL) policy without touching the other two modules.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MODULE 3 — Simulation, Benchmarks & Evaluation   (the harness)            │
│  discrete-time loop · 4 benchmark latency models · metrics · Fig. 4 / 5    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ drives, observes reward/metrics
┌───────────────────────────────▼──────────────────────────────────────────┐
│  MODULE 2 — Digital-Twin Control Plane   (the decision layer)  ◀── RL HERE │
│  Edge/Central DTs · MIGRATION POLICY (pluggable) · slicing · addressing    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ reads state / applies migration action
┌───────────────────────────────▼──────────────────────────────────────────┐
│  MODULE 1 — Physical & Network Substrate   (the "world" / environment)     │
│  constellation · geometry/delay · ground stations · network graph · PS–DT  │
└────────────────────────────────────────────────────────────────────────────┘
```

**Why this split matters for the RL roadmap.** Module 1 is written as a **Gym-style environment** (it can produce an *observation* at time `t`, apply a migration *action*, and score a *reward*). Module 2 consumes that observation and emits an action through a single `MigrationPolicy` interface. The default policy is the paper's greedy "nearest non-overloaded GS." Later, the AI-based migration (future-work item: *"cost-effective, seamless DT migration with an AI-based algorithm to choose the optimal migration destination"*) becomes just **one new file** — `module2_dt_control/policies/rl.py` — with **zero changes to Modules 1 and 3**.

---

## 0. What the paper actually specifies

Concrete, quotable facts every module must honor:

| Item | Value from paper |
|---|---|
| Constellation con1 | 6 orbits × 11 sats = 66 sats (Iridium-like) |
| Constellation con2 | 12 orbits × 22 sats = 264 sats |
| Ground stations | 23 geographically dispersed cities |
| NCC location | Washington, D.C. |
| Terrestrial latency source | real ping data, `https://wondernetwork.com/pings` |
| Simulation horizon (Fig. 4) | 36,000 s ≈ 6 orbital periods |
| Migration-frequency window (Fig. 5) | ~6,000 s (≈ 1 period) |
| Latency threshold sweep | 0 → 200 ms |
| HDTN-SCN avg PS–DT latency | ≈ 24 ms |
| Benchmark-1 (no migration) | ≈ 92 ms |
| Benchmark-2 (centralized, min-latency ISL) | ≈ 90 ms |
| Benchmark-3 (centralized, terrestrial offload) | ≈ 200 ms |
| PS–DT latency definition | propagation latency of routing path between PS and its DT |
| Migration trigger | PS–DT latency exceeds preset threshold → migrate to nearest non-overloaded GS |
| DT realization | **containers** for high-fidelity entities (satellites); lightweight models for terminals |

Communication classes: **PS–PS**, **PS–DT**, **DT–DT** (inter-DT + intra-DT). DT layer is hierarchical: **edge-DTs** at ground stations, **central-DTs** at the NCC.

---

## Repository layout

```
code/HDTN-SCN/
├── README.md                       # this file
├── config/
│   ├── con1.yaml                   # 6 orbits × 11 sats
│   ├── con2.yaml                   # 12 orbits × 22 sats
│   └── stations.yaml               # 23 cities + Washington NCC
├── data/
│   └── pings.csv                   # terrestrial ping matrix (GS<->GS, GS<->NCC)
│
├── module1_environment/            # ── MODULE 1: physical & network substrate ──
│   ├── constellation.py            # orbit propagation, positions(t)
│   ├── geometry.py                 # LoS, slant range, propagation delay
│   ├── ground.py                   # GroundStation, NCC, load model
│   ├── network.py                  # build ISL/GSL/terrestrial graph, route latency
│   └── environment.py              # unified state/observation/step API (RL-ready)
│
├── module2_dt_control/             # ── MODULE 2: DT control plane (RL plug-point) ──
│   ├── digital_twin.py             # EdgeDT, CentralDT, container abstraction
│   ├── migration.py                # MigrationPolicy interface + apply/seamless logic
│   ├── policies/
│   │   ├── greedy.py               # default: nearest non-overloaded GS (the paper)
│   │   └── rl.py                   # LATER: RLMigrationPolicy (learned destination)
│   ├── slicing.py                  # QoS-aware slices (Pillar B)
│   └── addressing.py               # locator-identifier mapping tables (Pillar C)
│
├── module3_simulation/             # ── MODULE 3: simulation, benchmarks, eval ──
│   ├── simulator.py                # discrete-time event loop
│   ├── benchmarks.py               # HDTN-SCN + Benchmark 1/2/3 latency models
│   ├── metrics.py                  # PS-DT latency, migration frequency, reward
│   └── experiments/
│       ├── fig4_latency_vs_time.py
│       └── fig5_freq_vs_threshold.py
└── results/
    ├── fig4.png
    └── fig5.png
```

---

## Environment & prerequisites

```bash
# Python 3.10+
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib networkx skyfield sgp4 pyyaml
# (later, for Module 2 RL policy) pip install gymnasium stable-baselines3 torch
```

| Library | Why |
|---|---|
| `sgp4` / `skyfield` | LEO orbit propagation from TLE (satellite positions over time) |
| `numpy` / `scipy` | geometry, distance, latency math |
| `networkx` | ISL/GSL/terrestrial graph + shortest-path routing (PS–DT path latency) |
| `pandas` | ping matrix, results tables |
| `matplotlib` | reproduce Fig. 4 and Fig. 5 |
| `pyyaml` | scenario config |

**[assumption]** Orbit propagation: paper says "Iridium-like." Use real Iridium TLEs (CelesTrak) for con1, or synthesize a Walker constellation with the stated geometry (6×11 and 12×22). A synthetic Walker constellation is more reproducible and is the recommended default.

---

# MODULE 1 — Physical & Network Substrate (the "world")

**Responsibility:** everything that is *not a decision* — orbital motion, geometry, the time-varying network graph, and the atomic quantity `PS–DT latency`. It exposes an **environment API** so Module 2 (and later an RL agent) can observe state and apply migration actions without knowing any physics.

**Files:** `constellation.py`, `geometry.py`, `ground.py`, `network.py`, `environment.py`.

### Step 1.1 — Propagate the constellation (`constellation.py`)

- Build a **Walker-δ** constellation: `con1 = (66 sats, 6 planes, 11/plane)`, `con2 = (264 sats, 12 planes, 22/plane)`.
- Iridium-like → altitude ≈ **780 km**, inclination ≈ **86.4°** **[assumption, matches real Iridium]**. Orbital period ≈ **6,000 s** → 36,000 s ≈ 6 periods, consistent with Fig. 4.
- Expose `positions(t) -> dict[sat_id -> (x,y,z)]`.

**Checkpoint:** plot sub-satellite tracks; confirm the period is ~6,000 s.

### Step 1.2 — Ground stations, NCC, terrestrial latency (`ground.py`)

- Place **23 GS** at real city coordinates + **NCC in Washington** (`config/stations.yaml`).
- Build the **terrestrial ping matrix** `data/pings.csv` (GS↔GS and GS↔NCC one-way latency) from wondernetwork.com/pings.
  **[assumption]** if you cannot scrape it, approximate one-way latency as `distance_km / (0.67 * c) + router_overhead` (fiber ≈ ⅔·c plus fixed per-hop overhead).

```python
class GroundStation:
    id: str; lat: float; lon: float
    load: float = 0.0          # current #edge-DTs hosted
    capacity: int = ...        # for overload check during migration
```

### Step 1.3 — Geometry & propagation delay (`geometry.py`)

Everything reduces to **propagation delay = distance / c**.

```python
C_KM_PER_S = 299_792.458          # speed of light in km/s
def prop_delay_ms(p_a, p_b):
    return dist_km(p_a, p_b) / C_KM_PER_S * 1000.0   # (km / (km/s)) = s → ×1000 = ms
def visible(sat_pos, gs_pos, min_elevation_deg=10.0) -> bool: ...
```

- **GSL** exists only when the satellite is above the GS's minimum elevation (**[assumption]** 10°).
- **ISL** exists between adjacent satellites (intra-plane neighbors + nearest cross-plane) within max ISL range.

**Checkpoint:** for a satellite directly overhead (~780 km), one-way GSL delay ≈ **2.6 ms**. (`780 / 299792.458 * 1000 ≈ 2.6`.) Unit bugs are the #1 source of wrong latency plots — verify here.

### Step 1.4 — Network graph & PS–DT path latency (`network.py`)

Build a time-varying graph `G(t)` with edge weight = propagation delay:

- Nodes: satellites, ground stations, NCC.
- Edges: ISL (sat–sat), GSL (sat–GS when visible), terrestrial (GS–GS, GS–NCC from ping matrix).
- **PS–DT latency** = `networkx.shortest_path_length(G, sat, dt_host, weight='delay')`.

This one function is the backbone of the main scheme **and** all three benchmarks (Module 3).

### Step 1.5 — Environment API (`environment.py`) — *the RL boundary*

Wrap Steps 1.1–1.4 in a Gym-style interface. This is what makes the later RL swap trivial.

```python
class HDTNEnvironment:
    def reset(self) -> "Observation": ...
    def observe(self, t) -> "Observation":
        # per edge-DT: current host, current PS-DT latency, candidate-GS latencies,
        # GS loads, (optional) predicted next-step latencies. This is the RL state.
        ...
    def step(self, actions: dict[dt_id, target_gs | None], t):
        # apply migrations, advance graph to t+dt, return (obs, reward, info)
        # reward = -(latency)  - migration_cost * (#migrations)   [tunable]
        ...
```

> **Design note:** an `Observation` must contain *everything a migration decision could need* — current/candidate latencies, GS loads, and (for RL) predicted trajectory features. Defining it now means the greedy policy uses a subset and the RL policy uses the full vector, with no API change.

---

# MODULE 2 — Digital-Twin Control Plane (the decision layer) ← RL plugs in here

**Responsibility:** the paper's three engineering pillars. The **migration decision** lives behind a single interface so it is swappable; slicing and addressing are the other two pillars.

**Files:** `digital_twin.py`, `migration.py`, `policies/greedy.py`, `policies/rl.py` (later), `slicing.py`, `addressing.py`.

### Step 2.1 — Digital-Twin objects (`digital_twin.py`)

```python
class EdgeDT:
    entity_id: str          # the satellite it mirrors
    host_gs: str            # current ground station hosting it (the "locator")
    model_state: dict       # device status, mobility, radio, QoE (Fig. 2)
    # "container" abstraction: migrate = move host_gs + copy model_state

class CentralDT:
    dt_id: str
    aggregated_topology: nx.Graph   # built from edge-DT reports
    purpose: str                    # verification | optimization | TE | slicing
```

- Each satellite maps to **exactly one** edge-DT (paper: one PS → unique edge-DT).
- Multiple central-DTs **share** edge-DT info (no duplication) — model as central-DTs reading a shared topology store, not copying per-DT.

### Step 2.2 — Migration policy interface (`migration.py`) — *the plug-point*

**This is the seam you will improve with RL.** Keep it minimal and stable.

```python
from abc import ABC, abstractmethod

class MigrationPolicy(ABC):
    @abstractmethod
    def decide(self, dt: EdgeDT, obs: "Observation", t: float) -> str | None:
        """Return target GS id to migrate `dt` to, or None to stay put."""

def apply_migration(dt, target_gs, addressing, metrics):
    """Mechanics of a migration (shared by ALL policies): update host, copy
    model_state, refresh addressing tables (Step 2.5), count the migration.
    Regular migration = seamless via DT duplication (serve from old host until
    the copy at target is ready). Irregular migration = same mechanics, triggered
    reactively by an injected fault instead of the threshold."""
```

- `apply_migration` is **policy-agnostic** — every policy reuses it. Only `decide()` changes between greedy and RL.
- **Load awareness** is enforced inside candidate generation, so no policy can migrate onto an overloaded GS.

### Step 2.3 — Default policy: greedy (`policies/greedy.py`)

The paper's algorithm — the baseline the RL policy must beat.

```python
class GreedyNearestPolicy(MigrationPolicy):
    def __init__(self, threshold_ms): self.threshold = threshold_ms
    def decide(self, dt, obs, t):
        L = obs.latency[dt.entity_id]
        if L <= self.threshold:
            return None                              # no migration
        cands = [gs for gs in obs.candidates(dt) if not obs.overloaded(gs)]
        target = min(cands, key=lambda gs: obs.cand_latency(dt, gs))
        return target if target != dt.host_gs else None
```

### Step 2.4 — LATER: RL policy (`policies/rl.py`) — *future work, do not build yet*

Same interface, learned `decide()`. Reward comes from Module 1's `step()`
(`-latency - migration_cost`). Because the interface is fixed, swapping is one line
in the experiment config: `policy = RLMigrationPolicy(model)` instead of
`GreedyNearestPolicy(threshold)`.

```python
class RLMigrationPolicy(MigrationPolicy):
    def __init__(self, model): self.model = model      # trained agent (PPO/DQN/...)
    def decide(self, dt, obs, t):
        return self.model.act(obs.feature_vector(dt))  # → target GS id or None
```

Targets for this later work: choose destination from **complex factors** (GS load, predicted trajectory, terrestrial congestion), minimize *both* latency and migration count, and support seamless/cost-aware migration. **Everything needed for it (observation, reward, action) is already defined in Modules 1–2.**

### Step 2.5 — Pillar B: QoS-aware slicing (`slicing.py`)

| Slice | Carries | Enforced property |
|---|---|---|
| Low-latency | fault-diagnosis monitoring, real-time control | minimize delay; route over optimal ISL path |
| High-bandwidth | general DT modeling data | reliability + throughput |
| Best-effort | AI-training feedback, delay-tolerant data | no guarantee |

```python
def classify(flow) -> str: ...          # by data type (Fig. 2 taxonomy)
def route(flow, G):                     # low_latency -> min-delay path (ISL-heavy)
    ...                                 # best_effort -> min-ISL / terrestrial-offload
```

Space-segment slice adapts to high dynamics/large delay/limited resources; terrestrial segment maps to existing **eMBB / uRLLC**.

### Step 2.6 — Pillar C: Locator-identifier addressing (`addressing.py`)

Decouple **identity** (object id) from **location** (network address) so migration never breaks references. Maintain the paper's tables:

1. **At each GS:** DT-identifier → PS network address.
2. **At each PS:** edge-DT identifier → GS network address (PS follows its migrating DT).
3. **At NCC:** a **PS→DT** table + a **DT locator-identifier** table.

```python
class MappingTable:
    id_to_locator: dict[str, str]
    def resolve(self, obj_id): return self.id_to_locator[obj_id]
    def update(self, obj_id, new_locator): self.id_to_locator[obj_id] = new_locator
```

`apply_migration()` (Step 2.2) must call `update()` on the relevant tables — this is what makes migration seamless.

---

# MODULE 3 — Simulation, Benchmarks & Evaluation (the harness)

**Responsibility:** drive Module 1's environment with a Module 2 policy, run the four latency models, collect metrics, and reproduce the figures.

**Files:** `simulator.py`, `benchmarks.py`, `metrics.py`, `experiments/*.py`.

### Step 3.1 — Benchmarks (`benchmarks.py`)

Four latency models sharing Module 1's path function:

- **HDTN-SCN:** edge-DT kept near via migration. `PS–DT = min over GS of sat→GS latency`. Expected avg ≈ **24 ms**.
- **Benchmark 1 (hierarchical, no migration):** edge-DT fixed at initial-nearest GS; sat reaches it over more sat links as it moves. Expected ≈ **92 ms**.
- **Benchmark 2 (centralized, ISL):** all DTs at NCC; sat→NCC over shortest ISL path. Expected ≈ **90 ms**.
- **Benchmark 3 (centralized, terrestrial offload):** sat → nearest GS → NCC over terrestrial network. Expected ≈ **200 ms**.

### Step 3.2 — Discrete-time simulator (`simulator.py`)

```python
def run(env, policy, horizon_s, dt_s=1.0):
    latencies, migrations = [], 0
    obs = env.reset()
    for t in arange(0, horizon_s, dt_s):
        actions = {dt.entity_id: policy.decide(dt, obs, t) for dt in env.edge_dts}
        obs, reward, info = env.step(actions, t)         # Module 1 applies + advances
        latencies += info["latencies"]; migrations += info["n_migrations"]
    return mean(latencies), migrations / len(env.edge_dts)
```

**[assumption]** step `dt_s = 1 s`. For 36,000 s that's 36k steps × 66 sats — fine in Python; drop to 5–10 s steps for con2 (264 sats) or vectorize with numpy.

### Step 3.3 — Metrics (`metrics.py`)

PS–DT latency (mean/percentiles), migration frequency (per edge-DT per ~6,000 s window), and the **reward** used by Module 1's `step()` — so the greedy and (future) RL policies are scored identically.

### Step 3.4 — Reproduce the figures (`experiments/`)

**Fig. 4 — PS–DT latency vs. time**
```bash
python module3_simulation/experiments/fig4_latency_vs_time.py --con con1 --horizon 36000
```
Success: HDTN-SCN mean ≈ 24 ms; B1 ≈ 92 ms; B2 ≈ 90 ms; B3 ≈ 200 ms (±20% = good magazine match).

**Fig. 5 — Migration frequency & average latency vs. threshold**
```bash
python module3_simulation/experiments/fig5_freq_vs_threshold.py --con con1 --con con2 --thresholds 0 20 40 ... 200 --window 6000
```
Success:
- threshold 0 ms → minimum latency, maximum migration frequency.
- threshold 200 ms → **zero** migrations (LEO propagation latency never exceeds 200 ms).
- con1 @ **120 ms** → ~1 migration/period → **~27%** latency reduction.
- con1 @ **80 ms** → ~3 migrations/period → **~50%** latency reduction.

---

## Validation checklist

- [ ] Overhead GSL delay ≈ 2.6 ms at 780 km (Module 1 unit check).
- [ ] Orbital period ≈ 6,000 s; 36,000 s spans ~6 periods.
- [ ] Threshold = 200 ms yields zero migrations.
- [ ] HDTN-SCN latency < Benchmark-1 < Benchmark-3; ordering matches Fig. 4.
- [ ] Increasing threshold monotonically ↓ migration frequency and ↑ latency (Fig. 5).
- [ ] con1 @120 ms ≈ 1 migration/period and ~27% reduction; @80 ms ≈ 3/period and ~50%.
- [ ] Migration updates all three addressing tables (no dangling DT references).
- [ ] Swapping `GreedyNearestPolicy` for a stub policy requires **no** change in Module 1 or Module 3.

---

## Paper → module traceability

| Paper concept | Module | File |
|---|---|---|
| Iridium-like con1/con2 | 1 | `constellation.py`, `config/*.yaml` |
| 23 GS + Washington NCC + pings | 1 | `ground.py`, `stations.yaml`, `data/pings.csv` |
| PS–PS / PS–DT / DT–DT comms, path latency | 1 | `network.py`, `geometry.py` |
| RL-ready state/action/reward boundary | 1 | `environment.py` |
| Edge-DT / Central-DT hierarchy | 2 | `digital_twin.py` |
| Regular + irregular migration (mechanics) | 2 | `migration.py` |
| Migration destination decision (greedy now, **RL later**) | 2 | `policies/greedy.py`, `policies/rl.py` |
| QoS-aware slicing (3 slice types) | 2 | `slicing.py` |
| Locator-identifier isolation (4 tables) | 2 | `addressing.py` |
| Latency threshold trigger | 2/3 | `policies/greedy.py`, `simulator.py` |
| Fig. 4 four-scheme comparison | 3 | `benchmarks.py`, `experiments/fig4_*.py` |
| Fig. 5 threshold sweep | 3 | `experiments/fig5_*.py` |

---

## Extension roadmap (paper's stated future work)

1. **AI-based migration (Module 2 only)** — implement `policies/rl.py`: replace greedy "nearest non-overloaded GS" with a learned policy (RL) choosing the optimal destination from complex factors (load, predicted trajectory, terrestrial congestion). The observation/action/reward are already defined in Modules 1–2, so **this touches one file**.
2. **TSN integration (Module 1)** — add Time-Sensitive Networking scheduling on the terrestrial segment to bound end-to-end latency for low-latency-slice data.
3. **Pervasive AI (Module 2)** — plug AI into central-DTs for the case-study workflow (K routing strategies → duplicate K central-DTs → deduce N topology snapshots → parallel QoS evaluation → select best), and for resource-allocation/routing optimization.

---

## Notes on fidelity

This is a **magazine article**, so exact numeric reproduction is not expected — the goal is qualitative and order-of-magnitude agreement (four-scheme ordering, the ~24 ms figure, the threshold trade-off shape). Every **[assumption]** is a knob; record chosen values in `config/*.yaml` for reproducibility. If you have real Iridium TLEs and the actual wondernetwork ping matrix, drop them in `data/` for closer fidelity.

---

### Quick start

```bash
cd code/HDTN-SCN
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib networkx skyfield sgp4 pyyaml
# implement module1_environment/ → module2_dt_control/ → module3_simulation/, then:
python module3_simulation/experiments/fig4_latency_vs_time.py --con con1
python module3_simulation/experiments/fig5_freq_vs_threshold.py --con con1 --con con2
# outputs -> results/fig4.png, results/fig5.png
```
