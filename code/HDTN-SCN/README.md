# HDTN-SCN — Step-by-Step Implementation Guide

A reference implementation plan for reproducing the paper:

> **A Hierarchical Digital Twin Network for Satellite Communication Networks (HDTN-SCN)**
> Y. Zhou, R. Zhang, J. Liu, T. Huang, Q. Tang, F. R. Yu — *IEEE Communications Magazine*, Nov. 2023.
> DOI: 10.1109/MCOM.001.2200864

This guide turns the magazine article (which is architectural, not algorithmic) into a concrete, runnable digital-twin simulation. It reconstructs the paper's two headline results:

1. **PS–DT latency vs. time** — HDTN-SCN ≈ **24 ms** average vs. Benchmark-1 (no migration) ≈ 92 ms, Benchmark-2 (centralized, ISL) ≈ 90 ms, Benchmark-3 (centralized, terrestrial offload) ≈ 200 ms (Fig. 4).
2. **Migration frequency & average latency vs. latency threshold** — as threshold rises 0 → 200 ms, migration frequency falls and latency rises (Fig. 5).

Everything below is derived from the paper. Where the paper leaves a quantity unspecified (it is a magazine article), sensible defaults are given and flagged as **[assumption]** so you can tune them.

---

## 0. What the paper actually specifies

Before coding, pin down the concrete, quotable facts you must honor:

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

The system is split into three communication classes: **PS–PS**, **PS–DT**, **DT–DT** (further split into inter-DT and intra-DT). The DT layer is hierarchical: **edge-DTs** at ground stations, **central-DTs** at the NCC.

---

## 1. Architecture recap (what you are building)

```
                    ┌───────────────────────────── NCC (Washington) ─────────────────────────┐
                    │  Global SDN controller        Central-DTs (K isolated personalized DTs) │
                    │  - global topology            - verification / global optimization       │
                    │  - traffic engineering        - traffic engineering / slicing            │
                    │  PS-DT mapping table + DT locator-identifier table                       │
                    └───────▲───────────────────────────────────────────────────▲─────────────┘
                            │ terrestrial network (real ping latencies)          │
              ┌─────────────┴───────────┐                          ┌─────────────┴───────────┐
              │  Ground Station GS_i     │   ... 23 stations ...    │  Ground Station GS_j     │
              │  Local SDN controller    │                          │  Local SDN controller    │
              │  Edge-DTs (nearby sats)  │                          │  Edge-DTs (nearby sats)  │
              │  locator-identifier map  │                          │  locator-identifier map  │
              └─────────────▲───────────┘                          └─────────────▲───────────┘
                            │ GSL                                                 │ GSL
        ┌───────────────────┴─────────────────────  Physical System  ────────────┴──────────────┐
        │           LEO satellites  ──ISL──  LEO satellites  ──ISL── ...   +   terminals          │
        └──────────────────────────────────────────────────────────────────────────────────────┘
```

Three engineering pillars from the paper you must implement:

- **Pillar A — Dynamic DT migration** (regular + irregular) to keep each edge-DT on a GS close to its satellite.
- **Pillar B — QoS-aware synchronization slicing** (low-latency / high-bandwidth / best-effort slices).
- **Pillar C — Locator-identifier-isolation addressing** (mapping tables so identity ≠ location, enabling seamless migration).

---

## 2. Prerequisites & environment

```bash
# Python 3.10+
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib networkx skyfield sgp4 pyyaml
```

| Library | Why |
|---|---|
| `sgp4` / `skyfield` | LEO orbit propagation from TLE (satellite positions over time) |
| `numpy` / `scipy` | geometry, distance, latency math |
| `networkx` | ISL/GSL/terrestrial graph + shortest-path routing (PS–DT path latency) |
| `pandas` | ping matrix, results tables |
| `matplotlib` | reproduce Fig. 4 and Fig. 5 |
| `pyyaml` | scenario config |

**[assumption]** Orbit propagation: the paper says "Iridium-like." Use real Iridium TLEs (from CelesTrak) for con1, or synthesize a Walker constellation with the stated geometry (6×11 and 12×22). A synthetic Walker constellation is more reproducible and is the recommended default.

---

## 3. Repository layout to create

```
code/HDTN-SCN/
├── README.md                  # this file
├── config/
│   ├── con1.yaml              # 6 orbits × 11 sats
│   ├── con2.yaml              # 12 orbits × 22 sats
│   └── stations.yaml          # 23 cities + Washington NCC
├── data/
│   └── pings.csv              # terrestrial ping matrix (GS<->GS, GS<->NCC)
├── src/
│   ├── constellation.py       # orbit propagation, positions(t)
│   ├── geometry.py            # LoS, slant range, propagation delay
│   ├── network.py             # build ISL/GSL/terrestrial graph, route latency
│   ├── ground.py             # GroundStation, NCC, load model
│   ├── digital_twin.py       # EdgeDT, CentralDT, container abstraction
│   ├── migration.py          # regular + irregular migration (Pillar A)
│   ├── slicing.py            # QoS-aware slices (Pillar B)
│   ├── addressing.py         # locator-identifier mapping tables (Pillar C)
│   ├── benchmarks.py         # HDTN-SCN + Benchmark 1/2/3 latency models
│   ├── simulator.py          # discrete-time event loop
│   └── metrics.py            # PS-DT latency, migration frequency
├── experiments/
│   ├── fig4_latency_vs_time.py
│   └── fig5_freq_vs_threshold.py
└── results/
    ├── fig4.png
    └── fig5.png
```

---

## 4. Step-by-step implementation

### Step 1 — Physical System: propagate the constellation

Model the LEO constellation and return each satellite's ECEF position at time `t`.

- Build a **Walker-δ** constellation matching the paper: `con1 = (66 sats, 6 planes, 11/plane)`, `con2 = (264 sats, 12 planes, 22/plane)`.
- Iridium-like → altitude ≈ **780 km**, inclination ≈ **86.4°** **[assumption, matches real Iridium]**. Orbital period ≈ **6,000 s** → 36,000 s ≈ 6 periods, consistent with Fig. 4.
- Expose `positions(t) -> dict[sat_id -> (x,y,z)]`.

```python
# src/constellation.py  (sketch)
# Use sgp4 with real TLEs, OR synthesize a Walker constellation:
#   plane RAAN spread = 360/planes, in-plane phase = 360/sats_per_plane,
#   inter-plane phasing factor F for Walker-delta.
# Propagate circular orbits analytically (mean motion) for reproducibility.
```

**Checkpoint:** plot sub-satellite tracks; confirm the period is ~6,000 s.

### Step 2 — Ground stations, NCC, and the terrestrial latency matrix

- Place **23 GS** at real city coordinates + **NCC in Washington** (`config/stations.yaml`).
- Build the **terrestrial ping matrix** `data/pings.csv` (GS↔GS and GS↔NCC one-way latency). Source: wondernetwork.com/pings as the paper does. **[assumption]** if you cannot scrape it, approximate terrestrial one-way latency as `0.5 * (distance_km / (0.67 * c)) + router_overhead`, i.e. speed-of-light in fiber (~2/3 c) plus a fixed per-hop overhead.

```python
# src/ground.py
class GroundStation:
    id: str; lat: float; lon: float
    load: float = 0.0          # current #edge-DTs hosted / capacity
    capacity: int = ...        # for overload check during migration
```

### Step 3 — Geometry & propagation delay (the atomic latency unit)

Everything reduces to **propagation delay = distance / c**.

```python
# src/geometry.py
C = 299_792.458  # km/ms  -> actually km per ms = 299.792458; keep units consistent!
def prop_delay_ms(p_a, p_b):
    return dist_km(p_a, p_b) / 299_792.458 * 1000  # ms
def visible(sat_pos, gs_pos, min_elevation_deg=10.0) -> bool: ...
```

- **GSL** exists only when the satellite is above the GS's minimum elevation (**[assumption]** 10°).
- **ISL** exists between adjacent satellites (intra-plane neighbors + nearest cross-plane) within max ISL range.

**Checkpoint:** for a satellite directly overhead (~780 km), one-way GSL delay ≈ 2.6 ms. Sanity-check your units here — unit bugs are the #1 source of wrong latency plots.

### Step 4 — Network graph & PS–DT path latency

Build a time-varying graph `G(t)` with weighted edges = propagation delay:

- Nodes: satellites, ground stations, NCC.
- Edges: ISL (sat–sat), GSL (sat–GS when visible), terrestrial (GS–GS, GS–NCC from ping matrix).
- **PS–DT latency** = shortest-path propagation latency between a satellite (PS) and the node hosting its DT (a GS for edge-DT, NCC for central-DT). Use `networkx.shortest_path_length(G, sat, dt_host, weight='delay')`.

This single function is the backbone of both the main scheme and all three benchmarks.

### Step 5 — Digital Twin objects (Pillar: DT realization)

```python
# src/digital_twin.py
class EdgeDT:
    entity_id: str          # the satellite it mirrors
    host_gs: str            # current ground station hosting it (the "locator")
    model_state: dict       # device status, mobility, radio, QoE  (Fig. 2)
    # "container" abstraction: migrate = move host_gs + copy model_state

class CentralDT:
    dt_id: str
    aggregated_topology: nx.Graph   # built from edge-DT reports
    purpose: str                    # verification | optimization | TE | slicing
```

- Each satellite maps to **exactly one** edge-DT (paper: one PS → unique edge-DT).
- Multiple central-DTs **share** edge-DT info (no duplication) — model this by having central-DTs read from a shared topology store, not copy per-DT.

### Step 6 — Pillar A: Dynamic DT migration

The core algorithm behind Fig. 4 and Fig. 5.

```
for each time step t:
    for each edge-DT dt:
        L = ps_dt_latency(dt.entity, dt.host_gs, t)       # Step 4
        if L > THRESHOLD:                                  # migration trigger
            candidates = [gs for gs in stations
                          if not overloaded(gs)]           # load-aware
            target = argmin_gs ps_dt_latency(dt.entity, gs, t)
            if target != dt.host_gs:
                migrate(dt, target)                        # regular migration
                addressing.update_mapping(dt, target)      # Pillar C
                metrics.count_migration(dt.entity)
```

Implement both flavors from the paper:

- **Regular migration** — pre-planned from predictable orbital motion. Precompute, per satellite, the schedule of "which GS is nearest at time t" and migrate at those timestamps. Improve with **DT duplication**: keep serving from the old host until the copy at the new host is ready → *seamless* to the PS.
- **Irregular migration** — reactive. On an injected fault (satellite failure → huge latency, or GS failure → DT unavailable), the NCC detects it, plans a new migration, reconfigures affected nodes, updates global topology, and recomputes the regular plan for affected DTs.

**Load awareness:** never migrate onto an overloaded GS (track `GroundStation.load` vs `capacity`).

### Step 7 — Pillar B: QoS-aware synchronization slicing

Classify every synchronization flow into one of three slices and enforce differentiated transport:

| Slice | Carries | Enforced property |
|---|---|---|
| Low-latency | fault-diagnosis monitoring, real-time control instructions | minimize delay; route over optimal ISL path |
| High-bandwidth | general DT modeling data | reliability + throughput |
| Best-effort | AI-training feedback, delay-tolerant optimization data | no guarantee |

```python
# src/slicing.py
SLICES = {"low_latency": ..., "high_bw": ..., "best_effort": ...}
def classify(flow) -> str: ...      # by data type (Fig. 2 taxonomy)
def route(flow, G):                 # low_latency -> min-delay path (ISL-heavy)
    ...                             # best_effort -> min-ISL / terrestrial-offload path
```

The space segment slice must adapt to high dynamics/large delay/limited resources; the terrestrial segment maps to existing **eMBB / uRLLC** slices.

### Step 8 — Pillar C: Locator-identifier-isolation addressing

Decouple **identity** (object identifier) from **location** (network address) so migration doesn't break references.

Maintain the mapping tables the paper enumerates:

1. **At each GS:** DT-identifier → PS network address (so a DT can locate its physical entity).
2. **At each PS:** edge-DT identifier → GS network address (so the PS follows its migrating DT).
3. **At NCC:** a **PS→DT** mapping table + a **DT locator-identifier** table (so central apps locate a DT from a PS identity).

```python
# src/addressing.py
class MappingTable:                 # one per (GS, PS, NCC scope)
    id_to_locator: dict[str, str]
    def resolve(self, obj_id): return self.id_to_locator[obj_id]
    def update(self, obj_id, new_locator): self.id_to_locator[obj_id] = new_locator
```

Every `migrate()` in Step 6 must call `update()` on the relevant tables — this is what makes migration seamless.

### Step 9 — Benchmarks (needed to reproduce Fig. 4)

Implement four latency models sharing Step-4's path function:

- **HDTN-SCN:** edge-DT always on a near GS (via migration). `PS–DT = min over GS of sat→GS latency` (edge-DT tracks the nearest GS). Expected avg ≈ **24 ms**.
- **Benchmark 1 (hierarchical, no migration):** edge-DT fixed at the initially-nearest GS forever; as the sat moves away it reaches its DT over more satellite links. Expected ≈ **92 ms**.
- **Benchmark 2 (centralized, ISL):** all DTs at NCC; sat→NCC over shortest **ISL** path. Expected ≈ **90 ms**.
- **Benchmark 3 (centralized, terrestrial offload):** sat → nearest GS (GSL/ISL) → NCC over **terrestrial** network. Expected ≈ **200 ms** (dominated by terrestrial latency).

### Step 10 — Discrete-time simulator

```python
# src/simulator.py
def run(scenario, scheme, threshold, horizon_s, dt_s=1.0):
    latencies, migrations = [], 0
    for t in arange(0, horizon_s, dt_s):
        G = build_graph(constellation.positions(t), stations, pings)   # Step 4
        for dt in edge_dts:
            if scheme == "HDTN-SCN":
                migrations += maybe_migrate(dt, G, threshold, t)       # Step 6
            L = ps_dt_latency_for_scheme(scheme, dt, G, t)             # Step 9
            latencies.append(L)
    return mean(latencies), migrations / len(edge_dts)
```

**[assumption]** step `dt_s = 1 s`. For 36,000 s that is 36k steps × 66 sats — fine in Python; drop to 5–10 s steps for con2 (264 sats) if slow, or vectorize with numpy.

---

## 5. Reproducing the figures

### Fig. 4 — PS–DT latency vs. time

```bash
python experiments/fig4_latency_vs_time.py --con con1 --horizon 36000
```

- Run all four schemes over 36,000 s; plot per-timestep (or moving-average) latency.
- **Success criterion:** HDTN-SCN mean ≈ 24 ms; B1 ≈ 92 ms; B2 ≈ 90 ms; B3 ≈ 200 ms (±20% is a good magazine-level match).

### Fig. 5 — Migration frequency & average latency vs. threshold

```bash
python experiments/fig5_freq_vs_threshold.py --con con1 --con con2 --thresholds 0 20 40 ... 200 --window 6000
```

- Sweep threshold 0 → 200 ms; for each, record avg migrations per edge-DT per ~6,000 s **and** avg PS–DT latency.
- **Success criteria from the paper:**
  - threshold 0 ms → minimum latency, maximum migration frequency.
  - threshold 200 ms → **zero** migrations (LEO propagation latency never exceeds 200 ms).
  - con1 @ **120 ms** → migrate ~once/period → **~27%** latency reduction.
  - con1 @ **80 ms** → migrate ~3×/period → **~50%** latency reduction.

---

## 6. Validation checklist

- [ ] Overhead GSL delay ≈ 2.6 ms at 780 km (unit check).
- [ ] Orbital period ≈ 6,000 s; 36,000 s spans ~6 periods.
- [ ] Threshold = 200 ms yields zero migrations.
- [ ] HDTN-SCN latency < Benchmark-1 < Benchmark-3; ordering matches Fig. 4.
- [ ] Increasing threshold monotonically ↓ migration frequency and ↑ latency (Fig. 5).
- [ ] con1 @120 ms ≈ 1 migration/period and ~27% reduction; @80 ms ≈ 3/period and ~50%.
- [ ] Migration updates all three addressing tables (no dangling DT references).

---

## 7. Mapping paper → code (traceability)

| Paper concept | Section | Module |
|---|---|---|
| Edge-DT / Central-DT hierarchy | Architecture | `digital_twin.py` |
| PS–PS / PS–DT / DT–DT comms | Key components | `network.py` |
| Regular + irregular migration | Implementation issues → DT migration | `migration.py` |
| Latency threshold trigger | DT migration | `simulator.py`, `metrics.py` |
| QoS-aware slicing (3 slice types) | Reliable PS-DT synchronization | `slicing.py` |
| Locator-identifier isolation (4 tables) | Addressing in DT system | `addressing.py` |
| Iridium-like con1/con2 | Simulation settings | `config/*.yaml`, `constellation.py` |
| 23 GS + Washington NCC + pings | Simulation settings | `stations.yaml`, `data/pings.csv` |
| Fig. 4 four-scheme comparison | PS-DT latency performance | `benchmarks.py`, `experiments/fig4_*.py` |
| Fig. 5 threshold sweep | Influence of latency threshold | `experiments/fig5_*.py` |

---

## 8. Extension roadmap (paper's stated future work)

1. **AI-based migration** — replace the greedy "nearest non-overloaded GS" with a learned policy (RL) choosing the optimal migration destination from complex factors (load, predicted trajectory, terrestrial congestion).
2. **TSN integration** — add Time-Sensitive Networking scheduling on the terrestrial segment to bound end-to-end latency for high-priority (low-latency slice) synchronization data.
3. **Pervasive AI** — plug AI models into central-DTs for the case-study workflow (K routing strategies → duplicate K central-DTs → deduce N topology snapshots → parallel QoS evaluation → select best), and for resource allocation/routing optimization.

---

## 9. Notes on fidelity

This is a **magazine article**, so exact numeric reproduction is not expected — the goal is qualitative and order-of-magnitude agreement (the four-scheme ordering, the ~24 ms figure, the threshold trade-off shape). Every **[assumption]** above is a knob; document the values you pick in `config/*.yaml` so results are reproducible. If you have real Iridium TLEs and the actual wondernetwork ping matrix, drop them in `data/` for closer fidelity.

---

### Quick start

```bash
cd code/HDTN-SCN
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib networkx skyfield sgp4 pyyaml
# implement src/ per Steps 1–10, then:
python experiments/fig4_latency_vs_time.py --con con1
python experiments/fig5_freq_vs_threshold.py --con con1 --con con2
# outputs -> results/fig4.png, results/fig5.png
```
