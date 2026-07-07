# Module 4 — multi-objective reward with Lagrangian capacity penalty and per-agent shaping.
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardConfig:
    w_lat: float = 1.0
    w_mig: float = 1.0
    w_load: float = 0.5
    w_aoi: float = 0.01
    w_shaping: float = 0.1
    beta: float = 0.0
    latency_scale: float = 100.0
    objective: str = "latency"
    w_sumrate: float = 1.0
    w_minrate: float = 1.0
    w_jain: float = 1.0
    rate_scale: float = 100.0
    w_queue: float = 0.0
    w_drop: float = 0.0
    w_battery: float = 0.0
    queue_scale: float = 30.0


@dataclass
class RewardBreakdown:
    latency: float = 0.0
    migration: float = 0.0
    load: float = 0.0
    aoi: float = 0.0
    overload: float = 0.0
    total: float = 0.0
    per_agent: dict = field(default_factory=dict)
    raw_mean_latency: float = 0.0
    raw_overload: float = 0.0
    raw_migrations: int = 0
    raw_load_var: float = 0.0
    sumrate: float = 0.0
    minrate: float = 0.0
    jain: float = 0.0
    raw_sum_rate: float = 0.0
    raw_min_rate: float = 0.0
    raw_jain: float = 0.0
    queue: float = 0.0
    drop: float = 0.0
    battery: float = 0.0
    raw_queue: float = 0.0
    raw_dropped: float = 0.0
    raw_flat_battery: int = 0


class RewardFunction:
    def __init__(self, cfg, obs_builder):
        self.cfg = cfg
        self.ob = obs_builder

    def load_variance(self, obs):
        loads = [obs.gs_load.get(g, 0) / max(1, obs.gs_capacity.get(g, 1))
                 for g in self.ob.gs_ids]
        return float(np.var(loads))

    def overload_violation(self, obs):
        v = 0
        for g in self.ob.gs_ids:
            over = obs.gs_load.get(g, 0) - obs.gs_capacity.get(g, 1)
            if over > 0:
                v += over
        return float(v)

    def mean_aoi(self, obs):
        ages = [obs.t - self.ob.last_migration_t.get(s, 0.0) for s in self.ob.sat_ids]
        return float(np.mean(ages)) if ages else 0.0

    def compute(self, obs, info, prev_latency=None):
        c = self.cfg
        finite = [x for x in info["latencies"] if x != float("inf")]
        mean_lat = float(np.mean(finite)) if finite else c.latency_scale
        n_mig = info["n_migrations"]
        lvar = self.load_variance(obs)
        aoi = self.mean_aoi(obs)
        overload = self.overload_violation(obs)
        b = RewardBreakdown()
        b.migration = -c.w_mig * (n_mig / max(1, len(self.ob.sat_ids)))
        b.overload = -c.beta * overload
        b.raw_mean_latency = mean_lat
        b.raw_overload = overload
        b.raw_migrations = n_mig
        b.raw_load_var = lvar
        b.raw_sum_rate = float(info.get("sum_rate", 0.0))
        b.raw_min_rate = float(info.get("min_rate", 0.0))
        b.raw_jain = float(info.get("jain", 0.0))
        b.raw_queue = float(info.get("mean_queue", 0.0))
        b.raw_dropped = float(info.get("dropped", 0.0))
        b.raw_flat_battery = int(info.get("flat_battery", 0))
        if c.objective == "rate":
            b.sumrate = c.w_sumrate * (b.raw_sum_rate / c.rate_scale)
            b.minrate = c.w_minrate * b.raw_min_rate
            b.jain = c.w_jain * b.raw_jain
            b.queue = -c.w_queue * (b.raw_queue / max(1e-6, c.queue_scale))
            b.drop = -c.w_drop * (b.raw_dropped / max(1, len(self.ob.sat_ids)))
            b.battery = -c.w_battery * (b.raw_flat_battery / max(1, len(self.ob.sat_ids)))
            b.total = (b.sumrate + b.minrate + b.jain + b.migration + b.overload
                       + b.queue + b.drop + b.battery)
            b.per_agent = self._rate_shaping(info)
        else:
            b.latency = -c.w_lat * (mean_lat / c.latency_scale)
            b.load = -c.w_load * lvar
            b.aoi = -c.w_aoi * (aoi / c.latency_scale)
            b.total = b.latency + b.migration + b.load + b.aoi + b.overload
            b.per_agent = self._shaping(obs, prev_latency)
        return b

    def _rate_shaping(self, info):
        c = self.cfg
        rates = info.get("rates", [])
        out = {}
        for s, r in zip(self.ob.sat_ids, rates):
            out[s] = c.w_shaping * (r / c.rate_scale)
        return out

    def _shaping(self, obs, prev_latency):
        c = self.cfg
        out = {}
        for s in self.ob.sat_ids:
            cur = obs.latency.get(s, float("inf"))
            cur = c.latency_scale if cur == float("inf") else cur
            if prev_latency is None:
                out[s] = 0.0
            else:
                pv = prev_latency.get(s, cur)
                pv = c.latency_scale if pv == float("inf") else pv
                out[s] = c.w_shaping * ((pv - cur) / c.latency_scale)
        return out

    def team_plus_shaping(self, obs, info, prev_latency=None):
        b = self.compute(obs, info, prev_latency)
        rewards = {s: b.total + b.per_agent.get(s, 0.0) for s in self.ob.sat_ids}
        return rewards, b
