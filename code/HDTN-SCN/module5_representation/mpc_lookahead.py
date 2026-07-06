# Module 5 — analytic H-step orbital roll-forward giving per-action latency previews (MPC).
import numpy as np


class MPCLookahead:
    def __init__(self, horizon=5, dt_s=None, discount=0.9, migration_cost=1.0,
                 latency_scale=100.0):
        self.horizon = horizon
        self.dt_s = dt_s
        self.discount = discount
        self.migration_cost = migration_cost
        self.latency_scale = latency_scale
        self._cache_key = None
        self._cache = None

    def feature_dim(self, n_gs):
        return 2 * (n_gs + 1) + 2

    def _dt(self, core):
        return self.dt_s or core.cfg.time_step_s

    def _ensure_cache(self, core, gs_ids):
        key = (id(core), round(core.t, 6), self.horizon, tuple(gs_ids))
        if self._cache_key == key:
            return self._cache
        t0 = core.t
        dt = self._dt(core)
        horizon_lat = []
        for k in range(self.horizon):
            core.net.build(t0 + k * dt)
            step = {s: core.net.latencies_from_sat(s, gs_ids)
                    for s in core.constellation.sat_ids}
            horizon_lat.append(step)
        core.net.build(t0)
        reach = {s: core.net.latencies_from_sat(s, gs_ids)
                 for s in core.constellation.sat_ids}
        core.net.build(t0)
        self._cache_key = key
        self._cache = (horizon_lat, reach)
        return self._cache

    def action_costs(self, sat_id, core, gs_ids):
        horizon_lat, reach = self._ensure_cache(core, gs_ids)
        cap = self.latency_scale
        host = core.edge_dt_host.get(sat_id)
        stay_acc, disc, total_disc = 0.0, 1.0, 0.0
        gs_acc = np.zeros(len(gs_ids), dtype=np.float64)
        for step in horizon_lat:
            lat = step[sat_id]
            for i, g in enumerate(gs_ids):
                v = lat[g]
                gs_acc[i] += disc * (cap if v == float("inf") else v)
            if host is not None:
                hv = lat.get(host, float("inf"))
                stay_acc += disc * (cap if hv == float("inf") else hv)
            else:
                stay_acc += disc * cap
            total_disc += disc
            disc *= self.discount
        costs = np.full(len(gs_ids) + 1, cap * total_disc, dtype=np.float64)
        costs[0] = stay_acc
        r = reach[sat_id]
        for i, g in enumerate(gs_ids):
            if r[g] != float("inf"):
                costs[i + 1] = gs_acc[i] + self.migration_cost
        return costs

    def best_action(self, sat_id, core, gs_ids):
        return int(np.argmin(self.action_costs(sat_id, core, gs_ids)))

    def _total_disc(self):
        d, acc = 1.0, 0.0
        for _ in range(self.horizon):
            acc += d
            d *= self.discount
        return acc

    def feature(self, sat_id, core, gs_ids):
        costs = self.action_costs(sat_id, core, gs_ids)
        scale = self.latency_scale * self._total_disc()
        norm = costs / scale
        best = np.argmin(costs)
        onehot = np.zeros(len(costs), dtype=np.float64)
        onehot[best] = 1.0
        stay_gain = (costs[0] - costs.min()) / scale
        margin = (np.partition(costs, 1)[1] - costs.min()) / scale
        return np.concatenate([norm, onehot, [stay_gain, margin]]).astype(np.float32)
