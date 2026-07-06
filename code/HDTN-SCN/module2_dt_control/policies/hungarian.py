# Module 2 — per-step optimal capacity-constrained assignment via min-cost bipartite matching.
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..migration import MigrationPolicy

BIG = 1e6


class HungarianPolicy(MigrationPolicy):
    def __init__(self, gs_ids):
        self.gs_ids = gs_ids
        self._t = None
        self._assign = {}

    def _solve(self, obs):
        sats = list(obs.latency.keys())
        slots, slot_gs = [], []
        for g in self.gs_ids:
            cap = max(1, obs.gs_capacity.get(g, 1))
            for _ in range(cap):
                slots.append(g)
                slot_gs.append(g)
        n_sat, n_slot = len(sats), len(slots)
        dim = max(n_sat, n_slot)
        cost = np.full((dim, dim), BIG, dtype=np.float64)
        for i, s in enumerate(sats):
            cand = obs.cand_latency.get(s, {})
            for j, g in enumerate(slots):
                lat = cand.get(g, float("inf"))
                cost[i, j] = BIG if lat == float("inf") else lat
        rows, cols = linear_sum_assignment(cost)
        assign = {}
        for i, j in zip(rows, cols):
            if i < n_sat and j < n_slot and cost[i, j] < BIG:
                assign[sats[i]] = slots[j]
        return assign

    def decide(self, dt, obs, t):
        if self._t != t:
            self._assign = self._solve(obs)
            self._t = t
        sid = dt.entity_id
        target = self._assign.get(sid)
        if target is None or target == dt.host_gs:
            return None
        return target
