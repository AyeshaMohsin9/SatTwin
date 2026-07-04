# Module 1 — per-edge-DT observation container exposing the RL state/action interface.
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    t: float
    host: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    cand_latency: dict = field(default_factory=dict)
    gs_load: dict = field(default_factory=dict)
    gs_capacity: dict = field(default_factory=dict)
    predicted_latency: dict = field(default_factory=dict)
    edge_station_ids: list = field(default_factory=list)

    def candidates(self, dt_id):
        return list(self.cand_latency[dt_id].keys())

    def overloaded(self, gs_id):
        return self.gs_load[gs_id] >= self.gs_capacity[gs_id]

    def candidate_latency(self, dt_id, gs_id):
        return self.cand_latency[dt_id].get(gs_id, float("inf"))

    def feature_vector(self, dt_id):
        cur = self.latency[dt_id]
        cands = self.cand_latency[dt_id]
        lat_vec = [cands.get(g, float("inf")) for g in self.edge_station_ids]
        load_vec = [self.gs_load[g] / max(1, self.gs_capacity[g])
                    for g in self.edge_station_ids]
        pred = self.predicted_latency.get(dt_id, cur)
        vec = [cur, pred] + lat_vec + load_vec
        arr = np.asarray(vec, dtype=float)
        arr[np.isinf(arr)] = 1e6
        return arr
