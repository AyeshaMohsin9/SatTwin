# Module 2 — default migration policy: threshold-triggered lowest-effective-latency GS.
from ..migration import MigrationPolicy


class GreedyNearestPolicy(MigrationPolicy):
    def __init__(self, threshold_ms, soft_capacity=True):
        self.threshold = threshold_ms
        self.soft_capacity = soft_capacity

    def decide(self, dt, obs, t):
        cur = obs.latency[dt.entity_id]
        if cur <= self.threshold:
            return None
        best_gs, best_lat = None, cur
        for gs in obs.candidates(dt.entity_id):
            if not self.soft_capacity and obs.overloaded(gs):
                continue
            lat = obs.candidate_latency(dt.entity_id, gs)
            if lat < best_lat:
                best_gs, best_lat = gs, lat
        if best_gs is None or best_gs == dt.host_gs:
            return None
        return best_gs
