# Module 2 — margin/hysteresis greedy: migrate only if a candidate beats current by margin.
from ..migration import MigrationPolicy


class HysteresisPolicy(MigrationPolicy):
    def __init__(self, margin_ms=20.0):
        self.margin = margin_ms

    def decide(self, dt, obs, t):
        sid = dt.entity_id
        cur = obs.latency.get(sid, float("inf"))
        best_gs, best_lat = None, cur
        for gs in obs.candidates(sid):
            if obs.overloaded(gs):
                continue
            lat = obs.candidate_latency(sid, gs)
            if lat < best_lat:
                best_gs, best_lat = gs, lat
        if best_gs is None or best_gs == dt.host_gs:
            return None
        if cur == float("inf"):
            return best_gs
        if (cur - best_lat) >= self.margin:
            return best_gs
        return None
