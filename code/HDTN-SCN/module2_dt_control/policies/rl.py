# Module 2 — RL migration policy stub (future work); same interface as GreedyNearestPolicy.
from ..migration import MigrationPolicy


class RLMigrationPolicy(MigrationPolicy):
    def __init__(self, model=None, threshold_ms=200.0):
        self.model = model
        self.threshold = threshold_ms

    def decide(self, dt, obs, t):
        if self.model is None:
            cur = obs.latency[dt.entity_id]
            if cur <= self.threshold:
                return None
            cands = [g for g in obs.candidates(dt.entity_id) if not obs.overloaded(g)]
            if not cands:
                return None
            best = min(cands, key=lambda g: obs.candidate_latency(dt.entity_id, g))
            return best if best != dt.host_gs else None
        feat = obs.feature_vector(dt.entity_id)
        action = self.model.act(feat)
        return self._decode(action, dt, obs)

    def _decode(self, action, dt, obs):
        if action is None or action < 0 or action >= len(obs.edge_station_ids):
            return None
        target = obs.edge_station_ids[action]
        if obs.overloaded(target) or target == dt.host_gs:
            return None
        return target
