# Module 2 — random-feasible policy: sanity floor picking a random non-overloaded candidate.
import random

from ..migration import MigrationPolicy


class RandomFeasiblePolicy(MigrationPolicy):
    def __init__(self, migrate_prob=0.3, seed=0):
        self.migrate_prob = migrate_prob
        self.rng = random.Random(seed)

    def decide(self, dt, obs, t):
        if self.rng.random() > self.migrate_prob:
            return None
        sid = dt.entity_id
        feasible = [g for g in obs.candidates(sid)
                    if not obs.overloaded(g) and g != dt.host_gs]
        if not feasible:
            return None
        return self.rng.choice(feasible)
