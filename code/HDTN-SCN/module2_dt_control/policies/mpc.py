# Module 2 — standalone MPC policy: pick the horizon-optimal action from the lookahead engine.
from ..migration import MigrationPolicy


class MPCPolicy(MigrationPolicy):
    def __init__(self, mpc_engine, gs_ids, core):
        self.mpc = mpc_engine
        self.gs_ids = gs_ids
        self.core = core

    def decide(self, dt, obs, t):
        sid = dt.entity_id
        action = self.mpc.best_action(sid, self.core, self.gs_ids)
        if action == 0:
            return None
        target = self.gs_ids[action - 1]
        if obs.overloaded(target) or target == dt.host_gs:
            return None
        return target
