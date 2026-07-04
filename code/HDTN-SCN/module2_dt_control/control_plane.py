# Module 2 — DT control plane orchestrating DTs, migration, slicing, addressing over Module 1.
from .digital_twin import EdgeDT, SharedTopologyStore, spawn_central_dts
from .addressing import AddressingSystem
from .slicing import SlicingManager
from .migration import MigrationExecutor

CENTRAL_PURPOSES = ["verification", "optimization", "traffic_engineering", "slicing"]


class DTControlPlane:
    def __init__(self, env, policy, seamless=True,
                 central_purposes=CENTRAL_PURPOSES):
        self.env = env
        self.policy = policy
        self.addressing = AddressingSystem()
        self.slicing = SlicingManager(env.net)
        self.executor = MigrationExecutor(env, self.addressing, seamless=seamless)
        self.store = SharedTopologyStore()
        self.central_dts = spawn_central_dts(central_purposes, self.store)
        self.edge_dts = {}
        self.failed_gs = set()
        self.failed_sat = set()

    def reset(self):
        obs = self.env.reset()
        self.edge_dts = {}
        self.failed_gs.clear()
        self.failed_sat.clear()
        self.policy.reset()
        self.addressing = AddressingSystem()
        self.executor.addressing = self.addressing
        self.executor.n_migrations = 0
        self.executor.n_seamless = 0
        for sid in self.env.edge_dt_ids:
            host = self.env.edge_dt_host[sid]
            dt = EdgeDT(entity_id=sid, host_gs=host,
                        model_state={"host_gs": host, "device_status": "ok"})
            self.edge_dts[sid] = dt
            self.addressing.register(dt)
        self.store.ingest(self.edge_dts, self.env.net, self.env.t)
        return obs

    def decide_actions(self, obs, t):
        actions = {}
        for sid, dt in self.edge_dts.items():
            if sid in self.failed_sat:
                continue
            target = self.policy.decide(dt, obs, t)
            if target is not None:
                actions[sid] = target
        return actions

    def apply_actions(self, actions):
        applied = 0
        for sid, target in actions.items():
            if self.executor.execute(self.edge_dts[sid], target):
                applied += 1
        return applied

    def step(self, obs, t):
        actions = self.decide_actions(obs, t)
        applied = self.apply_actions(actions)
        obs2, reward, info = self.env.step({}, t)
        info["n_migrations"] = applied
        info["total_migrations"] = self.executor.n_migrations
        info["seamless_migrations"] = self.executor.n_seamless
        reward = -(info["mean_latency"] + self.env.cfg.migration_cost * applied)
        self.store.ingest(self.edge_dts, self.env.net, t)
        return obs2, reward, info

    def inject_gs_failure(self, gs_id, obs, t):
        self.failed_gs.add(gs_id)
        affected = [dt for dt in self.edge_dts.values() if dt.host_gs == gs_id]
        for dt in affected:
            target = self._nearest_healthy(dt.entity_id, obs)
            if target is not None:
                self.executor.execute(dt, target)
        return len(affected)

    def inject_sat_failure(self, sat_id):
        self.failed_sat.add(sat_id)

    def _nearest_healthy(self, sat_id, obs):
        best, best_lat = None, float("inf")
        for gs in obs.candidates(sat_id):
            if gs in self.failed_gs or obs.overloaded(gs):
                continue
            lat = obs.candidate_latency(sat_id, gs)
            if lat < best_lat:
                best, best_lat = gs, lat
        return best

    def addressing_consistent(self):
        return self.addressing.consistent(self.edge_dts)
