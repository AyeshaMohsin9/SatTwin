# Module 2 — Pillar A: MigrationPolicy interface + policy-agnostic migration mechanics.
from abc import ABC, abstractmethod


class MigrationPolicy(ABC):
    @abstractmethod
    def decide(self, dt, obs, t):
        ...

    def reset(self):
        pass


class MigrationExecutor:
    def __init__(self, env, addressing, seamless=True):
        self.env = env
        self.addressing = addressing
        self.seamless = seamless
        self.n_migrations = 0
        self.n_seamless = 0

    def execute(self, dt, target_gs):
        if target_gs is None or target_gs == dt.host_gs:
            return False
        if self.env.ground.stations[target_gs].overloaded():
            return False
        dt.being_migrated = True
        old = dt.host_gs
        applied = self.env.apply_action(dt.entity_id, target_gs)
        if not applied:
            dt.being_migrated = False
            return False
        dt.host_gs = target_gs
        dt.model_state["host_gs"] = target_gs
        self.addressing.on_migration(dt, target_gs)
        if self.seamless:
            self.n_seamless += 1
        dt.being_migrated = False
        self.n_migrations += 1
        return True

    def stats(self):
        return {"n_migrations": self.n_migrations, "n_seamless": self.n_seamless}
