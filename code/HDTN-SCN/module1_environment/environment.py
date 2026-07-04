# Module 1 — Gym-style HDTN environment: observe(t), step(actions), reward, reset.
from .config import load_scenario
from .constellation import Constellation
from .ground import GroundSegment
from .network import NetworkGraph
from .observation import Observation


class HDTNEnvironment:
    def __init__(self, cfg, stations_path, ping_csv=None):
        self.cfg = cfg
        self.constellation = Constellation(cfg)
        self.ground = GroundSegment.from_yaml(stations_path, cfg, ping_csv)
        self.net = NetworkGraph(self.constellation, self.ground, cfg)
        self.edge_dt_host = {}
        self.t = 0.0
        self._built_t = None

    @classmethod
    def from_files(cls, scenario_path, stations_path, ping_csv=None):
        return cls(load_scenario(scenario_path), stations_path, ping_csv)

    @property
    def edge_dt_ids(self):
        return self.constellation.sat_ids

    def reset(self):
        self.t = 0.0
        self.ground.reset_loads()
        self.net.build(0.0)
        self._built_t = 0.0
        self.edge_dt_host = {}
        for sid in self.constellation.sat_ids:
            gs, _ = self.net.nearest_gs_latency(sid)
            self.edge_dt_host[sid] = gs
            if gs is not None:
                self.ground.stations[gs].add()
        return self.observe(0.0)

    def _ensure_graph(self, t):
        if self._built_t != t:
            self.net.build(t)
            self._built_t = t

    def observe(self, t, predict=True):
        self._ensure_graph(t)
        edge_ids = list(self.ground.edge_stations().keys())
        obs = Observation(t=t, edge_station_ids=edge_ids)
        for gid, gs in self.ground.stations.items():
            obs.gs_load[gid] = gs.load
            obs.gs_capacity[gid] = gs.capacity
        for sid in self.constellation.sat_ids:
            host = self.edge_dt_host[sid]
            obs.host[sid] = host
            cand = self.net.latencies_from_sat(sid, edge_ids)
            obs.cand_latency[sid] = cand
            obs.latency[sid] = cand.get(host, float("inf")) if host is not None \
                else float("inf")
        if predict:
            self._fill_predictions(obs, t)
        else:
            for sid in self.constellation.sat_ids:
                obs.predicted_latency[sid] = obs.latency[sid]
        return obs

    def _fill_predictions(self, obs, t):
        self.net.build(t + self.cfg.time_step_s)
        for sid in self.constellation.sat_ids:
            host = self.edge_dt_host[sid]
            obs.predicted_latency[sid] = (self.net.ps_dt_latency(sid, host)
                                          if host is not None else float("inf"))
        self.net.build(t)
        self._built_t = t

    def apply_action(self, dt_id, target_gs):
        if target_gs is None:
            return False
        cur = self.edge_dt_host[dt_id]
        if target_gs == cur:
            return False
        if self.ground.stations[target_gs].overloaded():
            return False
        if cur is not None:
            self.ground.stations[cur].remove()
        self.ground.stations[target_gs].add()
        self.edge_dt_host[dt_id] = target_gs
        return True

    def step(self, actions, t=None):
        target_t = self.t + self.cfg.time_step_s if t is None else t
        self._ensure_graph(target_t)
        n_migrations = 0
        for dt_id, target in actions.items():
            if self.apply_action(dt_id, target):
                n_migrations += 1
        latencies = []
        for sid in self.constellation.sat_ids:
            host = self.edge_dt_host[sid]
            latencies.append(self.net.ps_dt_latency(sid, host)
                             if host is not None else float("inf"))
        self.t = target_t
        mean_lat = sum(latencies) / len(latencies)
        reward = -(mean_lat + self.cfg.migration_cost * n_migrations)
        info = {"latencies": latencies, "n_migrations": n_migrations,
                "mean_latency": mean_lat, "t": target_t}
        return self.observe(target_t), reward, info
