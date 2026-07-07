# Module 4 — per-agent local observation and centralized global state (CTDE).
import numpy as np

BIG = 1e6
LAT_SCALE = 100.0
UNREACH = 2.0
AOI_SCALE = 6000.0


def _finite(x):
    return BIG if x == float("inf") else float(x)


def _norm_lat(x):
    if x == float("inf"):
        return UNREACH
    return min(UNREACH, float(x) / LAT_SCALE)


class ObsBuilder:
    def __init__(self, env):
        self.env = env
        self.gs_ids = list(env.ground.edge_stations().keys())
        self.sat_ids = list(env.constellation.sat_ids)
        self.n_gs = len(self.gs_ids)
        self._gs_index = {g: i for i, g in enumerate(self.gs_ids)}
        self.last_migration_t = {s: 0.0 for s in self.sat_ids}

    def local_dim(self):
        return 6 + 3 * self.n_gs

    def global_dim(self):
        return len(self.sat_ids) * (3 + self.n_gs) + 2 * self.n_gs

    def reset(self):
        self.last_migration_t = {s: 0.0 for s in self.sat_ids}

    def mark_migration(self, sat_id, t):
        self.last_migration_t[sat_id] = t

    def local_obs(self, sat_id, obs, mpc_preview=None):
        host = obs.host.get(sat_id)
        cur = _norm_lat(obs.latency.get(sat_id, float("inf")))
        pred = _norm_lat(obs.predicted_latency.get(sat_id, float("inf")))
        aoi = (obs.t - self.last_migration_t.get(sat_id, 0.0)) / AOI_SCALE
        host_idx = self._gs_index.get(host, -1)
        qcap = max(1.0, getattr(self.env.cfg, "buffer_capacity", 30.0))
        bcap = max(1.0, getattr(self.env.cfg, "battery_capacity", 100.0))
        q = min(1.5, obs.queue.get(sat_id, 0.0) / qcap) if obs.queue else 0.0
        bat = (obs.battery.get(sat_id, bcap) / bcap) if obs.battery else 1.0
        scalar = [cur, pred, aoi, host_idx / max(1, self.n_gs - 1), q, bat]
        cand = obs.cand_latency.get(sat_id, {})
        cand_vec, load_vec, rank_vec = [], [], []
        lat_pairs = []
        for g in self.gs_ids:
            cand_vec.append(_norm_lat(cand.get(g, float("inf"))))
            cap = max(1, obs.gs_capacity.get(g, 1))
            load_vec.append(obs.gs_load.get(g, 0) / cap)
            lat_pairs.append(cand.get(g, float("inf")))
        order = np.argsort(lat_pairs)
        rank = np.zeros(self.n_gs)
        for r, idx in enumerate(order):
            rank[idx] = r / max(1, self.n_gs - 1)
        rank_vec = rank.tolist()
        vec = np.asarray(scalar + cand_vec + load_vec + rank_vec, dtype=np.float32)
        if mpc_preview is not None:
            vec = np.concatenate([vec, mpc_preview.astype(np.float32)])
        return vec

    def global_state(self, obs):
        parts = []
        qcap = max(1.0, getattr(self.env.cfg, "buffer_capacity", 30.0))
        bcap = max(1.0, getattr(self.env.cfg, "battery_capacity", 100.0))
        for s in self.sat_ids:
            host = obs.host.get(s)
            onehot = np.zeros(self.n_gs, dtype=np.float32)
            if host in self._gs_index:
                onehot[self._gs_index[host]] = 1.0
            parts.append(onehot)
            q = min(1.5, obs.queue.get(s, 0.0) / qcap) if obs.queue else 0.0
            bat = (obs.battery.get(s, bcap) / bcap) if obs.battery else 1.0
            parts.append(np.asarray([_norm_lat(obs.latency.get(s, float("inf"))),
                                     q, bat], dtype=np.float32))
        loads = np.asarray([obs.gs_load.get(g, 0) / max(1, obs.gs_capacity.get(g, 1))
                            for g in self.gs_ids], dtype=np.float32)
        caps = np.asarray([1.0 for g in self.gs_ids],
                          dtype=np.float32)
        parts.append(loads)
        parts.append(caps)
        return np.concatenate(parts)
