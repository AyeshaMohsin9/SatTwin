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
        self._setup_remaining = {}
        self._rng_state = 12345
        self.net._fade_state = {}
        self._surge = {}
        self._gs_phase = {}
        self._obs_history = []
        self.queue = {}
        self.battery = {}
        self._dropped = {}
        self._step_drop = {}
        self._age = {}
        self._sat_phase = {}

    @classmethod
    def from_files(cls, scenario_path, stations_path, ping_csv=None):
        return cls(load_scenario(scenario_path), stations_path, ping_csv)

    def _init_demand(self):
        import math
        self._surge = {}
        edge = list(self.ground.edge_stations().items())
        n = max(1, len(edge))
        for i, (gid, gs) in enumerate(edge):
            self._gs_phase[gid] = 2.0 * math.pi * ((gs.lon % 360) / 360.0)
            self._surge[gid] = 0

    def _demand_ms(self, gid, t):
        import math
        cfg = self.cfg
        if cfg.demand_wave_ms <= 0.0 and cfg.demand_surge_ms <= 0.0:
            return 0.0
        slot = t / max(1.0, self.cfg.time_step_s)
        phase = self._gs_phase.get(gid, 0.0)
        wave = cfg.demand_wave_ms * 0.5 * (1.0 + math.sin(
            2.0 * math.pi * slot / max(1, cfg.demand_period_slots) - phase))
        surge = cfg.demand_surge_ms if self._surge.get(gid, 0) > 0 else 0.0
        return wave + surge

    def _init_state(self):
        import math
        self.queue = {s: 0.0 for s in self.constellation.sat_ids}
        self.battery = {s: self.cfg.battery_capacity for s in self.constellation.sat_ids}
        self._dropped = {s: 0.0 for s in self.constellation.sat_ids}
        self._age = {s: 0.0 for s in self.constellation.sat_ids}
        for i, s in enumerate(self.constellation.sat_ids):
            self._sat_phase[s] = 2.0 * math.pi * (i / max(1, len(self.constellation.sat_ids)))

    def _arrival(self, sid, t):
        import math
        cfg = self.cfg
        slot = t / max(1.0, cfg.time_step_s)
        base = cfg.sat_arrival_mean * (1.0 + cfg.sat_arrival_var
                                       * math.sin(0.3 * slot + self._sat_phase.get(sid, 0.0)))
        return max(0.0, base + cfg.sat_arrival_var * (self._rand() - 0.5))

    def _in_eclipse(self, sid, t):
        import math
        slot = t / max(1.0, self.cfg.time_step_s)
        period = self.constellation.period_s / max(1.0, self.cfg.time_step_s)
        phase = (slot / max(1.0, period) + self._sat_phase.get(sid, 0.0) / (2 * math.pi)) % 1.0
        return phase < self.cfg.eclipse_frac

    def _advance_state(self, latencies, target_t, migrated):
        cfg = self.cfg
        if not cfg.queue_enabled and not cfg.battery_enabled:
            return {}
        gw_count = {}
        for s in self.constellation.sat_ids:
            h = self.edge_dt_host.get(s)
            if h is not None:
                gw_count[h] = gw_count.get(h, 0) + 1
        info = {"queue": 0.0, "dropped": 0.0, "flat_battery": 0, "overflow": 0,
                "aoi": 0.0, "expired": 0}
        self._step_drop = {s: 0.0 for s in self.constellation.sat_ids}
        for sid, lat in zip(self.constellation.sat_ids, latencies):
            h = self.edge_dt_host.get(sid)
            bat = self.battery.get(sid, cfg.battery_capacity)
            if cfg.queue_enabled:
                self.queue[sid] = self.queue.get(sid, 0.0) + self._arrival(sid, target_t)
                can_tx = (not cfg.battery_enabled) or bat > 0.0
                drained = 0.0
                if h is not None and lat != float("inf") and can_tx:
                    svc = cfg.gateway_service / max(1, gw_count.get(h, 1))
                    drained = min(self.queue[sid], svc)
                    self.queue[sid] -= drained
                    if cfg.battery_enabled:
                        bat -= cfg.tx_energy * (drained / max(1e-6, cfg.gateway_service))
                if drained > 0.0:
                    self._age[sid] = 0.0 if self.queue[sid] <= 1e-6 else max(0.0, self._age[sid] - 1.0)
                else:
                    self._age[sid] = self._age.get(sid, 0.0) + (1.0 if self.queue[sid] > 1e-6 else 0.0)
                if cfg.data_deadline_slots > 0 and self._age[sid] > cfg.data_deadline_slots:
                    expired = self.queue[sid]
                    self._dropped[sid] += expired
                    self._step_drop[sid] += expired
                    info["dropped"] += expired
                    info["expired"] += 1
                    self.queue[sid] = 0.0
                    self._age[sid] = 0.0
                if self.queue[sid] > cfg.buffer_capacity:
                    over = self.queue[sid] - cfg.buffer_capacity
                    self._dropped[sid] += over
                    self._step_drop[sid] += over
                    info["dropped"] += over
                    self.queue[sid] = cfg.buffer_capacity
                    info["overflow"] += 1
                info["queue"] += self.queue[sid]
                info["aoi"] += self._age[sid]
            if cfg.battery_enabled:
                if migrated.get(sid):
                    bat -= cfg.migrate_energy
                if not self._in_eclipse(sid, target_t):
                    bat += cfg.recharge_rate
                bat = max(0.0, min(cfg.battery_capacity, bat))
                self.battery[sid] = bat
                if bat <= 0.0:
                    info["flat_battery"] += 1
        n = max(1, len(self.constellation.sat_ids))
        info["queue"] /= n
        info["aoi"] /= n
        return info

    def _advance_surges(self):
        if self.cfg.demand_surge_ms <= 0.0:
            return
        for gid, gs in self.ground.stations.items():
            if gs.is_ncc:
                continue
            rem = self._surge.get(gid, 0)
            if rem > 0:
                self._surge[gid] = rem - 1
            elif self._rand() < self.cfg.demand_surge_prob:
                self._surge[gid] = self.cfg.demand_surge_len

    def _rand(self):
        self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
        return self._rng_state / 0x7FFFFFFF

    def _advance_fade(self):
        if self.cfg.rain_fade_penalty_ms <= 0.0:
            return
        corr = self.cfg.rain_fade_corr
        for gid, gs in self.ground.stations.items():
            if gs.is_ncc:
                continue
            prev = self.net._fade_state.get(gid, 0.0)
            wet = prev > 0.0
            if wet:
                stay = self._rand() < corr
                self.net._fade_state[gid] = (
                    self.cfg.rain_fade_penalty_ms * (0.5 + 0.5 * self._rand())
                    if stay else 0.0)
            else:
                if self._rand() < self.cfg.rain_fade_prob:
                    self.net._fade_state[gid] = (
                        self.cfg.rain_fade_penalty_ms * (0.5 + 0.5 * self._rand()))

    @property
    def edge_dt_ids(self):
        return self.constellation.sat_ids

    def reset(self):
        self.t = 0.0
        self.ground.reset_loads()
        self._rng_state = 12345
        self.net._fade_state = {}
        self._setup_remaining = {}
        self._init_demand()
        self._init_state()
        self.net.build(0.0)
        self._built_t = 0.0
        self.edge_dt_host = {}
        for sid in self.constellation.sat_ids:
            gs, _ = self.net.nearest_feasible_gs(sid)
            self.edge_dt_host[sid] = gs
            self._setup_remaining[sid] = 0
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
        obs.queue = dict(self.queue)
        obs.battery = dict(self.battery)
        delay_t = t - self.cfg.feedback_delay_slots * self.cfg.time_step_s
        cong = self._congestion_by_gs(t=max(0.0, delay_t))
        for sid in self.constellation.sat_ids:
            host = self.edge_dt_host[sid]
            obs.host[sid] = host
            cand = self.net.latencies_from_sat(sid, edge_ids)
            if cong:
                cand = {g: (v + cong.get(g, 0.0) if v != float("inf") else v)
                        for g, v in cand.items()}
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
        tgt = self.ground.stations[target_gs]
        if self.cfg.soft_capacity:
            if tgt.load >= tgt.capacity * self.cfg.overload_cap_mult:
                return False
        elif tgt.overloaded():
            return False
        if cur is not None:
            self.ground.stations[cur].remove()
        self.ground.stations[target_gs].add()
        self.edge_dt_host[dt_id] = target_gs
        self._setup_remaining[dt_id] = self.cfg.handover_setup_slots
        return True

    def _congestion_by_gs(self, t=None):
        c = self.cfg.congestion_ms
        tt = self.t if t is None else t
        has_demand = self.cfg.demand_wave_ms > 0.0 or self.cfg.demand_surge_ms > 0.0
        if c <= 0.0 and not has_demand:
            return {}
        out = {}
        for gid, gs in self.ground.stations.items():
            if gs.is_ncc:
                continue
            frac = gs.load / max(1, gs.capacity)
            base = c * (frac ** self.cfg.congestion_exp) if c > 0.0 else 0.0
            dem = self._demand_ms(gid, tt) * (0.5 + frac)
            out[gid] = base + dem
        return out

    def data_rates(self, latencies, bw_weights=None):
        import math
        cfg = self.cfg
        wsum = {}
        for sid in self.constellation.sat_ids:
            h = self.edge_dt_host.get(sid)
            if h is None:
                continue
            w = 1.0 if bw_weights is None else max(1e-3, bw_weights.get(sid, 1.0))
            wsum[h] = wsum.get(h, 0.0) + w
        rates = []
        for sid, lat in zip(self.constellation.sat_ids, latencies):
            h = self.edge_dt_host.get(sid)
            if h is None or lat == float("inf"):
                rates.append(0.0)
                continue
            sinr = (cfg.sinr_ref_ms / max(1.0, lat)) / cfg.noise_floor
            spec_eff = math.log2(1.0 + max(0.0, sinr))
            w = 1.0 if bw_weights is None else max(1e-3, bw_weights.get(sid, 1.0))
            share = cfg.gateway_bandwidth * (w / max(1e-3, wsum.get(h, 1.0)))
            rates.append(spec_eff * share)
        return rates

    def rate_metrics(self, rates):
        import numpy as np
        r = np.asarray(rates, dtype=float)
        s = float(r.sum())
        mn = float(r.min()) if len(r) else 0.0
        jain = float(s * s / (len(r) * float((r * r).sum()) + 1e-9)) if len(r) else 0.0
        return {"sum_rate": s, "min_rate": mn, "jain": jain,
                "mean_rate": s / max(1, len(r))}

    def step(self, actions, t=None, bw_weights=None):
        target_t = self.t + self.cfg.time_step_s if t is None else t
        self._advance_fade()
        self._advance_surges()
        self._ensure_graph(target_t)
        n_migrations = 0
        migrated = {}
        for dt_id, target in actions.items():
            if self.apply_action(dt_id, target):
                n_migrations += 1
                migrated[dt_id] = True
        hp = self.cfg.handover_penalty_ms
        cong = self._congestion_by_gs(t=target_t)
        latencies = []
        for sid in self.constellation.sat_ids:
            host = self.edge_dt_host[sid]
            if host is None:
                latencies.append(float("inf"))
                continue
            lat = self.net.ps_dt_latency(sid, host) + cong.get(host, 0.0)
            rem = self._setup_remaining.get(sid, 0)
            if rem > 0:
                lat += hp
                self._setup_remaining[sid] = rem - 1
            latencies.append(lat)
        self.t = target_t
        st = self._advance_state(latencies, target_t, migrated)
        mean_lat = sum(latencies) / len(latencies)
        rates = self.data_rates(latencies, bw_weights)
        rmet = self.rate_metrics(rates)
        reward = -(mean_lat + self.cfg.migration_cost * n_migrations)
        info = {"latencies": latencies, "n_migrations": n_migrations,
                "mean_latency": mean_lat, "t": target_t,
                "rates": rates, **rmet}
        if st:
            info["mean_queue"] = st["queue"]
            info["dropped"] = st["dropped"]
            info["overflow"] = st["overflow"]
            info["flat_battery"] = st["flat_battery"]
            info["mean_aoi"] = st["aoi"]
            info["expired"] = st["expired"]
            info["mean_battery"] = (sum(self.battery.values()) / max(1, len(self.battery))
                                    if self.cfg.battery_enabled else 0.0)
        return self.observe(target_t), reward, info
