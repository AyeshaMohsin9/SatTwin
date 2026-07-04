# Module 1 — Walker-delta LEO constellation with analytic circular-orbit propagation.
import numpy as np

from .geometry import EARTH_RADIUS_KM, mean_motion_rad_s, orbital_period_s


class Constellation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planes = cfg.n_planes
        self.sats_per_plane = cfg.sats_per_plane
        self.altitude_km = cfg.altitude_km
        self.radius_km = EARTH_RADIUS_KM + cfg.altitude_km
        self.inc = np.radians(cfg.inclination_deg)
        self.n = mean_motion_rad_s(cfg.altitude_km)
        self.period_s = orbital_period_s(cfg.altitude_km)
        self.sat_ids = []
        self._raan = {}
        self._m0 = {}
        self._plane = {}
        self._index = {}
        self._build()

    def _build(self):
        raan_step = np.radians(self.cfg.raan_spread_deg) / self.n_planes
        phase_step = 2.0 * np.pi / self.sats_per_plane
        inter_plane = 2.0 * np.pi * self.cfg.phasing_f / self.cfg.n_sats
        for p in range(self.n_planes):
            for s in range(self.sats_per_plane):
                sid = f"sat_{p}_{s}"
                self.sat_ids.append(sid)
                self._raan[sid] = p * raan_step
                self._m0[sid] = s * phase_step + p * inter_plane
                self._plane[sid] = p
                self._index[sid] = s

    def _position(self, sid, t):
        raan = self._raan[sid]
        u = self._m0[sid] + self.n * t
        x_orb = self.radius_km * np.cos(u)
        y_orb = self.radius_km * np.sin(u)
        cos_i, sin_i = np.cos(self.inc), np.sin(self.inc)
        cos_o, sin_o = np.cos(raan), np.sin(raan)
        x = x_orb * cos_o - y_orb * cos_i * sin_o
        y = x_orb * sin_o + y_orb * cos_i * cos_o
        z = y_orb * sin_i
        return np.array([x, y, z])

    def positions(self, t):
        return {sid: self._position(sid, t) for sid in self.sat_ids}

    def plane_of(self, sid):
        return self._plane[sid]

    def index_in_plane(self, sid):
        return self._index[sid]

    def intra_plane_neighbors(self, sid):
        p, s = self._plane[sid], self._index[sid]
        prev_s = (s - 1) % self.sats_per_plane
        next_s = (s + 1) % self.sats_per_plane
        return [f"sat_{p}_{prev_s}", f"sat_{p}_{next_s}"]

    def cross_plane_neighbors(self, sid):
        p, s = self._plane[sid], self._index[sid]
        prev_p = (p - 1) % self.n_planes
        next_p = (p + 1) % self.n_planes
        return [f"sat_{prev_p}_{s}", f"sat_{next_p}_{s}"]
