# Module 1 — ground stations, NCC, terrestrial ping matrix, and load bookkeeping.
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .geometry import geodetic_to_ecef, dist_km, C_KM_PER_S


@dataclass
class GroundStation:
    id: str
    lat: float
    lon: float
    is_ncc: bool = False
    capacity: int = 20
    load: int = 0

    @property
    def ecef(self):
        return geodetic_to_ecef(self.lat, self.lon, 0.0)

    def overloaded(self):
        return self.load >= self.capacity

    def add(self):
        self.load += 1

    def remove(self):
        self.load = max(0, self.load - 1)


@dataclass
class GroundSegment:
    stations: dict = field(default_factory=dict)
    ncc_id: str = ""
    ping_ms: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, stations_path, cfg, ping_csv=None):
        with open(stations_path) as f:
            spec = yaml.safe_load(f)
        stations = {}
        ncc_id = ""
        for entry in spec["stations"]:
            is_ncc = bool(entry.get("ncc", False))
            gs = GroundStation(
                id=entry["id"], lat=entry["lat"], lon=entry["lon"],
                is_ncc=is_ncc, capacity=entry.get("capacity", cfg.gs_capacity),
            )
            stations[gs.id] = gs
            if is_ncc:
                ncc_id = gs.id
        seg = cls(stations=stations, ncc_id=ncc_id)
        seg.ping_ms = seg._build_pings(cfg, ping_csv)
        return seg

    def _build_pings(self, cfg, ping_csv):
        if ping_csv is not None:
            df = pd.read_csv(ping_csv)
            table = {}
            for _, row in df.iterrows():
                table[(row["src"], row["dst"])] = float(row["one_way_ms"])
                table[(row["dst"], row["src"])] = float(row["one_way_ms"])
            return table
        return self._synthetic_pings(cfg)

    def _synthetic_pings(self, cfg):
        table = {}
        ids = list(self.stations.keys())
        v = cfg.fiber_factor * C_KM_PER_S
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                d = dist_km(self.stations[a].ecef, self.stations[b].ecef)
                lat = d / v * 1000.0 + cfg.router_overhead_ms
                table[(a, b)] = lat
                table[(b, a)] = lat
        return table

    @property
    def ncc(self):
        return self.stations[self.ncc_id]

    def edge_stations(self):
        return {sid: gs for sid, gs in self.stations.items() if not gs.is_ncc}

    def ping(self, a, b):
        if a == b:
            return 0.0
        return self.ping_ms[(a, b)]

    def reset_loads(self):
        for gs in self.stations.values():
            gs.load = 0
