# Module 1 — scenario configuration dataclass and YAML loader.
from dataclasses import dataclass
import yaml


@dataclass
class ScenarioConfig:
    name: str
    n_planes: int
    sats_per_plane: int
    inclination_deg: float
    altitude_km: float
    phasing_f: int = 1
    raan_spread_deg: float = 360.0
    min_elevation_deg: float = 10.0
    max_isl_range_km: float = 5000.0
    gs_capacity: int = 20
    migration_cost: float = 1.0
    fiber_factor: float = 0.67
    router_overhead_ms: float = 5.0
    time_step_s: float = 1.0

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane


def load_scenario(path: str) -> ScenarioConfig:
    with open(path) as f:
        return ScenarioConfig(**yaml.safe_load(f))
