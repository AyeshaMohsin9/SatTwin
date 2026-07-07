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
    handover_penalty_ms: float = 0.0
    handover_setup_slots: int = 0
    elevation_penalty_ms: float = 0.0
    rain_fade_prob: float = 0.0
    rain_fade_penalty_ms: float = 0.0
    rain_fade_corr: float = 0.9
    congestion_ms: float = 0.0
    congestion_exp: float = 2.0
    gateway_bandwidth: float = 100.0
    sinr_ref_ms: float = 20.0
    noise_floor: float = 1.0
    feedback_delay_slots: int = 0
    demand_wave_ms: float = 0.0
    demand_period_slots: int = 40
    demand_surge_prob: float = 0.0
    demand_surge_ms: float = 0.0
    demand_surge_len: int = 5
    queue_enabled: bool = False
    gateway_service: float = 8.0
    queue_latency_coef: float = 12.0
    sat_arrival_mean: float = 1.0
    sat_arrival_var: float = 0.5
    buffer_capacity: float = 30.0
    buffer_penalty: float = 5.0
    battery_enabled: bool = False
    battery_capacity: float = 100.0
    tx_energy: float = 1.0
    migrate_energy: float = 3.0
    recharge_rate: float = 2.0
    eclipse_frac: float = 0.35
    low_battery_frac: float = 0.15
    soft_capacity: bool = False
    overload_cap_mult: float = 2.5

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane


def load_scenario(path: str) -> ScenarioConfig:
    with open(path) as f:
        return ScenarioConfig(**yaml.safe_load(f))
