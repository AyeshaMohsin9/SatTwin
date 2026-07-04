# Module 1 — physical & network substrate: public API surface.
from .config import ScenarioConfig, load_scenario
from .constellation import Constellation
from .geometry import (
    C_KM_PER_S, EARTH_RADIUS_KM, geodetic_to_ecef, dist_km, prop_delay_ms,
    orbital_period_s, mean_motion_rad_s, elevation_deg, visible, earth_blocked,
)
from .ground import GroundStation, GroundSegment
from .network import NetworkGraph
from .observation import Observation
from .environment import HDTNEnvironment

__all__ = [
    "ScenarioConfig", "load_scenario", "Constellation",
    "C_KM_PER_S", "EARTH_RADIUS_KM", "geodetic_to_ecef", "dist_km",
    "prop_delay_ms", "orbital_period_s", "mean_motion_rad_s", "elevation_deg",
    "visible", "earth_blocked", "GroundStation", "GroundSegment",
    "NetworkGraph", "Observation", "HDTNEnvironment",
]
