# Module 1 — ECEF geometry, propagation delay, and satellite/ground visibility.
import numpy as np

C_KM_PER_S = 299_792.458
EARTH_RADIUS_KM = 6_371.0
MU_EARTH_KM3_S2 = 398_600.4418
EARTH_ANGULAR_RATE_RAD_S = 7.2921159e-5


def geodetic_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


def dist_km(p_a, p_b):
    return float(np.linalg.norm(np.asarray(p_a) - np.asarray(p_b)))


def prop_delay_ms(p_a, p_b):
    return dist_km(p_a, p_b) / C_KM_PER_S * 1000.0


def orbital_period_s(altitude_km):
    a = EARTH_RADIUS_KM + altitude_km
    return 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH_KM3_S2)


def mean_motion_rad_s(altitude_km):
    a = EARTH_RADIUS_KM + altitude_km
    return np.sqrt(MU_EARTH_KM3_S2 / a ** 3)


def elevation_deg(sat_ecef, gs_ecef):
    sat = np.asarray(sat_ecef)
    gs = np.asarray(gs_ecef)
    up = gs / np.linalg.norm(gs)
    los = sat - gs
    los_norm = np.linalg.norm(los)
    if los_norm == 0.0:
        return 90.0
    sin_el = float(np.dot(los, up) / los_norm)
    sin_el = max(-1.0, min(1.0, sin_el))
    return np.degrees(np.arcsin(sin_el))


def visible(sat_ecef, gs_ecef, min_elevation_deg=10.0):
    return elevation_deg(sat_ecef, gs_ecef) >= min_elevation_deg


def earth_blocked(p_a, p_b):
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    d = b - a
    dd = float(np.dot(d, d))
    if dd == 0.0:
        return np.linalg.norm(a) < EARTH_RADIUS_KM
    t = -float(np.dot(a, d)) / dd
    t = max(0.0, min(1.0, t))
    closest = a + t * d
    return np.linalg.norm(closest) < EARTH_RADIUS_KM
