# Module 7 — capture a deterministic demo rollout (positions, hosts, migrations) for the GUI.
import numpy as np

from module1_environment.geometry import EARTH_RADIUS_KM


def _ecef_to_latlon(pos):
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = (x * x + y * y + z * z) ** 0.5
    if r == 0:
        return 0.0, 0.0
    lat = np.degrees(np.arcsin(max(-1.0, min(1.0, z / r))))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def _station_coords(env):
    out = {}
    for gid, gs in env.core.ground.stations.items():
        out[gid] = {"lat": gs.lat, "lon": gs.lon, "is_ncc": gs.is_ncc}
    return out


def capture_demo(env, policy, cp_factory, n_steps=40, max_sats=30, dt_s=90.0):
    from module2_dt_control import DTControlPlane
    cp = cp_factory() if cp_factory else DTControlPlane(env.core, policy)
    obs = cp.reset()
    sat_ids = list(env.core.constellation.sat_ids)[:max_sats]
    frames = []
    for k in range(n_steps):
        prev_host = {s: cp.edge_dts[s].host_gs for s in sat_ids}
        obs, reward, info = cp.step(obs, t=(k + 1) * dt_s)
        positions = cp.env.constellation.positions(cp.env.t)
        sats = []
        migrations = []
        for s in sat_ids:
            lat, lon = _ecef_to_latlon(positions[s])
            host = cp.edge_dts[s].host_gs
            latency = obs.latency.get(s, None)
            sats.append({"id": s, "lat": round(lat, 2), "lon": round(lon, 2),
                         "host": host,
                         "latency": None if latency in (None, float("inf"))
                         else round(latency, 2)})
            if host != prev_host[s] and host is not None:
                migrations.append({"sat": s, "from": prev_host[s], "to": host})
        frames.append({
            "step": k,
            "t": cp.env.t,
            "mean_latency": round(info["mean_latency"], 2),
            "migrations": info["n_migrations"],
            "sats": sats,
            "migration_events": migrations,
        })
    return {"stations": _station_coords(env), "frames": frames}
