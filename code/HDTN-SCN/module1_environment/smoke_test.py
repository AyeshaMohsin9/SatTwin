# Module 1 — self-contained smoke test validating geometry, graph, and environment API.
import os

from module1_environment import (
    HDTNEnvironment, geodetic_to_ecef, prop_delay_ms, orbital_period_s,
    EARTH_RADIUS_KM,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def test_overhead_gsl_delay():
    gs = geodetic_to_ecef(0.0, 0.0, 0.0)
    sat = geodetic_to_ecef(0.0, 0.0, 780.0)
    d = prop_delay_ms(sat, gs)
    assert 2.4 <= d <= 2.8, d
    print(f"[ok] overhead GSL delay = {d:.3f} ms (expect ~2.6)")


def test_orbital_period():
    p = orbital_period_s(780.0)
    assert 5800 <= p <= 6200, p
    print(f"[ok] orbital period = {p:.0f} s (expect ~6000)")


def test_env_reset_and_step():
    env = HDTNEnvironment.from_files(
        _p("config", "con1.yaml"), _p("config", "stations.yaml"))
    obs = env.reset()
    assert len(env.edge_dt_ids) == 66, len(env.edge_dt_ids)
    lats = [obs.latency[s] for s in env.edge_dt_ids if obs.latency[s] != float("inf")]
    mean_lat = sum(lats) / len(lats)
    print(f"[ok] con1 sats = {len(env.edge_dt_ids)}, initial mean PS-DT (nearest GS) "
          f"= {mean_lat:.2f} ms over {len(lats)} linked sats")
    actions = {}
    for sid in env.edge_dt_ids:
        gs, _ = env.net.nearest_gs_latency(sid)
        if gs is not None and gs != obs.host[sid]:
            actions[sid] = gs
    obs2, reward, info = env.step(actions, t=env.cfg.time_step_s)
    print(f"[ok] step -> mean_latency={info['mean_latency']:.2f} ms, "
          f"migrations={info['n_migrations']}, reward={reward:.2f}")
    fv = obs2.feature_vector(env.edge_dt_ids[0])
    assert fv.ndim == 1 and len(fv) == 2 + 2 * len(obs2.edge_station_ids), fv.shape
    print(f"[ok] feature_vector dim = {len(fv)}")


def test_benchmark_latencies():
    env = HDTNEnvironment.from_files(
        _p("config", "con1.yaml"), _p("config", "stations.yaml"))
    env.reset()
    env.net.build(1000.0)
    sid = env.edge_dt_ids[0]
    near_gs, near = env.net.nearest_gs_latency(sid)
    fixed = env.net.ps_dt_latency(sid, env.edge_dt_host[sid])
    isl = env.net.sat_to_ncc_isl_latency(sid)
    terr = env.net.sat_to_ncc_terrestrial_latency(sid)
    print(f"[ok] sample sat {sid}: nearest={near:.2f} ({near_gs}), fixed-host={fixed:.2f}, "
          f"ncc-isl={isl:.2f}, ncc-terr={terr:.2f} ms")
    assert near != float("inf")


if __name__ == "__main__":
    test_overhead_gsl_delay()
    test_orbital_period()
    test_env_reset_and_step()
    test_benchmark_latencies()
    print("\nAll Module 1 smoke tests passed.")
