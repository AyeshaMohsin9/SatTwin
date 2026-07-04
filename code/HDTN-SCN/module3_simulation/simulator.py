# Module 3 — discrete-time simulator driving HDTN-SCN (Module 2) and static benchmarks.
import numpy as np

from module1_environment import HDTNEnvironment, load_scenario
from module2_dt_control import DTControlPlane, GreedyNearestPolicy

from .benchmarks import (
    BenchmarkScheme, HDTN_SCN, BENCH1_NO_MIGRATION, BENCH2_CENTRAL_ISL,
    BENCH3_CENTRAL_TERRESTRIAL, ALL_SCHEMES,
)
from .metrics import RunResult


def make_env(scenario_path, stations_path, ping_csv=None):
    return HDTNEnvironment.from_files(scenario_path, stations_path, ping_csv)


def run_hdtn_scn(env, threshold_ms, horizon_s, dt_s=None, policy=None,
                 seamless=True):
    dt_s = dt_s or env.cfg.time_step_s
    policy = policy or GreedyNearestPolicy(threshold_ms=threshold_ms)
    cp = DTControlPlane(env, policy, seamless=seamless)
    obs = cp.reset()
    res = RunResult(scheme=HDTN_SCN, n_edge_dts=len(cp.edge_dts),
                    horizon_s=horizon_s)
    for t in np.arange(dt_s, horizon_s + dt_s, dt_s):
        obs, reward, info = cp.step(obs, t=float(t))
        res.add_step(float(t), info["latencies"], info["n_migrations"])
    return res


def run_benchmark(env, scheme_name, horizon_s, dt_s=None):
    dt_s = dt_s or env.cfg.time_step_s
    scheme = BenchmarkScheme(scheme_name, env)
    scheme.reset()
    res = RunResult(scheme=scheme_name,
                    n_edge_dts=len(env.constellation.sat_ids),
                    horizon_s=horizon_s)
    for t in np.arange(0.0, horizon_s, dt_s):
        res.add_step(float(t), scheme.latencies(float(t)), 0)
    return res


def run_scheme(scenario_path, stations_path, scheme_name, horizon_s,
               threshold_ms=200.0, dt_s=None, ping_csv=None):
    env = make_env(scenario_path, stations_path, ping_csv)
    if scheme_name == HDTN_SCN:
        env.reset()
        return run_hdtn_scn(env, threshold_ms, horizon_s, dt_s)
    return run_benchmark(env, scheme_name, horizon_s, dt_s)


def run_all_schemes(scenario_path, stations_path, horizon_s,
                    threshold_ms=200.0, dt_s=None, ping_csv=None):
    return {name: run_scheme(scenario_path, stations_path, name, horizon_s,
                             threshold_ms, dt_s, ping_csv)
            for name in ALL_SCHEMES}


def sweep_threshold(scenario_path, stations_path, thresholds, horizon_s,
                    window_s, dt_s=None, ping_csv=None):
    rows = []
    for thr in thresholds:
        env = make_env(scenario_path, stations_path, ping_csv)
        env.reset()
        res = run_hdtn_scn(env, thr, horizon_s, dt_s)
        rows.append({
            "threshold_ms": thr,
            "mean_latency_ms": res.mean_latency,
            "migration_frequency": res.migration_frequency(window_s),
            "total_migrations": res.total_migrations,
        })
    return rows
