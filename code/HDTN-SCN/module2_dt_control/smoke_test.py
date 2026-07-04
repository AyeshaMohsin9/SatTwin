# Module 2 — integration smoke test: control plane over Module 1 across all three pillars.
import os

from module1_environment import HDTNEnvironment
from module2_dt_control import (
    DTControlPlane, GreedyNearestPolicy, RLMigrationPolicy, SlicingManager, Flow,
    LOW_LATENCY, BEST_EFFORT,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def _env():
    return HDTNEnvironment.from_files(
        _p("config", "con1.yaml"), _p("config", "stations.yaml"))


def test_control_plane_reset():
    cp = DTControlPlane(_env(), GreedyNearestPolicy(threshold_ms=30.0))
    obs = cp.reset()
    assert len(cp.edge_dts) == 66
    assert cp.addressing_consistent()
    assert len(cp.central_dts) == 4
    print(f"[ok] reset: {len(cp.edge_dts)} edge-DTs, {len(cp.central_dts)} central-DTs, "
          f"addressing consistent = {cp.addressing_consistent()}")


def test_migration_run():
    cp = DTControlPlane(_env(), GreedyNearestPolicy(threshold_ms=20.0))
    obs = cp.reset()
    total_mig, lats = 0, []
    for k in range(120):
        obs, reward, info = cp.step(obs, t=(k + 1) * cp.env.cfg.time_step_s)
        total_mig += info["n_migrations"]
        lats.append(info["mean_latency"])
        assert cp.addressing_consistent(), f"addressing broke at step {k}"
    print(f"[ok] 120 steps @thr=20ms: migrations={total_mig}, "
          f"mean_latency={sum(lats)/len(lats):.2f} ms, addressing stayed consistent")


def test_threshold_monotonicity():
    results = {}
    for thr in [0.0, 40.0, 80.0, 120.0, 200.0]:
        cp = DTControlPlane(_env(), GreedyNearestPolicy(threshold_ms=thr))
        obs = cp.reset()
        mig, lats = 0, []
        for k in range(60):
            obs, reward, info = cp.step(obs, t=(k + 1) * cp.env.cfg.time_step_s)
            mig += info["n_migrations"]
            lats.append(info["mean_latency"])
        results[thr] = (mig, sum(lats) / len(lats))
        print(f"[ok] thr={thr:>5.0f} ms -> migrations={mig:>4d}, "
              f"mean_latency={results[thr][1]:.2f} ms")
    assert results[0.0][0] >= results[200.0][0]
    assert results[200.0][0] == 0
    print("[ok] monotonic: migrations fall as threshold rises; thr=200 -> 0 migrations")


def test_pillar_c_seamless():
    cp = DTControlPlane(_env(), GreedyNearestPolicy(threshold_ms=15.0), seamless=True)
    obs = cp.reset()
    for k in range(30):
        obs, reward, info = cp.step(obs, t=(k + 1) * cp.env.cfg.time_step_s)
    dt = cp.edge_dts[cp.env.edge_dt_ids[0]]
    dt_id, gs = cp.addressing.locate_dt_from_ps(dt.entity_id)
    ps = cp.addressing.locate_ps_from_dt(dt.dt_id)
    print(f"[ok] addressing lookups: PS {dt.entity_id} -> DT {dt_id} @ {gs}; "
          f"DT {dt.dt_id} -> PS {ps}; seamless migrations={cp.executor.n_seamless}")
    assert gs == dt.host_gs and ps == dt.entity_id


def test_pillar_b_slicing():
    env = _env()
    env.reset()
    env.net.build(1000.0)
    sm = SlicingManager(env.net)
    sid = env.edge_dt_ids[0]
    ncc = env.ground.ncc_id
    f_ll = Flow(src=sid, dst=ncc, data_type="fault_diagnosis")
    f_be = Flow(src=sid, dst=ncc, data_type="ai_feedback")
    assert sm.classify(f_ll) == LOW_LATENCY and sm.classify(f_be) == BEST_EFFORT
    p_ll = sm.route(f_ll)
    p_be = sm.route(f_be)
    print(f"[ok] slicing: low-latency path delay={sm.path_delay(p_ll):.2f} ms "
          f"({len(p_ll)} hops), best-effort path delay={sm.path_delay(p_be):.2f} ms "
          f"({len(p_be)} hops)")
    assert p_ll and p_be


def test_irregular_migration():
    cp = DTControlPlane(_env(), GreedyNearestPolicy(threshold_ms=30.0))
    obs = cp.reset()
    victim = None
    for gs_id, gs in cp.env.ground.edge_stations().items():
        if gs.load > 0:
            victim = gs_id
            break
    n = cp.inject_gs_failure(victim, obs, cp.env.t)
    assert cp.addressing_consistent()
    print(f"[ok] irregular migration: GS '{victim}' failed, {n} edge-DTs evacuated, "
          f"addressing consistent = {cp.addressing_consistent()}")


def test_rl_policy_interface():
    cp = DTControlPlane(_env(), RLMigrationPolicy(model=None, threshold_ms=20.0))
    obs = cp.reset()
    obs, reward, info = cp.step(obs, t=cp.env.cfg.time_step_s)
    print(f"[ok] RLMigrationPolicy (untrained fallback) runs via same interface: "
          f"migrations={info['n_migrations']}, reward={reward:.2f}")


if __name__ == "__main__":
    test_control_plane_reset()
    test_migration_run()
    test_threshold_monotonicity()
    test_pillar_c_seamless()
    test_pillar_b_slicing()
    test_irregular_migration()
    test_rl_policy_interface()
    print("\nAll Module 2 integration tests passed.")
