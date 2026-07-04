# Module 3 — integration smoke test: simulator, four benchmarks, threshold sweep, metrics.
import os

from module3_simulation import (
    run_scheme, run_all_schemes, sweep_threshold, HDTN_SCN, BENCH1_NO_MIGRATION,
    BENCH2_CENTRAL_ISL, BENCH3_CENTRAL_TERRESTRIAL,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def _scen():
    return _p("config", "con1.yaml"), _p("config", "stations.yaml")


def test_single_scheme():
    scen, sta = _scen()
    res = run_scheme(scen, sta, HDTN_SCN, horizon_s=1200.0, threshold_ms=200.0, dt_s=30.0)
    assert res.n_edge_dts == 66
    assert len(res.times) == 40
    print(f"[ok] HDTN-SCN 1200s/30s: mean={res.mean_latency:.2f} ms, "
          f"p95={res.p95:.2f} ms, migrations={res.total_migrations}")


def test_all_schemes_ordering():
    scen, sta = _scen()
    results = run_all_schemes(scen, sta, horizon_s=1200.0, threshold_ms=0.0, dt_s=60.0)
    m = {k: v.mean_latency for k, v in results.items()}
    for k in results:
        print(f"[ok] {k}: mean={m[k]:.2f} ms  {results[k].summary()}")
    assert m[HDTN_SCN] <= m[BENCH1_NO_MIGRATION], (m[HDTN_SCN], m[BENCH1_NO_MIGRATION])
    assert m[BENCH3_CENTRAL_TERRESTRIAL] >= m[BENCH2_CENTRAL_ISL]
    assert m[HDTN_SCN] <= m[BENCH3_CENTRAL_TERRESTRIAL]
    print("[ok] ordering: HDTN-SCN <= Benchmark-1; Benchmark-3 >= Benchmark-2")


def test_threshold_sweep():
    scen, sta = _scen()
    rows = sweep_threshold(scen, sta, [0, 80, 200], horizon_s=1200.0,
                           window_s=6000.0, dt_s=30.0)
    for r in rows:
        print(f"[ok] thr={r['threshold_ms']:>5.0f} ms -> "
              f"freq={r['migration_frequency']:.4f}, "
              f"lat={r['mean_latency_ms']:.2f} ms, mig={r['total_migrations']}")
    freqs = [r["migration_frequency"] for r in rows]
    assert freqs[0] >= freqs[-1]
    assert rows[-1]["total_migrations"] == 0
    print("[ok] sweep monotonic; thr=200 -> zero migrations")


if __name__ == "__main__":
    test_single_scheme()
    test_all_schemes_ordering()
    test_threshold_sweep()
    print("\nAll Module 3 integration tests passed.")
