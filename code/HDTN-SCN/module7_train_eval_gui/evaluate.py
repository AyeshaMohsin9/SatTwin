# Module 7 — deterministic evaluation of a trained MAPPO checkpoint vs greedy/benchmarks.
import argparse
import os

import numpy as np
import torch
import yaml

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import MPCLookahead
from module6_mappo import build_mappo, MAPPOPolicy
from module2_dt_control import DTControlPlane, GreedyNearestPolicy
from module3_simulation import run_scheme, BENCH1_NO_MIGRATION, BENCH2_CENTRAL_ISL, \
    BENCH3_CENTRAL_TERRESTRIAL
from module3_simulation.metrics import RunResult

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def _run_policy(env, policy, horizon_s, dt_s, label):
    cp = DTControlPlane(env.core, policy)
    obs = cp.reset()
    res = RunResult(scheme=label, n_edge_dts=len(cp.edge_dts), horizon_s=horizon_s)
    n_steps = int(horizon_s / dt_s)
    loads = []
    for k in range(n_steps):
        obs, r, info = cp.step(obs, t=(k + 1) * dt_s)
        res.add_step(cp.env.t, info["latencies"], info["n_migrations"])
        gl = [cp.env.ground.stations[g].load
              for g in cp.env.ground.edge_stations()]
        loads.append(np.var(gl))
    return res, float(np.mean(loads))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_p("module7_train_eval_gui", "config",
                                           "mappo_con1_balanced.yaml"))
    ap.add_argument("--ckpt", default=_p("results", "mappo_norm", "checkpoint.pt"))
    ap.add_argument("--horizon", type=float, default=6000.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dt_s = cfg["dt_s"]
    scen = _p("config", f"{cfg['con']}.yaml")
    sta = _p("config", cfg.get("stations_file", "stations.yaml"))

    mpc = MPCLookahead(horizon=cfg["mpc_horizon"]) if cfg["use_mpc"] else None
    env = HDTNParallelEnv(scen, sta, reward_cfg=RewardConfig(**cfg["reward"]),
                          horizon_s=args.horizon, dt_s=dt_s, mpc_engine=mpc)
    m = build_mappo(env, mpc_engine=mpc, use_gnn=cfg["use_gnn"],
                    use_temporal=cfg["use_temporal"], gnn_hidden=cfg["gnn_hidden"],
                    gnn_layers=cfg["gnn_layers"], device=device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    m["actor"].load_state_dict(ck["actor"])
    m["backbone"].load_state_dict(ck["backbone"])
    print(f">> loaded checkpoint @ iter {ck.get('iter')} "
          f"(train latency {ck.get('mean_latency'):.2f} ms)")

    from module2_dt_control import HysteresisPolicy, HungarianPolicy, \
        MPCPolicy, RandomFeasiblePolicy

    def fresh_env(with_mpc):
        return HDTNParallelEnv(scen, sta, reward_cfg=RewardConfig(**cfg["reward"]),
                               horizon_s=args.horizon, dt_s=dt_s,
                               mpc_engine=mpc if with_mpc else None)

    rows = []
    pol = MAPPOPolicy(m["backbone"], m["actor"], env.gs_ids, env.ob,
                      mpc_engine=mpc, deterministic=True, device=device)
    rows.append(_run_policy(env, pol, args.horizon, dt_s, "MAPPO (ours)"))

    e = fresh_env(False)
    rows.append(_run_policy(e, GreedyNearestPolicy(threshold_ms=0.0),
                            args.horizon, dt_s, "Greedy nearest"))
    e = fresh_env(False)
    rows.append(_run_policy(e, HysteresisPolicy(margin_ms=20.0),
                            args.horizon, dt_s, "Hysteresis (20ms)"))
    e = fresh_env(False)
    rows.append(_run_policy(e, HungarianPolicy(e.gs_ids),
                            args.horizon, dt_s, "Hungarian (per-step opt)"))
    e = fresh_env(True)
    rows.append(_run_policy(e, MPCPolicy(mpc, e.gs_ids, e.core),
                            args.horizon, dt_s, "MPC lookahead"))
    e = fresh_env(False)
    rows.append(_run_policy(e, RandomFeasiblePolicy(migrate_prob=0.3),
                            args.horizon, dt_s, "Random-feasible"))

    print("\n" + "=" * 78)
    print(f"{'Scheme':<26}{'mean lat':>10}{'p95 lat':>10}{'migr/DT':>10}{'load var':>10}")
    print("-" * 78)
    for res, lv in rows:
        mpd = res.total_migrations / max(1, res.n_edge_dts)
        lvs = f"{lv:.4f}" if lv == lv else "  -"
        print(f"{res.scheme:<26}{res.mean_latency:>10.2f}{res.p95:>10.2f}"
              f"{mpd:>10.1f}{lvs:>10}")
    print("=" * 78)
    mappo, greedy = rows[0][0], rows[1][0]
    gap = mappo.mean_latency - greedy.mean_latency
    print(f"\nMAPPO vs Greedy latency gap: {gap:+.2f} ms "
          f"({100*gap/greedy.mean_latency:+.1f}%)")
    print(f"MAPPO load variance {rows[0][1]:.4f} vs Greedy {rows[1][1]:.4f} "
          f"({'LOWER=better balance' if rows[0][1] < rows[1][1] else 'higher'})")


if __name__ == "__main__":
    main()
