# Module 7 — deterministic evaluation of trained hybrid MAPPO vs baselines on the stateful env.
import argparse
import json
import os

import numpy as np
import torch
import yaml

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module2_dt_control import (DTControlPlane, GreedyNearestPolicy, HysteresisPolicy,
                                RandomFeasiblePolicy)
from module6_mappo import build_hybrid_mappo, MAPPOConfig, HybridPolicy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def _reward_obj(env, info):
    b = env.reward_fn.compute(env._obs, info)
    return b.total


def rollout(env, policy, n_steps, dt_s, bw_fn=None):
    cp = DTControlPlane(env.core, policy)
    obs = cp.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    agg = {"sum_rate": [], "min_rate": [], "jain": [], "queue": [], "dropped": 0.0,
           "flat": 0, "mig": 0, "reward": []}
    for k in range(n_steps):
        actions = cp.decide_actions(obs, (k + 1) * dt_s)
        cp.apply_actions(actions)
        bw = bw_fn(env.core, obs) if bw_fn else None
        _, _, info = env.core.step({}, t=(k + 1) * dt_s, bw_weights=bw)
        obs = env.core.observe(env.core.t)
        cp.store.ingest(cp.edge_dts, env.core.net, env.core.t)
        agg["sum_rate"].append(info["sum_rate"])
        agg["min_rate"].append(info["min_rate"])
        agg["jain"].append(info["jain"])
        agg["queue"].append(info.get("mean_queue", 0.0))
        agg["dropped"] += info.get("dropped", 0.0)
        agg["flat"] += info.get("flat_battery", 0)
        agg["mig"] += info["n_migrations"]
        agg["reward"].append(env.reward_fn.compute(obs, info).total)
    return {
        "sum_rate": float(np.mean(agg["sum_rate"])),
        "min_rate": float(np.mean(agg["min_rate"])),
        "jain": float(np.mean(agg["jain"])),
        "queue": float(np.mean(agg["queue"])),
        "dropped": agg["dropped"],
        "flat_battery": agg["flat"],
        "migrations": agg["mig"],
        "reward": float(np.mean(agg["reward"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_p("module7_train_eval_gui", "config",
                                           "mappo_hard.yaml"))
    ap.add_argument("--ckpt", default=_p("results", "mappo_hard2", "checkpoint.pt"))
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt_s = cfg["dt_s"]
    scen = _p("config", f"{cfg['con']}.yaml")
    sta = _p("config", cfg.get("stations_file", "stations.yaml"))
    rcfg = RewardConfig(**cfg["reward"])

    def fresh():
        return HDTNParallelEnv(scen, sta, reward_cfg=rcfg,
                               horizon_s=cfg["episode_horizon_s"], dt_s=dt_s)

    rows = []

    env = fresh()
    m = build_hybrid_mappo(env, use_gnn=cfg["use_gnn"], use_temporal=cfg["use_temporal"],
                           gnn_hidden=cfg["gnn_hidden"], gnn_layers=cfg["gnn_layers"],
                           cfg=MAPPOConfig(**cfg["mappo"]), device=device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    m["actor"].load_state_dict(ck["actor"]); m["backbone"].load_state_dict(ck["backbone"])
    print(f">> loaded hybrid checkpoint @ iter {ck.get('iter')}")
    pol = HybridPolicy(m["backbone"], m["actor"], env.gs_ids, env.ob, device=device)
    rows.append(("MAPPO-hybrid (ours)", rollout(env, pol, args.steps, dt_s,
                                                bw_fn=pol.bw_weights)))

    for name, mk in [("Greedy nearest", lambda e: GreedyNearestPolicy(0.0)),
                     ("Hysteresis", lambda e: HysteresisPolicy(15.0)),
                     ("Static", lambda e: HysteresisPolicy(1e9)),
                     ("Random", lambda e: RandomFeasiblePolicy(0.5))]:
        e = fresh()
        rows.append((name, rollout(e, mk(e), args.steps, dt_s)))

    print("\n" + "=" * 96)
    hdr = f"{'Scheme':<22}{'reward':>9}{'sum_rate':>10}{'min_rate':>10}{'jain':>8}{'queue':>8}{'dropped':>10}{'migr':>8}"
    print(hdr); print("-" * 96)
    for name, r in rows:
        print(f"{name:<22}{r['reward']:>9.3f}{r['sum_rate']:>10.1f}{r['min_rate']:>10.3f}"
              f"{r['jain']:>8.3f}{r['queue']:>8.2f}{r['dropped']:>10.0f}{r['migrations']:>8d}")
    print("=" * 96)
    best_base = max(r["reward"] for n, r in rows[1:])
    ours = rows[0][1]["reward"]
    print(f"\nMAPPO-hybrid reward {ours:.3f} vs best baseline {best_base:.3f}  "
          f"({'WIN +' if ours > best_base else 'loss '}{ours-best_base:+.3f})")
    with open(_p("results", "hybrid_eval.json"), "w") as f:
        json.dump({n: r for n, r in rows}, f, indent=2)


if __name__ == "__main__":
    main()
