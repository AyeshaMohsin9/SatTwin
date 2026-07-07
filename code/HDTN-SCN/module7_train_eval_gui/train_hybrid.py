# Module 7 — single-process hybrid MAPPO trainer: gateway + bandwidth-weight PPO, jsonl logging.
import argparse
import json
import os
import time

import torch
import yaml

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import MPCLookahead
from module6_mappo import build_hybrid_mappo, MAPPOConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_p("module7_train_eval_gui", "config",
                                           "mappo_hybrid.yaml"))
    ap.add_argument("--run-dir", default=_p("results", "hybrid_run"))
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    mpc = MPCLookahead(horizon=cfg["mpc_horizon"]) if cfg["use_mpc"] else None
    env = HDTNParallelEnv(_p("config", f"{cfg['con']}.yaml"),
                          _p("config", cfg.get("stations_file", "stations.yaml")),
                          reward_cfg=RewardConfig(**cfg["reward"]),
                          horizon_s=cfg["episode_horizon_s"],
                          dt_s=cfg["dt_s"], mpc_engine=mpc)
    mcfg = MAPPOConfig(**cfg["mappo"])
    m = build_hybrid_mappo(env, mpc_engine=mpc, use_gnn=cfg["use_gnn"],
                           use_temporal=cfg["use_temporal"],
                           gnn_hidden=cfg["gnn_hidden"], gnn_layers=cfg["gnn_layers"],
                           cfg=mcfg, device=device)
    collector, learner = m["collector"], m["learner"]
    backbone, actor, critic = m["backbone"], m["actor"], m["critic"]
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")

    def save_ckpt(it):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "backbone": backbone.state_dict(), "iter": it}, ckpt_path)

    print(f">> Hybrid MAPPO training on {cfg['con']} (device={device})", flush=True)
    start = time.time()
    ckpt_every = cfg.get("ckpt_every", 25)
    for it in range(args.iters):
        buf, cmetrics = collector.collect(cfg["rollout_steps"])
        batch = buf.flatten()
        stats = learner.update(batch, iter=it, total_iters=args.iters)
        if it % ckpt_every == 0:
            save_ckpt(it)
        record = {
            "iter": it,
            "elapsed_s": round(time.time() - start, 1),
            "sum_rate": round(cmetrics["sum_rate"], 2),
            "min_rate": round(cmetrics["min_rate"], 4),
            "jain": round(cmetrics["jain"], 4),
            "migrations": round(float(cmetrics["migrations"]), 1),
            "dropped": round(float(cmetrics["dropped"]), 1),
            "expired": round(float(cmetrics["expired"]), 1),
            "mean_aoi": round(float(cmetrics["mean_aoi"]), 2),
            "flat_battery": round(float(cmetrics["flat_battery"]), 1),
            "mean_battery": round(float(cmetrics["mean_battery"]), 3),
            "entropy": round(stats["entropy"], 4),
            "entropy_coef": round(stats["entropy_coef"], 5),
            "policy_loss": round(stats["policy_loss"], 5),
            "value_loss": round(stats["value_loss"], 4),
            "kl": round(stats["kl"], 5),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{it:4d}] rate={record['sum_rate']:7.1f} jain={record['jain']:.3f} "
              f"min={record['min_rate']:.3f} mig={record['migrations']:5.0f} "
              f"ent={record['entropy']:.3f} pl={record['policy_loss']:.4f} "
              f"vl={record['value_loss']:.3f} kl={record['kl']:.4f}", flush=True)
    save_ckpt(args.iters - 1)
    print(">> Hybrid training done.", flush=True)


if __name__ == "__main__":
    main()
