# Module 7 — multi-GPU MAPPO training loop: parallel rollout, PPO update, checkpoint, stream.
import argparse
import os
import time

import numpy as np
import torch
import yaml

from module4_marl_env import HDTNParallelEnv, RewardConfig, collect_greedy_traces
from module5_representation import MPCLookahead
from module6_mappo import build_mappo, MAPPOConfig, LagrangianDual, behavior_clone
from module3_simulation import run_scheme, HDTN_SCN

from .event_bus import EventBus
from .demo_capture import capture_demo
from .parallel_collector import ParallelCollector, merge_batches

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_worker_cfg(cfg):
    return {
        "scenario": _p("config", f"{cfg['con']}.yaml"),
        "stations": _p("config", cfg.get("stations_file", "stations.yaml")),
        "reward": cfg["reward"],
        "horizon_s": cfg["episode_horizon_s"],
        "dt_s": cfg["dt_s"],
        "rollout_steps": cfg["rollout_steps"],
        "mpc_horizon": cfg["mpc_horizon"],
        "use_mpc": cfg["use_mpc"],
        "use_gnn": cfg["use_gnn"],
        "use_temporal": cfg["use_temporal"],
        "gnn_hidden": cfg["gnn_hidden"],
        "gnn_layers": cfg["gnn_layers"],
        "use_cuda": cfg["use_cuda"],
        "cpu_threads": cfg["cpu_threads_per_worker"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_p("module7_train_eval_gui", "config",
                                           "mappo_con1.yaml"))
    ap.add_argument("--run-dir", default=_p("results", "mappo_run"))
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.iterations:
        cfg["iterations"] = args.iterations
    if args.hours:
        cfg["max_hours"] = args.hours
    if args.workers:
        cfg["n_workers"] = args.workers

    device = "cuda:0" if (cfg["use_cuda"] and torch.cuda.is_available()) else "cpu"
    n_workers = cfg["n_workers"]
    if cfg["use_cuda"] and torch.cuda.is_available():
        n_workers = min(n_workers, torch.cuda.device_count())

    bus = EventBus(args.run_dir)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    resuming = args.resume and os.path.exists(ckpt_path)
    if not resuming:
        bus.reset()
    os.makedirs(args.run_dir, exist_ok=True)

    mpc = MPCLookahead(horizon=cfg["mpc_horizon"]) if cfg["use_mpc"] else None
    master_env = HDTNParallelEnv(_p("config", f"{cfg['con']}.yaml"),
                                 _p("config", cfg.get("stations_file", "stations.yaml")),
                                 reward_cfg=RewardConfig(**cfg["reward"]),
                                 horizon_s=cfg["episode_horizon_s"],
                                 dt_s=cfg["dt_s"], mpc_engine=mpc)
    mcfg = MAPPOConfig(**cfg["mappo"])
    m = build_mappo(master_env, mpc_engine=mpc, use_gnn=cfg["use_gnn"],
                    use_temporal=cfg["use_temporal"], gnn_hidden=cfg["gnn_hidden"],
                    gnn_layers=cfg["gnn_layers"], anchor_bias=cfg.get("anchor_bias", 3.0),
                    cfg=mcfg, device=device)
    backbone, actor, critic, learner = (m["backbone"], m["actor"], m["critic"],
                                        m["learner"])
    lagr = LagrangianDual(**cfg["lagrangian"])

    start_iter = 0
    if resuming:
        ck = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        backbone.load_state_dict(ck["backbone"])
        start_iter = int(ck.get("iter", 0)) + 1
        print(f">> Resumed from checkpoint at iter {start_iter} "
              f"(latency {ck.get('mean_latency')})", flush=True)

    if cfg["warm_start"] and not resuming:
        print(">> Collecting greedy traces for warm-start...", flush=True)
        traces = collect_greedy_traces(
            _p("config", f"{cfg['con']}.yaml"), _p("config", cfg.get("stations_file", "stations.yaml")),
            horizon_s=cfg["episode_horizon_s"], dt_s=cfg["dt_s"],
            reward_cfg=RewardConfig(**cfg["reward"]))
        dim = backbone.actor_dim

        def feat_fn(tr):
            v = np.asarray(tr.obs, dtype=np.float32)
            if v.shape[0] < dim:
                v = np.concatenate([v, np.zeros(dim - v.shape[0], dtype=np.float32)])
            return v[:dim]

        bc = behavior_clone(actor, traces.transitions, feat_fn,
                            epochs=cfg["warm_start_epochs"], device=device)
        print(f">> Warm-start BC: {bc}", flush=True)

    print(">> Computing greedy baseline latency...", flush=True)
    greedy_res = run_scheme(_p("config", f"{cfg['con']}.yaml"),
                            _p("config", cfg.get("stations_file", "stations.yaml")), HDTN_SCN,
                            horizon_s=cfg["episode_horizon_s"],
                            threshold_ms=0.0, dt_s=cfg["dt_s"])
    greedy_latency = greedy_res.mean_latency
    print(f">> Greedy baseline mean latency = {greedy_latency:.2f} ms", flush=True)

    pc = ParallelCollector(build_worker_cfg(cfg), n_workers)
    pc.start()
    print(f">> Started {n_workers} rollout workers "
          f"(GPUs={torch.cuda.device_count() if torch.cuda.is_available() else 0})",
          flush=True)

    start = time.time()
    max_iters = cfg.get("iterations", 100000)
    max_hours = cfg.get("max_hours", 9999)
    best_latency = float("inf")
    try:
        for it in range(start_iter, max_iters):
            elapsed_h = (time.time() - start) / 3600.0
            if elapsed_h >= max_hours:
                print(f">> Reached max_hours={max_hours}", flush=True)
                break
            pc.broadcast(actor, critic, backbone)
            batches, wmetrics = pc.gather()
            batch = merge_batches(batches, device)
            stats = learner.update(batch)

            overload = float(np.mean([w["mean_overload"] for w in wmetrics]))
            beta = lagr.update(overload)

            mean_ret = float(np.mean([w["return"] for w in wmetrics]))
            mean_lat = float(np.nanmean([w["mean_latency"] for w in wmetrics]))
            mean_mig = float(np.mean([w["migrations"] for w in wmetrics]))
            sum_rate = float(np.mean([w.get("sum_rate", 0.0) for w in wmetrics]))
            min_rate = float(np.mean([w.get("min_rate", 0.0) for w in wmetrics]))
            jain = float(np.mean([w.get("jain", 0.0) for w in wmetrics]))
            record = {
                "iter": it,
                "elapsed_s": round(time.time() - start, 1),
                "return": round(mean_ret, 4),
                "mean_latency": round(mean_lat, 3),
                "greedy_latency": round(greedy_latency, 3),
                "migrations": round(mean_mig, 1),
                "overload": round(overload, 4),
                "beta": round(beta, 4),
                "sum_rate": round(sum_rate, 2),
                "min_rate": round(min_rate, 4),
                "jain": round(jain, 4),
                "policy_loss": round(stats["policy_loss"], 5),
                "value_loss": round(stats["value_loss"], 4),
                "entropy": round(stats["entropy"], 4),
                "kl": round(stats["kl"], 5),
                "clipfrac": round(stats["clipfrac"], 4),
                "samples": int(batch["feats"].shape[0]),
            }
            bus.log_metrics(record)
            bus.write_status({"iter": it, "elapsed_h": round(elapsed_h, 3),
                              "mean_latency": round(mean_lat, 2),
                              "greedy_latency": round(greedy_latency, 2),
                              "running": True})

            if it % cfg["log_every"] == 0:
                print(f"[{it:5d}] ret={mean_ret:7.2f} rate={sum_rate:6.1f} "
                      f"jain={jain:.3f} min={min_rate:.2f} mig={mean_mig:5.0f} "
                      f"lat={mean_lat:5.1f} ent={stats['entropy']:.3f} "
                      f"beta={beta:.2f} {elapsed_h:.2f}h", flush=True)

            if it % cfg["demo_every"] == 0:
                from module6_mappo import MAPPOPolicy
                from module2_dt_control import DTControlPlane
                policy = MAPPOPolicy(backbone, actor, master_env.gs_ids,
                                     master_env.ob, mpc_engine=mpc,
                                     deterministic=True, device=device)
                demo = capture_demo(master_env, policy,
                                    lambda: DTControlPlane(master_env.core, policy),
                                    n_steps=cfg["demo_steps"],
                                    dt_s=cfg.get("demo_dt_s", 150.0))
                demo["iter"] = it
                demo["mean_latency"] = mean_lat
                demo["greedy_latency"] = greedy_latency
                bus.write_demo(demo)

            if it % cfg["ckpt_every"] == 0 or mean_lat < best_latency:
                best_latency = min(best_latency, mean_lat)
                torch.save({"actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "backbone": backbone.state_dict(),
                            "iter": it, "mean_latency": mean_lat},
                           os.path.join(args.run_dir, "checkpoint.pt"))
    except KeyboardInterrupt:
        print(">> Interrupted", flush=True)
    finally:
        pc.close()
        bus.write_status({"running": False})
        print(">> Training stopped.", flush=True)


if __name__ == "__main__":
    main()
