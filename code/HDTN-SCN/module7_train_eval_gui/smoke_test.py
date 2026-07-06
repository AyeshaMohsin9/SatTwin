# Module 7 — integration test: event bus, demo capture, and a short multi-worker train loop.
import os
import subprocess
import sys
import time

import torch

from module7_train_eval_gui.event_bus import EventBus
from module7_train_eval_gui.demo_capture import capture_demo
from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import MPCLookahead
from module6_mappo import build_mappo, MAPPOPolicy
from module2_dt_control import DTControlPlane

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def test_event_bus():
    run = _p("results", "_smoke_bus")
    bus = EventBus(run)
    bus.reset()
    for i in range(3):
        bus.log_metrics({"iter": i, "return": -i})
    bus.write_status({"iter": 2, "running": True})
    bus.write_demo({"frames": [], "iter": 2})
    assert len(bus.read_metrics()) == 3
    assert bus.read_status()["running"] is True
    assert bus.read_demo()["iter"] == 2
    print(f"[ok] event bus: 3 metrics logged + status + demo round-trip @ {run}")


def test_demo_capture():
    mpc = MPCLookahead(horizon=3)
    env = HDTNParallelEnv(_p("config", "con1.yaml"), _p("config", "stations.yaml"),
                          reward_cfg=RewardConfig(), horizon_s=600.0, dt_s=60.0,
                          mpc_engine=mpc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=dev)
    policy = MAPPOPolicy(m["backbone"], m["actor"], env.gs_ids, env.ob,
                         mpc_engine=mpc, deterministic=True, device=dev)
    demo = capture_demo(env, policy,
                        lambda: DTControlPlane(env.core, policy),
                        n_steps=5, max_sats=20)
    assert len(demo["frames"]) == 5
    assert len(demo["stations"]) == 24
    fr = demo["frames"][0]
    assert "sats" in fr and "migration_events" in fr and len(fr["sats"]) == 20
    print(f"[ok] demo capture: 5 frames, {len(demo['stations'])} stations, "
          f"20 sats/frame, keys ok")


def test_short_training():
    run = _p("results", "_smoke_train")
    cfg = _p("module7_train_eval_gui", "config", "mappo_con1.yaml")
    workers = min(2, torch.cuda.device_count() or 2)
    cmd = [sys.executable, "-m", "module7_train_eval_gui.train",
           "--config", cfg, "--run-dir", run, "--iterations", "3",
           "--workers", str(workers)]
    env = dict(os.environ)
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                          timeout=600, env=env)
    tail = "\n".join(proc.stdout.splitlines()[-8:])
    print("---- train stdout tail ----")
    print(tail)
    if proc.returncode != 0:
        print("---- stderr tail ----")
        print("\n".join(proc.stderr.splitlines()[-15:]))
    assert proc.returncode == 0, "training subprocess failed"
    bus = EventBus(run)
    rows = bus.read_metrics()
    assert len(rows) >= 3, f"expected >=3 metric rows, got {len(rows)}"
    assert bus.read_demo() is not None
    print(f"[ok] short training: {len(rows)} iters logged with {workers} workers, "
          f"final latency={rows[-1]['mean_latency']} ms "
          f"(greedy {rows[-1]['greedy_latency']} ms), demo written")


if __name__ == "__main__":
    test_event_bus()
    test_demo_capture()
    test_short_training()
    print("\nAll Module 7 integration tests passed.")
