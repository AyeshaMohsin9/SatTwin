# Module 7 — persistent per-GPU worker processes that collect rollouts in parallel.
import os

import numpy as np
import torch
import torch.multiprocessing as mp

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import MPCLookahead
from module6_mappo import build_mappo


def _worker(rank, world, cfg, weight_q, result_q, stop_ev):
    torch.set_num_threads(max(1, cfg["cpu_threads"]))
    use_cuda = torch.cuda.is_available() and cfg["use_cuda"]
    device = f"cuda:{rank % torch.cuda.device_count()}" if use_cuda else "cpu"
    mpc = MPCLookahead(horizon=cfg["mpc_horizon"]) if cfg["use_mpc"] else None
    env = HDTNParallelEnv(cfg["scenario"], cfg["stations"],
                          reward_cfg=RewardConfig(**cfg["reward"]),
                          horizon_s=cfg["horizon_s"], dt_s=cfg["dt_s"],
                          mpc_engine=mpc)
    m = build_mappo(env, mpc_engine=mpc, use_gnn=cfg["use_gnn"],
                    use_temporal=cfg["use_temporal"], gnn_hidden=cfg["gnn_hidden"],
                    gnn_layers=cfg["gnn_layers"], device=device)
    backbone, actor, critic = m["backbone"], m["actor"], m["critic"]
    collector = m["collector"]
    while not stop_ev.is_set():
        payload = weight_q.get()
        if payload is None:
            break
        a_sd, c_sd, b_sd = payload
        actor.load_state_dict({k: v.to(device) for k, v in a_sd.items()})
        critic.load_state_dict({k: v.to(device) for k, v in c_sd.items()})
        backbone.load_state_dict({k: v.to(device) for k, v in b_sd.items()})
        buf, metrics = collector.collect(cfg["rollout_steps"])
        batch = buf.flatten()
        cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        result_q.put((rank, cpu_batch, metrics))


class ParallelCollector:
    def __init__(self, cfg, n_workers):
        self.cfg = cfg
        self.n_workers = n_workers
        self.ctx = mp.get_context("spawn")
        self.weight_qs = [self.ctx.Queue(maxsize=2) for _ in range(n_workers)]
        self.result_q = self.ctx.Queue()
        self.stop_ev = self.ctx.Event()
        self.procs = []

    def start(self):
        for rank in range(self.n_workers):
            p = self.ctx.Process(target=_worker, args=(
                rank, self.n_workers, self.cfg, self.weight_qs[rank],
                self.result_q, self.stop_ev))
            p.daemon = True
            p.start()
            self.procs.append(p)

    def broadcast(self, actor, critic, backbone):
        a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        b = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
        for q in self.weight_qs:
            q.put((a, c, b))

    def gather(self):
        batches, metrics = [], []
        for _ in range(self.n_workers):
            rank, batch, m = self.result_q.get()
            batches.append(batch)
            metrics.append(m)
        return batches, metrics

    def close(self):
        self.stop_ev.set()
        for q in self.weight_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def merge_batches(batches, device):
    keys = ["feats", "masks", "anchors", "actions", "logps", "states",
            "adv_agent", "adv_state", "ret"]
    out = {}
    for k in keys:
        out[k] = torch.cat([b[k] for b in batches], dim=0).to(device)
    out["T"] = sum(b["T"] for b in batches)
    out["A"] = batches[0]["A"]
    return out
