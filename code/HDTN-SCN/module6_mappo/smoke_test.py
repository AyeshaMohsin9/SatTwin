# Module 6 — integration test: build, warm-start, collect, PPO update, MAPPOPolicy drop-in.
import os

import numpy as np
import torch

from module4_marl_env import HDTNParallelEnv, RewardConfig, collect_greedy_traces
from module5_representation import MPCLookahead
from module6_mappo import build_mappo, MAPPOConfig, MAPPOPolicy, behavior_clone
from module2_dt_control import DTControlPlane
from module3_simulation.metrics import RunResult

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _p(*parts):
    return os.path.join(HERE, *parts)


def _env(horizon=300.0, dt=60.0, mpc=None):
    return HDTNParallelEnv(_p("config", "con1.yaml"), _p("config", "stations.yaml"),
                           reward_cfg=RewardConfig(beta=1.0), horizon_s=horizon,
                           dt_s=dt, mpc_engine=mpc)


def test_build_and_collect():
    mpc = MPCLookahead(horizon=3)
    env = _env(mpc=mpc)
    m = build_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=DEV,
                    cfg=MAPPOConfig(epochs=2, minibatches=2))
    buf, metrics = m["collector"].collect(n_steps=5)
    assert len(buf.rewards) == 5
    assert buf.adv is not None and buf.ret is not None
    print(f"[ok] build+collect (dev={DEV}): 5 steps, return={metrics['return']:.3f}, "
          f"mean_latency={metrics['mean_latency']:.2f} ms, migrations={metrics['migrations']}")
    return m, env, mpc


def test_learner_update():
    mpc = MPCLookahead(horizon=3)
    env = _env(mpc=mpc)
    m = build_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=DEV,
                    cfg=MAPPOConfig(epochs=3, minibatches=2))
    buf, _ = m["collector"].collect(n_steps=6)
    batch = buf.flatten()
    stats = m["learner"].update(batch)
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    print(f"[ok] PPO update: policy_loss={stats['policy_loss']:.4f}, "
          f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}, "
          f"kl={stats['kl']:.4f}, clipfrac={stats['clipfrac']:.3f}")


def test_warm_start():
    mpc = MPCLookahead(horizon=3)
    env = _env(mpc=mpc)
    m = build_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=DEV)
    traces = collect_greedy_traces(_p("config", "con1.yaml"),
                                   _p("config", "stations.yaml"),
                                   horizon_s=180.0, dt_s=60.0,
                                   reward_cfg=RewardConfig())

    def feat_fn(tr):
        return np.asarray(tr.obs, dtype=np.float32)

    dim = m["backbone"].actor_dim
    def feat_fn_padded(tr):
        v = np.asarray(tr.obs, dtype=np.float32)
        if v.shape[0] < dim:
            v = np.concatenate([v, np.zeros(dim - v.shape[0], dtype=np.float32)])
        return v[:dim]

    stats = behavior_clone(m["actor"], traces.transitions, feat_fn_padded,
                           epochs=2, device=DEV)
    assert stats["n"] > 0 and np.isfinite(stats["bc_loss"])
    print(f"[ok] warm-start BC: {stats['n']} samples, final loss={stats['bc_loss']:.4f}")


def test_policy_dropin():
    mpc = MPCLookahead(horizon=3)
    env = _env(mpc=mpc)
    m = build_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=DEV)
    policy = MAPPOPolicy(m["backbone"], m["actor"], env.gs_ids, env.ob,
                         mpc_engine=mpc, deterministic=True, device=DEV)
    cp = DTControlPlane(env.core, policy)
    obs = cp.reset()
    res = RunResult(scheme="MAPPO", n_edge_dts=len(cp.edge_dts), horizon_s=300.0)
    for k in range(5):
        obs, reward, info = cp.step(obs, t=(k + 1) * cp.env.cfg.time_step_s)
        res.add_step(cp.env.t, info["latencies"], info["n_migrations"])
        assert cp.addressing_consistent()
    print(f"[ok] MAPPOPolicy drop-in via Module 2/3: mean_latency={res.mean_latency:.2f} ms, "
          f"migrations={res.total_migrations}, addressing consistent")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_build_and_collect()
    test_learner_update()
    test_warm_start()
    test_policy_dropin()
    print("\nAll Module 6 integration tests passed.")
