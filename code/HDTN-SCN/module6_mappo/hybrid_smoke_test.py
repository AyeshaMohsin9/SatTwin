# Module 6 — integration test for hybrid MAPPO: build, collect, PPO update, bandwidth-weight variation.
import os

import numpy as np
import torch

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import MPCLookahead
from module6_mappo import build_hybrid_mappo, MAPPOConfig, HybridPolicy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _p(*parts):
    return os.path.join(HERE, *parts)


def _env(mpc=None):
    return HDTNParallelEnv(
        _p("config", "con1_congested.yaml"), _p("config", "stations_congested.yaml"),
        reward_cfg=RewardConfig(objective="rate", w_sumrate=1.0, w_minrate=2.0,
                                w_jain=3.0, w_mig=0.2, w_shaping=0.5,
                                rate_scale=500.0, beta=0.0),
        horizon_s=6000.0, dt_s=30.0, mpc_engine=mpc)


def test_build_collect_update():
    mpc = MPCLookahead(horizon=4)
    env = _env(mpc=mpc)
    m = build_hybrid_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2,
                           device=DEV, cfg=MAPPOConfig(epochs=3, minibatches=2))
    buf, metrics = m["collector"].collect(n_steps=5)
    assert len(buf.rewards) == 5
    assert buf.adv is not None and buf.ret is not None
    batch = buf.flatten()
    stats = m["learner"].update(batch)
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    assert np.isfinite(stats["entropy"])
    print(f"[ok] hybrid build+collect+update (dev={DEV}): 5 steps, "
          f"return={metrics['return']:.3f}, sum_rate={metrics['sum_rate']:.2f}, "
          f"jain={metrics['jain']:.3f}, migrations={metrics['migrations']}")
    print(f"[ok] PPO update: policy_loss={stats['policy_loss']:.4f}, "
          f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}, "
          f"kl={stats['kl']:.4f}, clipfrac={stats['clipfrac']:.3f}")
    return m, env, mpc


def test_bw_weights_vary():
    mpc = MPCLookahead(horizon=4)
    env = _env(mpc=mpc)
    m = build_hybrid_mappo(env, mpc_engine=mpc, gnn_hidden=32, gnn_layers=2, device=DEV)
    obs, _ = env.reset()
    policy = HybridPolicy(m["backbone"], m["actor"], env.gs_ids, env.ob,
                          mpc_engine=mpc, stochastic=True, device=DEV)
    weights = policy.bw_weights(env.core, env._obs)
    vals = np.asarray(list(weights.values()), dtype=np.float32)
    assert vals.std() > 0.0, f"bw weights degenerate (std={vals.std()})"
    assert np.all((vals >= 0.2) & (vals <= 5.0))
    print(f"[ok] bw weights vary across {len(vals)} sats: "
          f"mean={vals.mean():.3f}, std={vals.std():.3f}, "
          f"range=[{vals.min():.3f}, {vals.max():.3f}]")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_build_collect_update()
    test_bw_weights_vary()
    print("\nAll Module 6 hybrid integration tests passed.")
