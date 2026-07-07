# Module 6 — smoke test 2: hybrid MAPPO on con_hard with return-norm, entropy anneal, and non-degenerate deterministic policy.
import os

import numpy as np
import torch

from module4_marl_env import HDTNParallelEnv, RewardConfig
from module6_mappo import build_hybrid_mappo, MAPPOConfig
from module4_marl_env.action_space import STAY

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _p(*parts):
    return os.path.join(HERE, *parts)


def _env():
    return HDTNParallelEnv(
        _p("config", "con_hard.yaml"), _p("config", "stations_congested.yaml"),
        reward_cfg=RewardConfig(objective="rate", w_sumrate=1.0, w_minrate=1.0,
                                w_jain=3.0, w_mig=0.5, w_queue=1.5, w_drop=2.5,
                                w_battery=4.0, w_shaping=0.4, rate_scale=400.0,
                                queue_scale=25.0),
        horizon_s=6000.0, dt_s=30.0)


def _det_gateway_actions(m, env, obs, temperature):
    agents = list(env.possible_agents)
    local = {a: obs[a]["observation"] for a in agents}
    feats = m["backbone"].batch_actor_features(agents, local, env.core.net, env._obs)
    masks = torch.as_tensor(
        np.stack([obs[a]["action_mask"] for a in agents]), device=DEV)
    with torch.no_grad():
        a_gw, a_bw, _, _ = m["actor"].act(feats, masks, deterministic=True,
                                          temperature=temperature)
    return a_gw.cpu().numpy(), a_bw.cpu().numpy(), masks.cpu().numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    env = _env()
    cfg = MAPPOConfig(epochs=3, minibatches=2, entropy_coef=0.01,
                      entropy_coef_final=0.001, entropy_anneal_iters=100,
                      local_credit=0.3)
    m = build_hybrid_mappo(env, gnn_hidden=32, gnn_layers=2, device=DEV, cfg=cfg)
    collector, learner = m["collector"], m["learner"]

    buf, metrics = collector.collect(n_steps=8)
    assert len(buf.rewards) == 8
    batch = buf.flatten()
    assert "local_dev" in batch
    for it in range(2):
        stats = learner.update(batch, iter=it, total_iters=100)
        assert np.isfinite(stats["policy_loss"]), stats
        assert np.isfinite(stats["value_loss"]), stats
        assert np.isfinite(stats["entropy"]), stats
        assert -0.05 <= stats["kl"] <= 0.1, f"kl out of range: {stats['kl']}"
        assert stats["value_loss"] < 1e4, f"value_loss too large: {stats['value_loss']}"
    print(f"[ok] update: pl={stats['policy_loss']:.4f} vl={stats['value_loss']:.4f} "
          f"ent={stats['entropy']:.4f} ent_coef={stats['entropy_coef']:.5f} "
          f"kl={stats['kl']:.4f}")

    for it in range(2, 8):
        buf, _ = collector.collect(n_steps=8)
        learner.update(buf.flatten(), iter=it, total_iters=100)

    obs, _ = env.reset()
    a_gw, a_bw, masks = _det_gateway_actions(m, env, obs, temperature=0.5)
    n_move = int((a_gw != STAY).sum())
    n_feasible_move = int((masks[:, 1:].sum(axis=1) > 0).sum())
    bw_std = float(np.std(a_bw))
    print(f"[det] gateway moves={n_move}/{len(a_gw)} "
          f"(feasible-move agents={n_feasible_move}), bw std={bw_std:.4f}, "
          f"bw range=[{a_bw.min():.3f},{a_bw.max():.3f}]")
    assert n_move > 0, "deterministic policy degenerate: all agents STAY"
    assert bw_std > 1e-4, f"deterministic bw weights degenerate (std={bw_std})"

    a_gw_hi, _, _ = _det_gateway_actions(m, env, obs, temperature=1.0)
    print(f"[ok] non-degenerate deterministic policy: {n_move} moves @ temp=0.5, "
          f"{int((a_gw_hi != STAY).sum())} moves @ temp=1.0")
    print("[pass] hybrid_smoke_test2")


if __name__ == "__main__":
    main()
