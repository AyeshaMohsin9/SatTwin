# Module 5 — integration test: MPC-vs-truth, GNN inductivity, backbone fusion over Modules 1/4.
import os

import numpy as np
import torch

from module1_environment import HDTNEnvironment
from module4_marl_env import HDTNParallelEnv, RewardConfig
from module5_representation import (
    MPCLookahead, build_graph_tensors, GNNEncoder, RepresentationBackbone,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def test_mpc_matches_truth():
    core = HDTNEnvironment.from_files(_p("config", "con1.yaml"),
                                      _p("config", "stations.yaml"))
    core.reset()
    core.t = 300.0
    mpc = MPCLookahead(horizon=1, discount=1.0, migration_cost=0.0)
    gs_ids = list(core.ground.edge_stations().keys())
    sid = core.constellation.sat_ids[0]
    costs = mpc.action_costs(sid, core, gs_ids)
    core.net.build(300.0)
    truth_stay = core.net.ps_dt_latency(sid, core.edge_dt_host[sid])
    assert abs(costs[0] - truth_stay) < 1e-6, (costs[0], truth_stay)
    print(f"[ok] MPC horizon-1 'stay' cost = truth latency = {costs[0]:.4f} ms")


def test_gnn_inductive():
    enc = GNNEncoder(hidden=32, layers=2)
    outs = {}
    for con in ["con1", "con2"]:
        core = HDTNEnvironment.from_files(_p("config", f"{con}.yaml"),
                                          _p("config", "stations.yaml"))
        obs = core.reset()
        graph = build_graph_tensors(core.net, obs)
        emb = enc.encode(graph)
        outs[con] = (graph.num_nodes, emb.shape)
        assert emb.shape[0] == graph.num_nodes
        assert emb.shape[1] == 32
    print(f"[ok] GNN inductive: con1 {outs['con1']}, con2 {outs['con2']} "
          f"(same weights, different node counts)")


def test_backbone_fusion():
    env = HDTNParallelEnv(_p("config", "con1.yaml"), _p("config", "stations.yaml"),
                          reward_cfg=RewardConfig(), horizon_s=300.0, dt_s=60.0)
    obs, _ = env.reset()
    bb = RepresentationBackbone(local_dim=env.ob.local_dim(), gnn_hidden=64,
                                temporal_hidden=32, gnn_layers=2)
    agents = env.agents[:5]
    local = {a: obs[a]["observation"] for a in agents}
    feats = bb.batch_actor_features(agents, local, env.core.net, env._obs)
    cf = bb.critic_features(env.core.net, env._obs)
    assert feats.shape == (5, bb.actor_dim)
    assert cf.shape[0] == bb.critic_dim
    print(f"[ok] backbone: actor_features {tuple(feats.shape)} "
          f"(dim={bb.actor_dim}), critic_features dim={cf.shape[0]}")


def test_mpc_feature_into_env():
    mpc = MPCLookahead(horizon=3)
    env = HDTNParallelEnv(_p("config", "con1.yaml"), _p("config", "stations.yaml"),
                          reward_cfg=RewardConfig(), horizon_s=300.0, dt_s=60.0,
                          mpc_engine=mpc)
    obs, _ = env.reset()
    a = env.agents[0]
    dim = obs[a]["observation"].shape[0]
    expected = env.ob.local_dim() + mpc.feature_dim(env.ob.n_gs)
    assert dim == expected, (dim, expected)
    print(f"[ok] MPC feature wired into Module 4 env: obs_dim={dim} "
          f"(local {env.ob.local_dim()} + mpc {mpc.feature_dim(env.ob.n_gs)})")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_mpc_matches_truth()
    test_gnn_inductive()
    test_backbone_fusion()
    test_mpc_feature_into_env()
    print("\nAll Module 5 integration tests passed.")
