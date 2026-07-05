# Module 6 — factory wiring env + Module 5 backbone + actor/critic/learner/collector.
import torch

from module5_representation import RepresentationBackbone, MPCLookahead
from .actor import ResidualActor
from .critic import CentralCritic
from .mappo_learner import MAPPOLearner, MAPPOConfig
from .collector import Collector


def build_mappo(env, mpc_engine=None, use_gnn=True, use_temporal=True,
                gnn_hidden=64, temporal_hidden=32, gnn_layers=2,
                actor_hidden=128, critic_hidden=256, anchor_bias=3.0,
                cfg=None, device="cpu"):
    local_dim = env.ob.local_dim()
    if mpc_engine is not None:
        local_dim += mpc_engine.feature_dim(env.ob.n_gs)
    backbone = RepresentationBackbone(
        local_dim=local_dim, gnn_hidden=gnn_hidden, temporal_hidden=temporal_hidden,
        gnn_layers=gnn_layers, use_gnn=use_gnn, use_temporal=use_temporal).to(device)
    actor = ResidualActor(
        feat_dim=backbone.actor_dim, n_actions=env.n_act, hidden=actor_hidden,
        anchor_bias=anchor_bias).to(device)
    critic = CentralCritic(
        state_dim=env.ob.global_dim(), gnn_dim=0,
        hidden=critic_hidden).to(device)
    learner = MAPPOLearner(actor, critic, cfg or MAPPOConfig(), device=device)
    collector = Collector(env, backbone, actor, critic, mpc_engine=mpc_engine,
                          device=device)
    return {"backbone": backbone, "actor": actor, "critic": critic,
            "learner": learner, "collector": collector}
