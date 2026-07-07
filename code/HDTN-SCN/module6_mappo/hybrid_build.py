# Module 6 — factory wiring env + Module 5 backbone + hybrid actor/critic/learner/collector.
from module5_representation import RepresentationBackbone
from .hybrid_actor import HybridActor
from .critic import CentralCritic
from .mappo_learner import MAPPOConfig
from .hybrid_learner import HybridMAPPOLearner
from .hybrid_collector import HybridCollector


def build_hybrid_mappo(env, mpc_engine=None, use_gnn=True, use_temporal=True,
                       gnn_hidden=64, gnn_layers=2, actor_hidden=128,
                       critic_hidden=256, cfg=None, device="cpu"):
    local_dim = env.ob.local_dim()
    if mpc_engine is not None:
        local_dim += mpc_engine.feature_dim(env.ob.n_gs)
    backbone = RepresentationBackbone(
        local_dim=local_dim, gnn_hidden=gnn_hidden, gnn_layers=gnn_layers,
        use_gnn=use_gnn, use_temporal=use_temporal).to(device)
    actor = HybridActor(
        feat_dim=backbone.actor_dim, n_actions=env.n_act,
        hidden=actor_hidden).to(device)
    critic = CentralCritic(
        state_dim=env.ob.global_dim(), gnn_dim=0,
        hidden=critic_hidden).to(device)
    learner = HybridMAPPOLearner(actor, critic, cfg or MAPPOConfig(), device=device)
    collector = HybridCollector(env, backbone, actor, critic, device=device)
    return {"backbone": backbone, "actor": actor, "critic": critic,
            "learner": learner, "collector": collector}
