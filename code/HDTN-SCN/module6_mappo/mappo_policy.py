# Module 6 — inference adapter: wraps backbone+actor as a Module-2 MigrationPolicy.
import numpy as np
import torch

from module2_dt_control.migration import MigrationPolicy
from module4_marl_env.masking import feasible
from module4_marl_env.action_space import decode


class MAPPOPolicy(MigrationPolicy):
    def __init__(self, backbone, actor, gs_ids, ob, mpc_engine=None,
                 anchor_from_mpc=True, deterministic=True, device="cpu"):
        self.backbone = backbone
        self.actor = actor
        self.gs_ids = gs_ids
        self.ob = ob
        self.mpc_engine = mpc_engine
        self.anchor_from_mpc = anchor_from_mpc
        self.deterministic = deterministic
        self.device = device

    def reset(self):
        self.backbone.reset()

    @torch.no_grad()
    def decide(self, dt, obs, t):
        sat_id = dt.entity_id
        core = _core_from_obs(obs, self.ob)
        mpc_feat = self.mpc_engine.feature(sat_id, core, self.gs_ids) \
            if self.mpc_engine is not None else None
        local = self.ob.local_obs(sat_id, obs, mpc_preview=mpc_feat)
        feat = self.backbone.actor_features(sat_id, local, core.net, obs)
        mask = torch.as_tensor(feasible(sat_id, obs, self.gs_ids),
                               device=feat.device).unsqueeze(0)
        anchor = self._anchor(sat_id, core, feat.device)
        action, _, _ = self.actor.act(feat.unsqueeze(0), mask, anchor,
                                      deterministic=self.deterministic)
        return decode(int(action.item()), self.gs_ids)

    def _anchor(self, sat_id, core, device):
        if not self.anchor_from_mpc or self.mpc_engine is None:
            return None
        a = self.mpc_engine.best_action(sat_id, core, self.gs_ids)
        onehot = np.zeros(len(self.gs_ids) + 1, dtype=np.float32)
        onehot[a] = 1.0
        return torch.as_tensor(onehot, device=device).unsqueeze(0)


def _core_from_obs(obs, ob):
    return ob.env
