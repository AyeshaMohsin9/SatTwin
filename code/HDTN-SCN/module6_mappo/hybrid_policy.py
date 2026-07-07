# Module 6 — inference adapter: wraps backbone+HybridActor as a Module-2 MigrationPolicy plus a bandwidth head.
import torch

from module2_dt_control.migration import MigrationPolicy
from module4_marl_env.masking import feasible
from module4_marl_env.action_space import decode, decode_bw


class HybridPolicy(MigrationPolicy):
    def __init__(self, backbone, actor, gs_ids, ob, mpc_engine=None,
                 deterministic=True, device="cpu"):
        self.backbone = backbone
        self.actor = actor
        self.gs_ids = gs_ids
        self.ob = ob
        self.mpc_engine = mpc_engine
        self.deterministic = deterministic
        self.device = device

    def reset(self):
        self.backbone.reset()

    def _feat(self, sat_id, core, obs):
        mpc_feat = self.mpc_engine.feature(sat_id, core, self.gs_ids) \
            if self.mpc_engine is not None else None
        local = self.ob.local_obs(sat_id, obs, mpc_preview=mpc_feat)
        return self.backbone.actor_features(sat_id, local, core.net, obs)

    @torch.no_grad()
    def decide(self, dt, obs, t):
        sat_id = dt.entity_id
        core = _core_from_obs(obs, self.ob)
        feat = self._feat(sat_id, core, obs)
        mask = torch.as_tensor(feasible(sat_id, obs, self.gs_ids),
                               device=feat.device).unsqueeze(0)
        a_gw, _, _, _ = self.actor.act(feat.unsqueeze(0), mask,
                                       deterministic=self.deterministic)
        return decode(int(a_gw.item()), self.gs_ids)

    @torch.no_grad()
    def bw_weights(self, env_core, obs):
        out = {}
        for sat_id in self.ob.sat_ids:
            feat = self._feat(sat_id, env_core, obs)
            mask = torch.as_tensor(feasible(sat_id, obs, self.gs_ids),
                                   device=feat.device).unsqueeze(0)
            _, a_bw, _, _ = self.actor.act(feat.unsqueeze(0), mask,
                                           deterministic=self.deterministic)
            out[sat_id] = decode_bw(float(a_bw.item()))
        return out


def _core_from_obs(obs, ob):
    return ob.env
