# Module 6 — hybrid rollout collector: drives the env with a HybridActor (gateway + bandwidth-weight).
import numpy as np
import torch

from module4_marl_env.action_space import STAY, decode, decode_bw
from module4_marl_env.masking import feasible
from .hybrid_buffer import HybridRolloutBuffer


class HybridCollector:
    def __init__(self, env, backbone, actor, critic, device="cpu"):
        self.env = env
        self.backbone = backbone
        self.actor = actor
        self.critic = critic
        self.device = device
        self.agents = list(env.possible_agents)
        self.n_agents = len(self.agents)
        self.buffer = HybridRolloutBuffer(self.n_agents, backbone.actor_dim,
                                          env.n_act, env.ob.global_dim(), device)

    @torch.no_grad()
    def _forward(self, obs):
        local = {a: obs[a]["observation"] for a in self.agents}
        feats = self.backbone.batch_actor_features(
            self.agents, local, self.env.core.net, self.env._obs)
        masks = torch.as_tensor(
            np.stack([obs[a]["action_mask"] for a in self.agents]),
            device=self.device)
        a_gw, a_bw, logps, ent = self.actor.act(feats, masks)
        state = torch.as_tensor(self.env.state(), dtype=torch.float32,
                                device=self.device)
        value = self.critic(state.unsqueeze(0))
        return feats, masks, a_gw, a_bw, logps, state, float(value.item())

    def _apply(self, a_gw, a_bw, next_t):
        env = self.env
        env._prev_latency = dict(env._obs.latency)
        decoded = {}
        for i, a in enumerate(self.agents):
            act = int(a_gw[i].item())
            mask = feasible(a, env._obs, env.gs_ids)
            if act != STAY and not mask[act]:
                act = STAY
            gs = decode(act, env.gs_ids)
            if gs is not None:
                decoded[a] = gs
        bw_weights = {a: decode_bw(float(a_bw[i].item()))
                      for i, a in enumerate(self.agents)}
        applied = env.control.apply_actions(decoded)
        for a, gs in decoded.items():
            if env.control.edge_dts[a].host_gs == gs:
                env.ob.mark_migration(a, next_t)
        env.t = next_t
        next_obs, _, info = env.core.step({}, t=next_t, bw_weights=bw_weights)
        info["n_migrations"] = applied
        env._obs = next_obs
        env.control.store.ingest(env.control.edge_dts, env.core.net, next_t)
        rewards, breakdown = env.reward_fn.team_plus_shaping(
            next_obs, info, env._prev_latency)
        done = env.t >= env.horizon_s
        obs_dict = env._build_obs_dict(next_obs)
        if done:
            env.agents = []
        rewards = {a: float(rewards[a]) for a in self.agents}
        return obs_dict, rewards, done, breakdown, info

    def _local_reward(self, info):
        rc = self.env.reward_fn.cfg
        rates = info.get("rates", [])
        qcap = max(1e-6, rc.queue_scale)
        obs = self.env._obs
        out = np.zeros(self.n_agents, dtype=np.float32)
        for i, (a, r) in enumerate(zip(self.agents, rates)):
            q = obs.queue.get(a, 0.0) if getattr(obs, "queue", None) else 0.0
            out[i] = rc.w_sumrate * (float(r) / rc.rate_scale) - rc.w_queue * (q / qcap)
        return out

    def collect(self, n_steps):
        self.buffer.clear()
        obs, _ = self.env.reset()
        if not self.env.agents:
            obs, _ = self.env.reset()
        ep_return, ep_mig = 0.0, 0
        ep_sum, ep_min, ep_jain = [], [], []
        ep_drop, ep_exp, ep_aoi, ep_flat, ep_bat = 0.0, 0.0, [], 0, []
        for _ in range(n_steps):
            feats, masks, a_gw, a_bw, logps, state, value = self._forward(obs)
            next_t = self.env.t + self.env.dt_s
            next_obs, rewards, done, breakdown, info = self._apply(a_gw, a_bw, next_t)
            r_vec = np.asarray([rewards.get(a, 0.0) for a in self.agents],
                               dtype=np.float32)
            local_r = self._local_reward(info)
            self.buffer.add(
                feats.detach().cpu().numpy(),
                masks.detach().cpu().numpy(),
                a_gw.detach().cpu().numpy(),
                a_bw.detach().cpu().numpy(),
                logps.detach().cpu().numpy(),
                r_vec, value,
                state.detach().cpu().numpy(), done, local_r)
            if breakdown is not None:
                ep_return += float(np.mean(r_vec))
                ep_mig += breakdown.raw_migrations
                ep_sum.append(breakdown.raw_sum_rate)
                ep_min.append(breakdown.raw_min_rate)
                ep_jain.append(breakdown.raw_jain)
                ep_drop += breakdown.raw_dropped
                ep_exp += float(info.get("expired", 0.0))
                ep_aoi.append(float(info.get("mean_aoi", 0.0)))
                ep_flat += breakdown.raw_flat_battery
                ep_bat.append(float(info.get("mean_battery", 0.0)))
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
        last_value = self._forward(obs)[6] if self.env.agents else 0.0
        self.buffer.compute_gae(last_value)
        metrics = {
            "return": ep_return,
            "migrations": ep_mig,
            "sum_rate": float(np.mean(ep_sum)) if ep_sum else 0.0,
            "min_rate": float(np.mean(ep_min)) if ep_min else 0.0,
            "jain": float(np.mean(ep_jain)) if ep_jain else 0.0,
            "dropped": ep_drop,
            "expired": ep_exp,
            "mean_aoi": float(np.mean(ep_aoi)) if ep_aoi else 0.0,
            "flat_battery": ep_flat,
            "mean_battery": float(np.mean(ep_bat)) if ep_bat else 0.0,
        }
        return self.buffer, metrics
