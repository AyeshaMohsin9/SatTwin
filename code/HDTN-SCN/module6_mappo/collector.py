# Module 6 — rollout collector: drives HDTNParallelEnv with actor+critic, fills the buffer.
import numpy as np
import torch

from module4_marl_env.action_space import STAY
from .rollout_buffer import RolloutBuffer


class Collector:
    def __init__(self, env, backbone, actor, critic, mpc_engine=None,
                 anchor_from_mpc=True, device="cpu"):
        self.env = env
        self.backbone = backbone
        self.actor = actor
        self.critic = critic
        self.mpc_engine = mpc_engine
        self.anchor_from_mpc = anchor_from_mpc
        self.device = device
        self.agents = list(env.possible_agents)
        self.n_agents = len(self.agents)
        self.buffer = RolloutBuffer(self.n_agents, backbone.actor_dim,
                                    env.n_act, env.ob.global_dim(), device)

    def _anchors(self, obs):
        n_act = self.env.n_act
        out = np.zeros((self.n_agents, n_act), dtype=np.float32)
        if not self.anchor_from_mpc or self.mpc_engine is None:
            return out
        for i, a in enumerate(self.agents):
            best = self.mpc_engine.best_action(a, self.env.core, self.env.gs_ids)
            out[i, best] = 1.0
        return out

    @torch.no_grad()
    def _forward(self, obs):
        local = {a: obs[a]["observation"] for a in self.agents}
        feats = self.backbone.batch_actor_features(
            self.agents, local, self.env.core.net, self.env._obs)
        masks = torch.as_tensor(
            np.stack([obs[a]["action_mask"] for a in self.agents]),
            device=self.device)
        anchors_np = self._anchors(obs)
        anchors = torch.as_tensor(anchors_np, device=self.device)
        actions, logps, _ = self.actor.act(feats, masks, anchors)
        state = torch.as_tensor(self.env.state(), dtype=torch.float32,
                                device=self.device)
        value = self.critic(state.unsqueeze(0))
        return feats, masks, anchors, actions, logps, state, float(value.item()), anchors_np

    def collect(self, n_steps):
        self.buffer.clear()
        obs, _ = self.env.reset()
        if not self.env.agents:
            obs, _ = self.env.reset()
        ep_return, ep_lat, ep_mig, ep_over = 0.0, [], 0, []
        for _ in range(n_steps):
            (feats, masks, anchors, actions, logps, state, value,
             anchors_np) = self._forward(obs)
            act_dict = {a: int(actions[i].item()) for i, a in enumerate(self.agents)}
            next_obs, rewards, term, trunc, infos = self.env.step(act_dict)
            done = all(term.get(a, False) or trunc.get(a, False)
                       for a in self.agents)
            r_vec = np.asarray([rewards.get(a, 0.0) for a in self.agents],
                               dtype=np.float32)
            self.buffer.add(
                feats.detach().cpu().numpy(),
                masks.detach().cpu().numpy(),
                anchors_np,
                actions.detach().cpu().numpy(),
                logps.detach().cpu().numpy(),
                r_vec, value,
                state.detach().cpu().numpy(), done)
            b = next(iter(infos.values()))["reward_breakdown"] if infos else None
            if b is not None:
                ep_return += float(np.mean(r_vec))
                ep_lat.append(b.raw_mean_latency)
                ep_mig += b.raw_migrations
                ep_over.append(b.raw_overload)
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
        last_value = self._forward(obs)[6] if self.env.agents else 0.0
        self.buffer.compute_gae(last_value)
        metrics = {
            "return": ep_return,
            "mean_latency": float(np.mean(ep_lat)) if ep_lat else float("nan"),
            "migrations": ep_mig,
            "mean_overload": float(np.mean(ep_over)) if ep_over else 0.0,
        }
        return self.buffer, metrics
