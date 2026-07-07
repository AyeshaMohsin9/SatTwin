# Module 6 — hybrid MAPPO learner: PPO-clip over joint gateway+bandwidth actor with clipped-value critic.
import numpy as np
import torch
import torch.nn as nn


class HybridMAPPOLearner:
    def __init__(self, actor, critic, cfg, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.cfg = cfg
        self.device = device
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    def update(self, batch, state_graph_embed=None):
        c = self.cfg
        feats = batch["feats"]; masks = batch["masks"]
        a_gw = batch["a_gw"]; a_bw = batch["a_bw"]; old_logps = batch["logps"]
        adv_agent = batch["adv_agent"]
        ret = batch["ret"]; states = batch["states"]
        T, A = batch["T"], batch["A"]

        if c.normalize_adv:
            adv_agent = (adv_agent - adv_agent.mean()) / (adv_agent.std() + 1e-8)

        with torch.no_grad():
            old_values = self.critic(states, state_graph_embed)

        n = feats.shape[0]
        idx = np.arange(n)
        mb_size = max(1, n // c.minibatches)
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0, "clipfrac": 0}
        count = 0
        stop = False
        for _ in range(c.epochs):
            if stop:
                break
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                logp, ent = self.actor.evaluate(
                    feats[mb_t], masks[mb_t], a_gw[mb_t], a_bw[mb_t])
                ratio = torch.exp(logp - old_logps[mb_t])
                a = adv_agent[mb_t]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy = ent.mean()

                approx_kl = (old_logps[mb_t] - logp).mean().item()
                clipfrac = ((ratio - 1.0).abs() > c.clip).float().mean().item()

                self.opt_actor.zero_grad()
                (policy_loss - c.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.opt_actor.step()

                stats["policy_loss"] += policy_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                count += 1
                if approx_kl > c.target_kl:
                    stop = True
                    break

        state_idx = np.arange(T)
        vmb = max(1, T // c.minibatches)
        vcount = 0
        for _ in range(c.epochs):
            np.random.shuffle(state_idx)
            for start in range(0, T, vmb):
                mb = torch.as_tensor(state_idx[start:start + vmb],
                                     dtype=torch.long, device=self.device)
                values = self.critic(states[mb], state_graph_embed)
                v_clipped = old_values[mb] + torch.clamp(
                    values - old_values[mb], -c.value_clip, c.value_clip)
                vl1 = (values - ret[mb]) ** 2
                vl2 = (v_clipped - ret[mb]) ** 2
                value_loss = c.value_coef * torch.max(vl1, vl2).mean()
                self.opt_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.opt_critic.step()
                stats["value_loss"] += value_loss.item()
                vcount += 1

        for k in ("policy_loss", "entropy", "kl", "clipfrac"):
            stats[k] /= max(1, count)
        stats["value_loss"] /= max(1, vcount)
        return stats
