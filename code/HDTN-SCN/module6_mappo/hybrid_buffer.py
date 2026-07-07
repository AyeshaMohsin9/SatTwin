# Module 6 — hybrid rollout buffer: stores discrete gateway + continuous bandwidth actions with GAE.
import numpy as np
import torch


class HybridRolloutBuffer:
    def __init__(self, n_agents, feat_dim, n_actions, state_dim, device="cpu"):
        self.n_agents = n_agents
        self.feat_dim = feat_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.device = device
        self.clear()

    def clear(self):
        self.feats = []
        self.masks = []
        self.a_gw = []
        self.a_bw = []
        self.logps = []
        self.rewards = []
        self.local_r = []
        self.values = []
        self.state = []
        self.dones = []
        self.adv = None
        self.ret = None

    def add(self, feats, masks, a_gw, a_bw, logps, rewards, value, state, done,
            local_r=None):
        self.feats.append(feats)
        self.masks.append(masks)
        self.a_gw.append(a_gw)
        self.a_bw.append(a_bw)
        self.logps.append(logps)
        self.rewards.append(rewards)
        self.local_r.append(local_r if local_r is not None
                             else np.zeros(self.n_agents, dtype=np.float32))
        self.values.append(value)
        self.state.append(state)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        T = len(self.rewards)
        values = self.values + [last_value]
        team_r = [float(np.mean(r)) for r in self.rewards]
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(self.dones[t])
            delta = team_r[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + np.asarray(values[:-1], dtype=np.float32)
        self.adv = adv
        self.ret = ret
        return adv, ret

    def flatten(self):
        dev = self.device
        T = len(self.feats)
        feats = torch.stack([torch.as_tensor(np.stack(f), dtype=torch.float32)
                             for f in self.feats]).to(dev)
        masks = torch.stack([torch.as_tensor(np.stack(m)) for m in self.masks]).to(dev)
        a_gw = torch.stack([torch.as_tensor(np.stack(a), dtype=torch.long)
                            for a in self.a_gw]).to(dev)
        a_bw = torch.stack([torch.as_tensor(np.stack(a), dtype=torch.float32)
                            for a in self.a_bw]).to(dev)
        logps = torch.stack([torch.as_tensor(np.stack(lp), dtype=torch.float32)
                             for lp in self.logps]).to(dev)
        states = torch.as_tensor(np.stack(self.state), dtype=torch.float32).to(dev)
        adv = torch.as_tensor(self.adv, dtype=torch.float32).to(dev)
        ret = torch.as_tensor(self.ret, dtype=torch.float32).to(dev)
        A = self.n_agents
        local = np.stack(self.local_r).astype(np.float32)
        local_dev = local - local.mean(axis=1, keepdims=True)
        local_dev = torch.as_tensor(local_dev, dtype=torch.float32).to(dev)
        return {
            "feats": feats.reshape(T * A, -1),
            "masks": masks.reshape(T * A, -1),
            "a_gw": a_gw.reshape(T * A),
            "a_bw": a_bw.reshape(T * A),
            "logps": logps.reshape(T * A),
            "states": states,
            "adv_agent": adv.repeat_interleave(A),
            "local_dev": local_dev.reshape(T * A),
            "ret": ret,
            "T": T,
            "A": A,
        }
