# Module 6 — multi-agent rollout buffer with shared-critic GAE-lambda advantages.
import numpy as np
import torch


class RolloutBuffer:
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
        self.anchors = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.state = []
        self.dones = []
        self.adv = None
        self.ret = None

    def add(self, feats, masks, anchors, actions, logps, rewards, value, state, done):
        self.feats.append(feats)
        self.masks.append(masks)
        self.anchors.append(anchors)
        self.actions.append(actions)
        self.logps.append(logps)
        self.rewards.append(rewards)
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
        anchors = torch.stack([torch.as_tensor(np.stack(a), dtype=torch.float32)
                               for a in self.anchors]).to(dev)
        actions = torch.stack([torch.as_tensor(np.stack(a), dtype=torch.long)
                               for a in self.actions]).to(dev)
        logps = torch.stack([torch.as_tensor(np.stack(lp), dtype=torch.float32)
                             for lp in self.logps]).to(dev)
        states = torch.as_tensor(np.stack(self.state), dtype=torch.float32).to(dev)
        adv = torch.as_tensor(self.adv, dtype=torch.float32).to(dev)
        ret = torch.as_tensor(self.ret, dtype=torch.float32).to(dev)
        A = self.n_agents
        return {
            "feats": feats.reshape(T * A, -1),
            "masks": masks.reshape(T * A, -1),
            "anchors": anchors.reshape(T * A, -1),
            "actions": actions.reshape(T * A),
            "logps": logps.reshape(T * A),
            "states": states,
            "adv_agent": adv.repeat_interleave(A),
            "adv_state": adv,
            "ret": ret,
            "T": T,
            "A": A,
        }
