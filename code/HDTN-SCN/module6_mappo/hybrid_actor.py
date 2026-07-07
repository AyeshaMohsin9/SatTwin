# Module 6 — hybrid actor: masked discrete gateway head + Beta-distributed bandwidth head.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

NEG_INF = -1e9
EPS = 1e-6


class HybridActor(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.body = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.gw_head = nn.Linear(hidden, n_actions)
        self.bw_head = nn.Linear(hidden, 2)
        for m in list(self.body) + [self.gw_head]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.bw_head.weight, gain=1.0)
        nn.init.zeros_(self.bw_head.bias)

    def _dists(self, feat, mask, temperature=1.0):
        h = self.body(feat)
        logits = self.gw_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), NEG_INF)
        gw = Categorical(logits=logits / max(1e-3, temperature))
        ab = F.softplus(self.bw_head(h)) + 1.0
        bw = Beta(ab[..., 0], ab[..., 1])
        return gw, bw, ab

    @staticmethod
    def _clip01(x):
        return x.clamp(EPS, 1.0 - EPS)

    @staticmethod
    def _beta_point(ab):
        a = ab[..., 0]
        b = ab[..., 1]
        mode = (a - 1.0) / (a + b - 2.0).clamp(min=EPS)
        mean = a / (a + b)
        both_gt1 = (a > 1.0) & (b > 1.0)
        return torch.where(both_gt1, mode, mean)

    def act(self, feat, mask, deterministic=False, temperature=1.0):
        gw, bw, ab = self._dists(feat, mask, temperature=temperature)
        if deterministic:
            a_gw = gw.probs.argmax(dim=-1)
            a_bw = self._beta_point(ab)
        else:
            a_gw = gw.sample()
            a_bw = bw.sample()
        a_bw = self._clip01(a_bw)
        logp = gw.log_prob(a_gw) + bw.log_prob(a_bw)
        ent = gw.entropy() + bw.entropy()
        return a_gw, a_bw, logp, ent

    def eval_act(self, feat, mask, temperature=0.5):
        gw, bw, ab = self._dists(feat, mask, temperature=temperature)
        a_gw = gw.sample()
        a_bw = self._clip01(self._beta_point(ab))
        logp = gw.log_prob(a_gw) + bw.log_prob(a_bw)
        ent = gw.entropy() + bw.entropy()
        return a_gw, a_bw, logp, ent

    def evaluate(self, feat, mask, a_gw, a_bw):
        gw, bw, _ = self._dists(feat, mask)
        logp = gw.log_prob(a_gw) + bw.log_prob(self._clip01(a_bw))
        ent = gw.entropy() + bw.entropy()
        return logp, ent
