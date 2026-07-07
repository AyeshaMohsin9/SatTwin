# Module 6 — hybrid actor: masked discrete gateway head + continuous bandwidth-weight head.
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

NEG_INF = -1e9


class HybridActor(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=128, log_std_init=-0.5):
        super().__init__()
        self.n_actions = n_actions
        self.body = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.gw_head = nn.Linear(hidden, n_actions)
        self.bw_head = nn.Linear(hidden, 1)
        self.bw_log_std = nn.Parameter(torch.ones(1) * log_std_init)
        for m in list(self.body) + [self.gw_head, self.bw_head]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def _dists(self, feat, mask):
        h = self.body(feat)
        logits = self.gw_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), NEG_INF)
        gw = Categorical(logits=logits)
        mean = torch.sigmoid(self.bw_head(h)).squeeze(-1)
        std = torch.exp(self.bw_log_std).expand_as(mean)
        bw = Normal(mean, std)
        return gw, bw

    def act(self, feat, mask, deterministic=False):
        gw, bw = self._dists(feat, mask)
        if deterministic:
            a_gw = gw.probs.argmax(dim=-1)
            a_bw = bw.mean
        else:
            a_gw = gw.sample()
            a_bw = bw.sample()
        a_bw_c = a_bw.clamp(0.0, 1.0)
        logp = gw.log_prob(a_gw) + bw.log_prob(a_bw)
        ent = gw.entropy() + bw.entropy()
        return a_gw, a_bw_c, logp, ent

    def evaluate(self, feat, mask, a_gw, a_bw):
        gw, bw = self._dists(feat, mask)
        logp = gw.log_prob(a_gw) + bw.log_prob(a_bw)
        ent = gw.entropy() + bw.entropy()
        return logp, ent
