# Module 6 — shared MAPPO actor: residual policy over the MPC anchor with masked logits.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NEG_INF = -1e9


class ResidualActor(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=128, anchor_bias=3.0,
                 mpc_slice=None):
        super().__init__()
        self.n_actions = n_actions
        self.anchor_bias = anchor_bias
        self.mpc_slice = mpc_slice
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def _anchor(self, feat, anchor_onehot):
        if anchor_onehot is not None:
            return anchor_onehot
        if self.mpc_slice is not None:
            return feat[..., self.mpc_slice]
        return torch.zeros(feat.shape[:-1] + (self.n_actions,), device=feat.device)

    def logits(self, feat, mask, anchor_onehot=None):
        residual = self.net(feat)
        anchor = self._anchor(feat, anchor_onehot)
        logits = residual + self.anchor_bias * anchor
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), NEG_INF)
        return logits

    def distribution(self, feat, mask, anchor_onehot=None):
        return Categorical(logits=self.logits(feat, mask, anchor_onehot))

    def act(self, feat, mask, anchor_onehot=None, deterministic=False):
        dist = self.distribution(feat, mask, anchor_onehot)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, feat, mask, actions, anchor_onehot=None):
        dist = self.distribution(feat, mask, anchor_onehot)
        return dist.log_prob(actions), dist.entropy()
