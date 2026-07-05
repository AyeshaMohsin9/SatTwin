# Module 6 — centralized value network over global (contention-aware) state for CTDE.
import torch
import torch.nn as nn


class CentralCritic(nn.Module):
    def __init__(self, state_dim, gnn_dim=0, hidden=256):
        super().__init__()
        in_dim = state_dim + gnn_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        self.gnn_dim = gnn_dim

    def forward(self, state, graph_embed=None):
        if self.gnn_dim > 0 and graph_embed is not None:
            state = torch.cat([state, graph_embed], dim=-1)
        return self.net(state).squeeze(-1)
