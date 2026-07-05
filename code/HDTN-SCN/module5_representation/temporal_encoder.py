# Module 5 — per-agent GRU over recent local-obs history for latent trend forecasting.
from collections import defaultdict, deque

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.out_dim = hidden

    def forward(self, seq):
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        out, h = self.gru(seq)
        return h[-1]

    def encode_last(self, seq):
        return self.forward(seq).squeeze(0)


class HistoryBuffer:
    def __init__(self, in_dim, length=8):
        self.in_dim = in_dim
        self.length = length
        self.buf = defaultdict(lambda: deque(maxlen=length))

    def push(self, agent, vec):
        self.buf[agent].append(torch.as_tensor(vec, dtype=torch.float32))

    def sequence(self, agent):
        h = self.buf[agent]
        if len(h) == 0:
            return torch.zeros(1, self.in_dim, dtype=torch.float32)
        pad = self.length - len(h)
        seq = list(h)
        if pad > 0:
            seq = [torch.zeros(self.in_dim, dtype=torch.float32)] * pad + seq
        return torch.stack(seq, dim=0)

    def reset(self, agent=None):
        if agent is None:
            self.buf.clear()
        else:
            self.buf[agent].clear()
