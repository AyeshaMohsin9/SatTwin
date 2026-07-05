# Module 5 — inductive edge-conditioned message-passing GNN over the constellation graph.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from .graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM


class EdgeConvLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__(aggr="mean")
        self.msg = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))
        self.upd = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0.0, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd(torch.cat([x, out], dim=-1))

    def message(self, x_j, edge_attr):
        return self.msg(torch.cat([x_j, edge_attr], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, hidden=64, layers=3, node_dim=NODE_FEAT_DIM,
                 edge_dim=EDGE_FEAT_DIM):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(
            [EdgeConvLayer(hidden, hidden, edge_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out_dim = hidden

    def forward(self, node_feat, edge_index, edge_attr):
        h = F.relu(self.input(node_feat))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.relu(conv(h, edge_index, edge_attr)))
        return h

    def encode(self, graph):
        return self.forward(graph.node_feat, graph.edge_index, graph.edge_attr)


def global_pool(node_emb):
    return node_emb.mean(dim=0)
