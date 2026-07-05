# Module 5 — fuses GNN + temporal + MPC + raw obs into actor/critic feature tensors.
import torch
import torch.nn as nn

from .graph_builder import build_graph_tensors
from .gnn_encoder import GNNEncoder, global_pool
from .temporal_encoder import TemporalEncoder, HistoryBuffer


class RepresentationBackbone(nn.Module):
    def __init__(self, local_dim, gnn_hidden=64, temporal_hidden=32,
                 temporal_len=8, gnn_layers=3, use_gnn=True, use_temporal=True):
        super().__init__()
        self.local_dim = local_dim
        self.use_gnn = use_gnn
        self.use_temporal = use_temporal
        self.gnn = GNNEncoder(hidden=gnn_hidden, layers=gnn_layers) if use_gnn else None
        self.temporal = TemporalEncoder(local_dim, hidden=temporal_hidden) \
            if use_temporal else None
        self.history = HistoryBuffer(local_dim, length=temporal_len) \
            if use_temporal else None
        self.gnn_dim = gnn_hidden if use_gnn else 0
        self.temporal_dim = temporal_hidden if use_temporal else 0
        self.actor_dim = local_dim + self.gnn_dim + self.temporal_dim
        self.critic_dim = self.gnn_dim

    def reset(self):
        if self.history is not None:
            self.history.reset()

    def _graph_embed(self, net, obs):
        graph = build_graph_tensors(net, obs)
        node_emb = self.gnn.encode(graph)
        return graph, node_emb

    def actor_features(self, agent, local_vec, net, obs, node_cache=None):
        parts = [torch.as_tensor(local_vec, dtype=torch.float32)]
        if self.use_gnn:
            if node_cache is None:
                graph, node_emb = self._graph_embed(net, obs)
            else:
                graph, node_emb = node_cache
            idx = graph.node_index.get(agent)
            g = node_emb[idx] if idx is not None else torch.zeros(self.gnn_dim)
            parts.append(g)
        if self.use_temporal:
            self.history.push(agent, local_vec)
            parts.append(self.temporal.encode_last(self.history.sequence(agent)))
        return torch.cat(parts, dim=-1)

    def batch_actor_features(self, agents, local_vecs, net, obs):
        node_cache = self._graph_embed(net, obs) if self.use_gnn else None
        return torch.stack([
            self.actor_features(a, local_vecs[a], net, obs, node_cache)
            for a in agents], dim=0)

    def critic_features(self, net, obs):
        if not self.use_gnn:
            return torch.zeros(0)
        _, node_emb = self._graph_embed(net, obs)
        return global_pool(node_emb)
