# Module 5 — converts Module 1 NetworkGraph.G (networkx) into torch tensors for the GNN.
import numpy as np
import torch

NODE_KINDS = {"sat": 0, "gs": 1, "ncc": 2}
EDGE_LINKS = {"isl": 0, "gsl": 1, "terrestrial": 2}
N_NODE_KIND = len(NODE_KINDS)
N_EDGE_LINK = len(EDGE_LINKS)
NODE_FEAT_DIM = N_NODE_KIND + 3
EDGE_FEAT_DIM = N_EDGE_LINK + 1


class GraphTensors:
    def __init__(self, node_index, node_feat, edge_index, edge_attr, kind):
        self.node_index = node_index
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.kind = kind

    @property
    def num_nodes(self):
        return self.node_feat.shape[0]


def build_graph_tensors(net, obs, gs_capacity=1.0, delay_scale=100.0):
    G = net.G
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    feats = np.zeros((len(nodes), NODE_FEAT_DIM), dtype=np.float32)
    kinds = np.zeros(len(nodes), dtype=np.int64)
    for n, i in node_index.items():
        data = G.nodes[n]
        kind = data.get("kind", "sat")
        if kind == "gs" and getattr(net.ground.stations.get(n, None), "is_ncc", False):
            kind = "ncc"
        k = NODE_KINDS.get(kind, 0)
        kinds[i] = k
        feats[i, k] = 1.0
        pos = data.get("pos")
        if pos is not None:
            feats[i, N_NODE_KIND] = float(np.linalg.norm(pos)) / 1e4
        if kind in ("gs", "ncc") and obs is not None:
            cap = max(1, obs.gs_capacity.get(n, 1))
            feats[i, N_NODE_KIND + 1] = obs.gs_load.get(n, 0) / cap
        if kind == "sat" and obs is not None:
            lat = obs.latency.get(n, float("inf"))
            feats[i, N_NODE_KIND + 2] = 0.0 if lat == float("inf") \
                else min(1.0, lat / delay_scale)

    src, dst, eattr = [], [], []
    for u, v, d in G.edges(data=True):
        iu, iv = node_index[u], node_index[v]
        link = EDGE_LINKS.get(d.get("link", "isl"), 0)
        w = min(1.0, d.get("delay", 0.0) / delay_scale)
        e = [0.0] * EDGE_FEAT_DIM
        e[link] = 1.0
        e[N_EDGE_LINK] = w
        for a, b in ((iu, iv), (iv, iu)):
            src.append(a); dst.append(b); eattr.append(list(e))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32) if eattr \
        else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
    return GraphTensors(node_index, torch.from_numpy(feats), edge_index,
                        edge_attr, torch.from_numpy(kinds))
