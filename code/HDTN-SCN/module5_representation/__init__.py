# Module 5 — representation: GNN + temporal encoders, MPC lookahead, fusion backbone.
from .mpc_lookahead import MPCLookahead
from .graph_builder import (
    build_graph_tensors, GraphTensors, NODE_FEAT_DIM, EDGE_FEAT_DIM,
    NODE_KINDS, EDGE_LINKS,
)
from .gnn_encoder import GNNEncoder, EdgeConvLayer, global_pool
from .temporal_encoder import TemporalEncoder, HistoryBuffer
from .backbone import RepresentationBackbone

__all__ = [
    "MPCLookahead", "build_graph_tensors", "GraphTensors", "NODE_FEAT_DIM",
    "EDGE_FEAT_DIM", "NODE_KINDS", "EDGE_LINKS", "GNNEncoder", "EdgeConvLayer",
    "global_pool", "TemporalEncoder", "HistoryBuffer", "RepresentationBackbone",
]
