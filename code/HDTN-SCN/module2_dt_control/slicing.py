# Module 2 — Pillar B: QoS-aware synchronization slicing and slice-specific routing.
from dataclasses import dataclass

import networkx as nx

LOW_LATENCY = "low_latency"
HIGH_BW = "high_bw"
BEST_EFFORT = "best_effort"

DATA_TYPE_SLICE = {
    "fault_diagnosis": LOW_LATENCY,
    "control_instruction": LOW_LATENCY,
    "device_status": HIGH_BW,
    "mobility": HIGH_BW,
    "radio_resource": HIGH_BW,
    "model_data": HIGH_BW,
    "qoe": HIGH_BW,
    "ai_feedback": BEST_EFFORT,
    "optimization": BEST_EFFORT,
}


@dataclass
class Flow:
    src: str
    dst: str
    data_type: str

    @property
    def slice(self):
        return DATA_TYPE_SLICE.get(self.data_type, BEST_EFFORT)


class SlicingManager:
    def __init__(self, net):
        self.net = net

    def classify(self, flow):
        return flow.slice

    def route(self, flow):
        s = self.classify(flow)
        if s == LOW_LATENCY:
            return self._min_delay_path(flow.src, flow.dst)
        if s == HIGH_BW:
            return self._min_delay_path(flow.src, flow.dst)
        return self._min_isl_path(flow.src, flow.dst)

    def _min_delay_path(self, src, dst):
        try:
            return nx.shortest_path(self.net.G, src, dst, weight="delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _min_isl_path(self, src, dst):
        def w(u, v, d):
            return 1.0 if d.get("link") == "isl" else 50.0
        try:
            return nx.shortest_path(self.net.G, src, dst, weight=w)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def path_delay(self, path):
        if not path or len(path) < 2:
            return 0.0
        return sum(self.net.G[path[i]][path[i + 1]]["delay"]
                   for i in range(len(path) - 1))
