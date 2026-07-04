# Module 2 — Edge/Central digital-twin objects and the shared topology store they read.
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class EdgeDT:
    entity_id: str
    host_gs: str
    model_state: dict = field(default_factory=dict)
    being_migrated: bool = False

    @property
    def dt_id(self):
        return f"dt_{self.entity_id}"

    def clone_state(self):
        return dict(self.model_state)

    def update_report(self, **fields):
        self.model_state.update(fields)


class SharedTopologyStore:
    def __init__(self):
        self.reports = {}
        self.graph = nx.Graph()
        self.t = None

    def ingest(self, edge_dts, net, t):
        self.t = t
        self.reports = {dt.entity_id: dt.clone_state() for dt in edge_dts.values()}
        self.graph = net.G.copy() if net.G is not None else nx.Graph()

    def topology(self):
        return self.graph

    def hosts(self):
        return {eid: st.get("host_gs") for eid, st in self.reports.items()}


@dataclass
class CentralDT:
    dt_id: str
    purpose: str
    store: SharedTopologyStore

    def view(self):
        return self.store.topology()

    def entities(self):
        return list(self.store.reports.keys())


def spawn_central_dts(purposes, store):
    return {p: CentralDT(dt_id=f"central_{p}", purpose=p, store=store) for p in purposes}
