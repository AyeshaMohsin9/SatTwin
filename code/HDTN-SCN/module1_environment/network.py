# Module 1 — time-varying ISL/GSL/terrestrial graph and PS-DT shortest-path latency.
import networkx as nx

from .geometry import prop_delay_ms, dist_km, visible, earth_blocked


class NetworkGraph:
    def __init__(self, constellation, ground, cfg):
        self.constellation = constellation
        self.ground = ground
        self.cfg = cfg
        self._t = None
        self._pos = None
        self.G = None

    def build(self, t):
        self._t = t
        self._pos = self.constellation.positions(t)
        G = nx.Graph()
        for sid, p in self._pos.items():
            G.add_node(sid, kind="sat", pos=p)
        for gid, gs in self.ground.stations.items():
            G.add_node(gid, kind="gs", pos=gs.ecef)
        self._add_isl(G)
        self._add_gsl(G)
        self._add_terrestrial(G)
        self.G = G
        return G

    def _add_isl(self, G):
        seen = set()
        for sid in self.constellation.sat_ids:
            neigh = (self.constellation.intra_plane_neighbors(sid)
                     + self.constellation.cross_plane_neighbors(sid))
            for nb in neigh:
                key = tuple(sorted((sid, nb)))
                if key in seen or nb == sid:
                    continue
                seen.add(key)
                d = dist_km(self._pos[sid], self._pos[nb])
                if d <= self.cfg.max_isl_range_km:
                    G.add_edge(sid, nb, delay=prop_delay_ms(self._pos[sid], self._pos[nb]),
                               link="isl")

    def _add_gsl(self, G):
        for gid, gs in self.ground.stations.items():
            if gs.is_ncc:
                continue
            gpos = gs.ecef
            for sid, spos in self._pos.items():
                if visible(spos, gpos, self.cfg.min_elevation_deg):
                    G.add_edge(gid, sid, delay=prop_delay_ms(spos, gpos), link="gsl")

    def _add_terrestrial(self, G):
        ids = list(self.ground.stations.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                G.add_edge(a, b, delay=self.ground.ping(a, b), link="terrestrial")

    def ps_dt_latency(self, sat_id, host_id):
        if sat_id == host_id:
            return 0.0
        try:
            return nx.shortest_path_length(self.G, sat_id, host_id, weight="delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def ps_dt_path(self, sat_id, host_id):
        try:
            return nx.shortest_path(self.G, sat_id, host_id, weight="delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def nearest_gs_latency(self, sat_id):
        best_gs, best_lat = None, float("inf")
        for gid, gs in self.ground.edge_stations().items():
            lat = self.ps_dt_latency(sat_id, gid)
            if lat < best_lat:
                best_gs, best_lat = gid, lat
        return best_gs, best_lat

    def sat_to_ncc_isl_latency(self, sat_id):
        sat_only = self.G.subgraph(self.constellation.sat_ids)
        best = float("inf")
        ncc_pos = self.ground.ncc.ecef
        for sid, spos in self._pos.items():
            if earth_blocked(spos, ncc_pos):
                continue
            down = prop_delay_ms(spos, ncc_pos)
            if sid == sat_id:
                best = min(best, down)
                continue
            try:
                up = nx.shortest_path_length(sat_only, sat_id, sid, weight="delay")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            best = min(best, up + down)
        return best

    def sat_to_ncc_terrestrial_latency(self, sat_id):
        gs, up = self.nearest_gs_latency(sat_id)
        if gs is None or up == float("inf"):
            return float("inf")
        return up + self.ground.ping(gs, self.ground.ncc_id)
