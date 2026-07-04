# Module 2 — Pillar C: locator-identifier isolation via the paper's four mapping tables.
from dataclasses import dataclass, field


class MappingTable:
    def __init__(self, name):
        self.name = name
        self.id_to_locator = {}

    def resolve(self, obj_id):
        return self.id_to_locator.get(obj_id)

    def update(self, obj_id, new_locator):
        self.id_to_locator[obj_id] = new_locator

    def remove(self, obj_id):
        self.id_to_locator.pop(obj_id, None)


@dataclass
class AddressingSystem:
    gs_dt_to_ps: MappingTable = field(default_factory=lambda: MappingTable("gs_dt_to_ps"))
    ps_edgedt_to_gs: MappingTable = field(default_factory=lambda: MappingTable("ps_edgedt_to_gs"))
    ncc_ps_to_dt: MappingTable = field(default_factory=lambda: MappingTable("ncc_ps_to_dt"))
    ncc_dt_to_gs: MappingTable = field(default_factory=lambda: MappingTable("ncc_dt_to_gs"))

    def register(self, dt):
        self.gs_dt_to_ps.update(dt.dt_id, dt.entity_id)
        self.ps_edgedt_to_gs.update(dt.entity_id, dt.host_gs)
        self.ncc_ps_to_dt.update(dt.entity_id, dt.dt_id)
        self.ncc_dt_to_gs.update(dt.dt_id, dt.host_gs)

    def on_migration(self, dt, new_gs):
        self.ps_edgedt_to_gs.update(dt.entity_id, new_gs)
        self.ncc_dt_to_gs.update(dt.dt_id, new_gs)

    def locate_dt_from_ps(self, entity_id):
        dt_id = self.ncc_ps_to_dt.resolve(entity_id)
        return dt_id, self.ncc_dt_to_gs.resolve(dt_id) if dt_id else None

    def locate_ps_from_dt(self, dt_id):
        return self.gs_dt_to_ps.resolve(dt_id)

    def consistent(self, edge_dts):
        for dt in edge_dts.values():
            if self.ps_edgedt_to_gs.resolve(dt.entity_id) != dt.host_gs:
                return False
            if self.ncc_dt_to_gs.resolve(dt.dt_id) != dt.host_gs:
                return False
        return True
