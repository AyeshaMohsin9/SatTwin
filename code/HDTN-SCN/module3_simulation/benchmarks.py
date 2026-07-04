# Module 3 — four PS-DT latency schemes (HDTN-SCN + Benchmarks 1/2/3) over Module 1.
HDTN_SCN = "HDTN-SCN"
BENCH1_NO_MIGRATION = "Benchmark-1"
BENCH2_CENTRAL_ISL = "Benchmark-2"
BENCH3_CENTRAL_TERRESTRIAL = "Benchmark-3"

ALL_SCHEMES = [HDTN_SCN, BENCH1_NO_MIGRATION, BENCH2_CENTRAL_ISL,
               BENCH3_CENTRAL_TERRESTRIAL]


class BenchmarkScheme:
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.fixed_host = {}

    def reset(self):
        self.fixed_host = {}
        self.env.net.build(0.0)
        for sid in self.env.constellation.sat_ids:
            gs, _ = self.env.net.nearest_gs_latency(sid)
            self.fixed_host[sid] = gs

    def latencies(self, t):
        self.env.net.build(t)
        if self.name == BENCH1_NO_MIGRATION:
            return self._bench1()
        if self.name == BENCH2_CENTRAL_ISL:
            return self._bench2()
        if self.name == BENCH3_CENTRAL_TERRESTRIAL:
            return self._bench3()
        return self._hdtn_scn()

    def _hdtn_scn(self):
        out = []
        for sid in self.env.constellation.sat_ids:
            _, lat = self.env.net.nearest_gs_latency(sid)
            out.append(lat)
        return out

    def _bench1(self):
        out = []
        for sid in self.env.constellation.sat_ids:
            out.append(self.env.net.ps_dt_latency(sid, self.fixed_host[sid]))
        return out

    def _bench2(self):
        return [self.env.net.sat_to_ncc_isl_latency(sid)
                for sid in self.env.constellation.sat_ids]

    def _bench3(self):
        return [self.env.net.sat_to_ncc_terrestrial_latency(sid)
                for sid in self.env.constellation.sat_ids]
