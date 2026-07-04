# Module 3 — latency/migration metric accumulators and result containers.
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    scheme: str
    times: list = field(default_factory=list)
    mean_latency_series: list = field(default_factory=list)
    all_latencies: list = field(default_factory=list)
    total_migrations: int = 0
    n_edge_dts: int = 0
    horizon_s: float = 0.0

    def add_step(self, t, latencies, n_migrations=0):
        finite = [x for x in latencies if x != float("inf")]
        self.times.append(t)
        self.mean_latency_series.append(np.mean(finite) if finite else float("inf"))
        self.all_latencies.extend(finite)
        self.total_migrations += n_migrations

    @property
    def mean_latency(self):
        return float(np.mean(self.all_latencies)) if self.all_latencies else float("inf")

    @property
    def p50(self):
        return float(np.percentile(self.all_latencies, 50)) if self.all_latencies else float("inf")

    @property
    def p95(self):
        return float(np.percentile(self.all_latencies, 95)) if self.all_latencies else float("inf")

    def migration_frequency(self, window_s):
        if self.n_edge_dts == 0 or self.horizon_s == 0:
            return 0.0
        windows = self.horizon_s / window_s
        return self.total_migrations / self.n_edge_dts / windows

    def summary(self):
        return {
            "scheme": self.scheme,
            "mean_latency_ms": round(self.mean_latency, 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "total_migrations": self.total_migrations,
        }
