# Module 4 — per-agent feasibility mask over {stay} ∪ 𝒢 (capacity, reachability, self).
import numpy as np

from .action_space import STAY


def feasible(sat_id, obs, gs_ids):
    mask = np.zeros(len(gs_ids) + 1, dtype=bool)
    mask[STAY] = True
    host = obs.host.get(sat_id)
    cand = obs.cand_latency.get(sat_id, {})
    for i, g in enumerate(gs_ids):
        lat = cand.get(g, float("inf"))
        reachable = lat != float("inf")
        not_full = not obs.overloaded(g)
        not_self = g != host
        mask[i + 1] = bool(reachable and not_full and not_self)
    return mask


def mask_matrix(sat_ids, obs, gs_ids):
    return {sid: feasible(sid, obs, gs_ids) for sid in sat_ids}
