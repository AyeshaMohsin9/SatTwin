# Module 4 — discrete migration action space {stay} ∪ 𝒢 with index<->GS mapping.
STAY = 0


def n_actions(n_gs):
    return n_gs + 1


def decode(action, gs_ids):
    if action == STAY:
        return None
    idx = action - 1
    if idx < 0 or idx >= len(gs_ids):
        return None
    return gs_ids[idx]


def encode(gs_id, gs_ids):
    if gs_id is None:
        return STAY
    if gs_id not in gs_ids:
        return STAY
    return gs_ids.index(gs_id) + 1
