# Module 4 — hybrid action: discrete gateway {stay} ∪ 𝒢 + continuous bandwidth weight.
STAY = 0
BW_MIN = 0.2
BW_MAX = 5.0


def decode_bw(raw):
    r = max(0.0, min(1.0, float(raw)))
    return BW_MIN + r * (BW_MAX - BW_MIN)


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
