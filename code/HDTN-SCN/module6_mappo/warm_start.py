# Module 6 — behavior-clone the actor from greedy traces to remove cold-start thrash.
import numpy as np
import torch
import torch.nn.functional as F


def behavior_clone(actor, transitions, backbone_feat_fn, epochs=5, batch_size=256,
                   lr=1e-3, device="cpu"):
    feats, masks, actions = [], [], []
    for tr in transitions:
        f = backbone_feat_fn(tr)
        if f is None:
            continue
        feats.append(np.asarray(f, dtype=np.float32))
        masks.append(np.asarray(tr.mask, dtype=np.int8))
        actions.append(int(tr.action))
    if not feats:
        return {"bc_loss": float("nan"), "n": 0}
    feats = torch.as_tensor(np.stack(feats), device=device)
    masks = torch.as_tensor(np.stack(masks), device=device)
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = feats.shape[0]
    last = float("nan")
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logits = actor.logits(feats[mb], masks[mb])
            loss = F.cross_entropy(logits, actions[mb])
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
    return {"bc_loss": last, "n": n}
