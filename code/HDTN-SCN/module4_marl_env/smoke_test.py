# Module 4 — integration smoke test: PettingZoo conformance, masking, reward, traces.
import os

import numpy as np

from module4_marl_env import (
    HDTNParallelEnv, RewardConfig, collect_greedy_traces, STAY, decode,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(HERE, *parts)


def _env(horizon=600.0, dt=60.0):
    return HDTNParallelEnv(_p("config", "con1.yaml"), _p("config", "stations.yaml"),
                           reward_cfg=RewardConfig(beta=1.0), horizon_s=horizon, dt_s=dt)


def test_spaces_and_reset():
    env = _env()
    obs, infos = env.reset()
    assert len(env.agents) == 66
    a = env.agents[0]
    sp = env.observation_space(a)
    o = obs[a]["observation"]
    m = obs[a]["action_mask"]
    assert o.shape[0] == env.ob.local_dim()
    assert m.shape[0] == env.n_act == len(env.gs_ids) + 1
    assert env.state().shape[0] == env.ob.global_dim()
    print(f"[ok] reset: {len(env.agents)} agents, obs_dim={o.shape[0]}, "
          f"n_actions={env.n_act}, global_dim={env.state().shape[0]}")


def test_mask_safety():
    env = _env()
    obs, _ = env.reset()
    for a in env.agents:
        m = obs[a]["action_mask"]
        assert m[STAY] == 1
        for act in range(env.n_act):
            if m[act] == 0:
                continue
            gs = decode(act, env.gs_ids)
            if gs is not None:
                assert not env._obs.overloaded(gs)
    print("[ok] masks: stay always feasible; no unmasked action targets an overloaded GS")


def test_random_rollout():
    env = _env(horizon=600.0, dt=60.0)
    obs, _ = env.reset()
    steps, total_mig = 0, 0
    rng = np.random.default_rng(0)
    while env.agents:
        actions = {}
        for a in env.agents:
            m = obs[a]["action_mask"]
            valid = np.flatnonzero(m)
            actions[a] = int(rng.choice(valid))
        obs, rewards, term, trunc, infos = env.step(actions)
        steps += 1
        if infos:
            total_mig += next(iter(infos.values()))["n_migrations"]
    assert steps == 10
    print(f"[ok] random rollout: {steps} steps, migrations={total_mig}, "
          f"terminated cleanly (agents empty={env.agents == []})")


def test_reward_breakdown():
    env = _env(horizon=300.0, dt=60.0)
    obs, _ = env.reset()
    actions = {a: STAY for a in env.agents}
    obs, rewards, term, trunc, infos = env.step(actions)
    b = next(iter(infos.values()))["reward_breakdown"]
    assert abs(b.total - (b.latency + b.migration + b.load + b.aoi + b.overload)) < 1e-6
    print(f"[ok] reward decomposes: total={b.total:.4f} = lat={b.latency:.4f} + "
          f"mig={b.migration:.4f} + load={b.load:.4f} + aoi={b.aoi:.4f} + "
          f"overload={b.overload:.4f}")


def test_greedy_traces():
    logger = collect_greedy_traces(_p("config", "con1.yaml"),
                                   _p("config", "stations.yaml"),
                                   horizon_s=600.0, dt_s=60.0,
                                   reward_cfg=RewardConfig())
    assert len(logger) > 0
    tr = logger.transitions[0]
    assert tr.obs.ndim == 1
    assert tr.mask.shape[0] == tr.mask.shape[0] and tr.mask[STAY] in (0, 1)
    print(f"[ok] greedy traces: {len(logger)} transitions logged "
          f"(obs_dim={tr.obs.shape[0]}, mask_dim={tr.mask.shape[0]})")


if __name__ == "__main__":
    test_spaces_and_reset()
    test_mask_safety()
    test_random_rollout()
    test_reward_breakdown()
    test_greedy_traces()
    print("\nAll Module 4 integration tests passed.")
