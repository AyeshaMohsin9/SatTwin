# Module 4 — HDTNParallelEnv: PettingZoo ParallelEnv wrapping Modules 1+2 as a Dec-POMDP.
import functools

import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as DictSpace
from pettingzoo import ParallelEnv

from module1_environment import HDTNEnvironment, load_scenario
from module2_dt_control import DTControlPlane, GreedyNearestPolicy

from .action_space import n_actions, decode, encode, STAY
from .masking import feasible
from .obs_builder import ObsBuilder, BIG
from .reward import RewardConfig, RewardFunction
from .trace_logger import TraceLogger


class HDTNParallelEnv(ParallelEnv):
    metadata = {"name": "hdtn_scn_v0", "is_parallelizable": True}

    def __init__(self, scenario_path, stations_path, ping_csv=None,
                 reward_cfg=None, horizon_s=6000.0, dt_s=None, mpc_engine=None,
                 seed=None):
        self.scenario_path = scenario_path
        self.stations_path = stations_path
        self.ping_csv = ping_csv
        self.horizon_s = horizon_s
        self._seed = seed
        self.mpc_engine = mpc_engine

        self.core = HDTNEnvironment.from_files(scenario_path, stations_path, ping_csv)
        self.dt_s = dt_s or self.core.cfg.time_step_s
        self.control = DTControlPlane(self.core, GreedyNearestPolicy(threshold_ms=0.0))

        self.ob = ObsBuilder(self.core)
        self.gs_ids = self.ob.gs_ids
        self.n_act = n_actions(self.ob.n_gs)
        self.reward_fn = RewardFunction(reward_cfg or RewardConfig(), self.ob)

        self.possible_agents = list(self.core.constellation.sat_ids)
        self.agents = list(self.possible_agents)
        self._obs = None
        self._prev_latency = None
        self.t = 0.0
        self._mpc_dim = 0
        if mpc_engine is not None:
            self._mpc_dim = mpc_engine.feature_dim(self.ob.n_gs)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        dim = self.ob.local_dim() + self._mpc_dim
        return DictSpace({
            "observation": Box(low=-BIG, high=BIG, shape=(dim,), dtype=np.float32),
            "action_mask": Box(low=0, high=1, shape=(self.n_act,), dtype=np.int8),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.n_act)

    def state_space(self):
        return Box(low=-BIG, high=BIG, shape=(self.ob.global_dim(),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._obs = self.control.reset()
        self.ob.reset()
        self.t = 0.0
        self._prev_latency = dict(self._obs.latency)
        obs_dict = self._build_obs_dict(self._obs)
        infos = {a: {} for a in self.agents}
        return obs_dict, infos

    def _mpc_feature(self, sat_id):
        if self.mpc_engine is None:
            return None
        return self.mpc_engine.feature(sat_id, self.core, self.gs_ids)

    def _build_obs_dict(self, obs):
        out = {}
        for a in self.agents:
            mpc = self._mpc_feature(a)
            vec = self.ob.local_obs(a, obs, mpc_preview=mpc)
            mask = feasible(a, obs, self.gs_ids, self.core.cfg.soft_capacity).astype(np.int8)
            out[a] = {"observation": vec, "action_mask": mask}
        return out

    def state(self):
        return self.ob.global_state(self._obs)

    def step(self, actions):
        self._prev_latency = dict(self._obs.latency)
        decoded = {}
        for a, act in actions.items():
            mask = feasible(a, self._obs, self.gs_ids, self.core.cfg.soft_capacity)
            act = int(act)
            if act != STAY and not mask[act]:
                act = STAY
            gs = decode(act, self.gs_ids)
            if gs is not None:
                decoded[a] = gs

        applied = self.control.apply_actions(decoded)
        for a, gs in decoded.items():
            if self.control.edge_dts[a].host_gs == gs:
                self.ob.mark_migration(a, self.t)

        self.t += self.dt_s
        next_obs, _, info = self.core.step({}, t=self.t)
        info["n_migrations"] = applied
        self._obs = next_obs
        self.control.store.ingest(self.control.edge_dts, self.core.net, self.t)

        rewards, breakdown = self.reward_fn.team_plus_shaping(
            next_obs, info, self._prev_latency)

        done = self.t >= self.horizon_s
        obs_dict = self._build_obs_dict(next_obs)
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {"reward_breakdown": breakdown, "n_migrations": applied}
                 for a in self.agents}
        rewards = {a: float(rewards[a]) for a in self.agents}
        if done:
            self.agents = []
        return obs_dict, rewards, terminations, truncations, infos

    def greedy_action(self, agent):
        dt = self.control.edge_dts[agent]
        target = GreedyNearestPolicy(threshold_ms=0.0).decide(dt, self._obs, self.t)
        return encode(target, self.gs_ids)


def collect_greedy_traces(scenario_path, stations_path, horizon_s=6000.0,
                          dt_s=None, ping_csv=None, reward_cfg=None):
    env = HDTNParallelEnv(scenario_path, stations_path, ping_csv=ping_csv,
                          reward_cfg=reward_cfg, horizon_s=horizon_s, dt_s=dt_s)
    logger = TraceLogger()
    obs, _ = env.reset()
    while env.agents:
        actions = {a: env.greedy_action(a) for a in env.agents}
        gstate = env.state()
        next_obs, rewards, term, trunc, infos = env.step(actions)
        dones = {a: (term.get(a, False) or trunc.get(a, False)) for a in actions}
        logger.log_step(obs, actions, rewards, next_obs, gstate, dones, env.t)
        obs = next_obs
    return logger
