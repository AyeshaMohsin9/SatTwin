# Module 4 — logs (obs, mask, action, reward, next_obs, global_state) for offline warm-start.
import pickle
from dataclasses import dataclass, field


@dataclass
class Transition:
    agent: str
    obs: object
    mask: object
    action: int
    reward: float
    next_obs: object
    global_state: object
    done: bool
    t: float


@dataclass
class TraceLogger:
    transitions: list = field(default_factory=list)

    def log(self, agent, obs, mask, action, reward, next_obs, global_state, done, t):
        self.transitions.append(Transition(
            agent=agent, obs=obs, mask=mask, action=action, reward=reward,
            next_obs=next_obs, global_state=global_state, done=done, t=t))

    def log_step(self, obs_dict, actions, rewards, next_obs_dict, global_state,
                 dones, t):
        for a in actions:
            self.log(a, obs_dict[a]["observation"], obs_dict[a]["action_mask"],
                     actions[a], rewards.get(a, 0.0),
                     next_obs_dict.get(a, {}).get("observation"),
                     global_state, dones.get(a, False), t)

    def __len__(self):
        return len(self.transitions)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.transitions, f)

    @staticmethod
    def load(path):
        tl = TraceLogger()
        with open(path, "rb") as f:
            tl.transitions = pickle.load(f)
        return tl

    def clear(self):
        self.transitions = []
