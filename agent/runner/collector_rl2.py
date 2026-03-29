import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from agent.runner.lstm_policy import LSTMPolicy


@dataclass
class EpisodeStatsRL2:
    trial_id: int
    episode_in_trial: int
    episode_return: float
    steps: int
    filled_total: float
    exec_vwap: float
    arrival_mid: float
    cost_cash_vs_mid: float
    remaining_qty: float


def collect_trials(
    env,
    policy: LSTMPolicy,
    n_trials: int,
    episodes_per_trial: int,
    deterministic: bool = False,
) -> Tuple[Dict[str, np.ndarray], List[EpisodeStatsRL2]]:
    """
    RL²-style collection.

    Difference vs standard PPO collector:
      - policy.reset() happens ONCE per trial
      - hidden state persists across multiple episodes inside the same trial
      - env.reset() still happens at the start of each episode

    Output format intentionally matches the existing rollout dict interface so the
    current GAE / rollout buffer / PPO update code can be reused unchanged.
    """

    obs_all: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    raw_u_all: List[float] = []
    logp_all: List[float] = []
    values_all: List[float] = []
    rewards_all: List[float] = []
    dones_all: List[bool] = []
    h_all: List[np.ndarray] = []
    c_all: List[np.ndarray] = []

    ep_stats: List[EpisodeStatsRL2] = []

    for trial_id in range(n_trials):
        policy.reset()

        for episode_in_trial in range(episodes_per_trial):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            info_last: Dict[str, Any] = {}

            while not done:
                ps = policy.step(obs, deterministic=deterministic)

                obs_all.append(obs.copy())
                actions_all.append(ps.action.copy())
                raw_u_all.append(ps.raw_u)
                logp_all.append(ps.logp)
                values_all.append(ps.value)
                h_all.append(ps.state_h.copy())
                c_all.append(ps.state_c.copy())

                obs, reward, terminated, truncated, info = env.step(ps.action)
                done = bool(terminated or truncated)

                rewards_all.append(float(reward))
                dones_all.append(done)
                ep_ret += float(reward)
                steps += 1
                info_last = info_last if not info else info

            ep_stats.append(
                EpisodeStatsRL2(
                    trial_id=trial_id,
                    episode_in_trial=episode_in_trial,
                    episode_return=ep_ret,
                    steps=steps,
                    filled_total=float(info_last.get("filled_total", np.nan)),
                    exec_vwap=float(info_last.get("exec_vwap", np.nan)),
                    arrival_mid=float(info_last.get("arrival_mid", np.nan)),
                    cost_cash_vs_mid=float(info_last.get("cost_cash_vs_mid", np.nan)),
                    remaining_qty=float(info_last.get("remaining_qty", np.nan)),
                )
            )

    rollout = {
        "obs": np.asarray(obs_all, dtype=np.float32),
        "actions": np.asarray(actions_all, dtype=np.float32),
        "raw_u": np.asarray(raw_u_all, dtype=np.float32),
        "logp": np.asarray(logp_all, dtype=np.float32),
        "values": np.asarray(values_all, dtype=np.float32),
        "rewards": np.asarray(rewards_all, dtype=np.float32),
        "dones": np.asarray(dones_all, dtype=np.bool_),
        "h": np.asarray(h_all, dtype=np.float32),
        "c": np.asarray(c_all, dtype=np.float32),
    }
    return rollout, ep_stats
