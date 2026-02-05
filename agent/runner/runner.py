import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class EpisodeLog:
    episode_return: float
    filled_total: float
    exec_vwap: float
    arrival_mid: float
    cost_cash_vs_mid: float
    remaining_qty: float


def run_episode(env, policy_fn, deterministic: bool = False, max_steps: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], EpisodeLog]:
    """
    Runs one episode in a Gymnasium-style env.

    policy_fn(obs, deterministic) -> action
      - obs: np.ndarray shape (obs_dim,)
      - action: np.ndarray shape (1,) in [0,1]

    Returns:
      traj dict of arrays + EpisodeLog from final info.
    """
    obs, _ = env.reset()
    done = False
    t = 0
    ep_ret = 0.0

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []
    done_list: List[bool] = []
    info_last: Dict[str, Any] = {}

    while not done:
        obs_list.append(obs.copy())

        action = policy_fn(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        act_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        rew_list.append(float(reward))
        done_list.append(done)
        ep_ret += float(reward)

        info_last = info
        t += 1
        if max_steps is not None and t >= max_steps:
            break

    traj = {
        "obs": np.asarray(obs_list, dtype=np.float32),        # [T, obs_dim]
        "actions": np.asarray(act_list, dtype=np.float32),    # [T, 1]
        "rewards": np.asarray(rew_list, dtype=np.float32),    # [T]
        "dones": np.asarray(done_list, dtype=np.bool_),       # [T]
    }

    log = EpisodeLog(
        episode_return=ep_ret,
        filled_total=float(info_last.get("filled_total", np.nan)),
        exec_vwap=float(info_last.get("exec_vwap", np.nan)),
        arrival_mid=float(info_last.get("arrival_mid", np.nan)),
        cost_cash_vs_mid=float(info_last.get("cost_cash_vs_mid", np.nan)),
        remaining_qty=float(info_last.get("remaining_qty", np.nan)),
    )
    return traj, log


def random_policy(action_space):
    def _pi(obs, deterministic=False):
        return action_space.sample()
    return _pi
