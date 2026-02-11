import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import EpisodeStats


@dataclass
class EvalMetrics:
    n_episodes: int
    mean_return: float
    mean_steps: float
    mean_is_bps: float
    mean_completion: float
    mean_cost_cash_vs_mid: float


def _is_bps_from_stats(s: EpisodeStats) -> float:
    # IS (cash vs mid) already includes spread/slip, so express in bps of arrival notional
    denom = s.filled_total * s.arrival_mid
    if denom <= 0 or math.isnan(denom):
        return float("nan")
    return 1e4 * (s.cost_cash_vs_mid / denom)


def evaluate_policy(env, policy: LSTMPolicy, n_episodes: int = 50, deterministic: bool = True) -> EvalMetrics:
    """
    Runs full episodes (no learning) and returns execution metrics.
    """
    returns: List[float] = []
    steps: List[int] = []
    is_bps: List[float] = []
    completion: List[float] = []
    costs: List[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()
        done = False
        ep_ret = 0.0
        info_last: Dict[str, Any] = {}

        t = 0
        while not done:
            ps = policy.step(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(ps.action)
            done = bool(terminated or truncated)
            ep_ret += float(r)
            t += 1
            if info:
                info_last = info

        s = EpisodeStats(
            episode_return=ep_ret,
            steps=t,
            filled_total=float(info_last.get("filled_total", np.nan)),
            exec_vwap=float(info_last.get("exec_vwap", np.nan)),
            arrival_mid=float(info_last.get("arrival_mid", np.nan)),
            cost_cash_vs_mid=float(info_last.get("cost_cash_vs_mid", np.nan)),
            remaining_qty=float(info_last.get("remaining_qty", np.nan)),
        )

        returns.append(s.episode_return)
        steps.append(s.steps)
        costs.append(s.cost_cash_vs_mid)

        isbps = _is_bps_from_stats(s)
        if not math.isnan(isbps):
            is_bps.append(isbps)

        if s.filled_total > 0 and not math.isnan(s.filled_total):
            completion.append(s.filled_total / env.target_qty)

    return EvalMetrics(
        n_episodes=n_episodes,
        mean_return=float(np.mean(returns)) if returns else float("nan"),
        mean_steps=float(np.mean(steps)) if steps else float("nan"),
        mean_is_bps=float(np.mean(is_bps)) if is_bps else float("nan"),
        mean_completion=float(np.mean(completion)) if completion else float("nan"),
        mean_cost_cash_vs_mid=float(np.mean(costs)) if costs else float("nan"),
    )
