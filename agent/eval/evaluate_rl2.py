import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector_rl2 import EpisodeStatsRL2


# Aggregating evaluation metrics across all the RL² trials
@dataclass
class EvalMetricsRL2:
    n_trials: int
    episodes_per_trial: int
    n_episodes: int
    mean_return: float
    mean_steps: float
    mean_is_bps: float
    mean_completion: float
    mean_cost_cash_vs_mid: float


# Converting episode stats into IS (implementation shortfall (bps))
def _is_bps_from_stats(s: EpisodeStatsRL2) -> float:
    denom = s.filled_total * s.arrival_mid
    if denom <= 0 or math.isnan(denom):
        return float("nan")
    return 1e4 * (s.cost_cash_vs_mid / denom)


def evaluate_policy_rl2(
    env,
    policy: LSTMPolicy,
    n_trials: int = 50,
    episodes_per_trial: int = 4,
    deterministic: bool = True,
) -> EvalMetricsRL2:
    # RL² evaluation: the hidden state persists across episodes within a trial

    returns: List[float] = []
    steps: List[int] = []
    is_bps: List[float] = []
    completion: List[float] = []
    costs: List[float] = []

    for _ in range(n_trials):
        # Reset the hidden state once per trial
        policy.reset()

        for _ep in range(episodes_per_trial):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            info_last: Dict[str, Any] = {}
            t = 0

            # Run the episode
            while not done:
                ps = policy.step(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(ps.action)
                done = bool(terminated or truncated)
                ep_ret += float(r)
                t += 1
                if info:
                    info_last = info

            # Building the episode stats
            s = EpisodeStatsRL2(
                trial_id=0,
                episode_in_trial=0,
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

            # Compute IS (bps)
            isbps = _is_bps_from_stats(s)
            if not math.isnan(isbps):
                is_bps.append(isbps)

            # The completion ratio
            if s.filled_total > 0 and not math.isnan(s.filled_total):
                completion.append(s.filled_total / env.target_qty)

    n_episodes = n_trials * episodes_per_trial

    # Aggregate the final metrics
    return EvalMetricsRL2(
        n_trials=n_trials,
        episodes_per_trial=episodes_per_trial,
        n_episodes=n_episodes,
        mean_return=float(np.mean(returns)) if returns else float("nan"),
        mean_steps=float(np.mean(steps)) if steps else float("nan"),
        mean_is_bps=float(np.mean(is_bps)) if is_bps else float("nan"),
        mean_completion=float(np.mean(completion)) if completion else float("nan"),
        mean_cost_cash_vs_mid=float(np.mean(costs)) if costs else float("nan"),
    )
