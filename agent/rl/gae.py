import numpy as np
from typing import Tuple


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.999,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    rewards: [N]
    values:  [N]
    dones:   [N] boolean, True when episode ends at that step
    Returns:
      advantages: [N]
      returns:    [N]  where returns = advantages + values
    """
    N = len(rewards)
    adv = np.zeros(N, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(N)):
        if dones[t]:
            next_nonterminal = 0.0
            next_value = 0.0
        else:
            next_nonterminal = 1.0
            next_value = values[t + 1] if (t + 1) < N else 0.0

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)
