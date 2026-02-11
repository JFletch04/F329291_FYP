import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, Optional

from agent.models.actor_critic_lstm import RecurrentActorCritic, act_step


@dataclass
class PolicyStep:
    action: np.ndarray      # shape (1,)
    raw_u: float            # scalar
    logp: float             # scalar
    value: float            # scalar
    state_h: np.ndarray     # shape (lstm_units,)  state BEFORE acting
    state_c: np.ndarray     # shape (lstm_units,)


class LSTMPolicy:
    """
    Wrapper around TF recurrent actor-critic.

    Supports:
      - reset(): reset internal state for 1 env
      - step(obs): uses internal state, updates it, returns PolicyStep (for training/collection)
      - initial_state(batch_size): passthrough to model.initial_state
      - act(obs, h, c): stateless call used by evaluation code (does NOT change internal state)
    """

    def __init__(self, model: RecurrentActorCritic):
        self.model = model
        self.state: Optional[Tuple[tf.Tensor, tf.Tensor]] = None

    def initial_state(self, batch_size: int = 1) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.model.initial_state(batch_size=batch_size)

    def reset(self):
        self.state = self.model.initial_state(batch_size=1)

    def act(
        self,
        obs: np.ndarray,
        h: tf.Tensor,
        c: tf.Tensor,
        deterministic: bool = False,
    ):
        """
        Stateless action call used by evaluation scripts.

        Returns exactly what test_best_ppo_lstm expects:
        action (np array shape (1,)),
        logp (float),
        value (float),
        (next_h, next_c)
        """
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        a, raw_u, logp, v, (h2, c2) = act_step(self.model, obs_tf, (h, c), deterministic)

        action = np.array([float(a)], dtype=np.float32)  # <-- key line (makes action[0] valid)
        return action, float(logp), float(v), (h2, c2)


    def step(self, obs: np.ndarray, deterministic: bool = False) -> PolicyStep:
        """
        Uses and updates INTERNAL state (used by collector).
        """
        if self.state is None:
            self.reset()

        h, c = self.state

        # Save state BEFORE acting (needed for training chunks)
        h_np = h.numpy().reshape(-1)
        c_np = c.numpy().reshape(-1)

        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        a, raw_u, logp, v, next_state = act_step(self.model, obs_tf, self.state, deterministic)
        self.state = next_state

        return PolicyStep(
            action=np.array([float(a)], dtype=np.float32),
            raw_u=float(raw_u),
            logp=float(logp),
            value=float(v),
            state_h=h_np.astype(np.float32),
            state_c=c_np.astype(np.float32),
        )

