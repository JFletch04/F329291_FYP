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
    Holds the TF model + the current LSTM state for ONE environment.

    You reset() it at episode start.
    You step(obs) to get action + diagnostics and update hidden state.
    """

    def __init__(self, model: RecurrentActorCritic):
        self.model = model
        self.state: Optional[Tuple[tf.Tensor, tf.Tensor]] = None

    def reset(self):
        self.state = self.model.initial_state(batch_size=1)

    def step(self, obs: np.ndarray, deterministic: bool = False) -> PolicyStep:
        """
        obs: np.ndarray shape (obs_dim,)
        returns: PolicyStep (includes state BEFORE acting, useful for training)
        """
        if self.state is None:
            self.reset()

        # Save state BEFORE acting (needed for training chunks)
        h, c = self.state
        h_np = h.numpy().reshape(-1)  # (lstm_units,)
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
