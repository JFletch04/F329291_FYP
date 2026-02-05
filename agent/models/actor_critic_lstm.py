import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


LOG_2PI = np.log(2.0 * np.pi)


class RecurrentActorCritic(tf.keras.Model):
    """
    MLP -> LSTM -> policy/value heads.
    Policy is a squashed Gaussian: raw u ~ N(mu, std), action a = sigmoid(u) in [0,1].
    """

    def __init__(self, obs_dim: int, hidden_units: int = 128, lstm_units: int = 128):
        super().__init__()

        # Feature encoder
        self.fc1 = layers.Dense(hidden_units, activation="tanh")
        self.fc2 = layers.Dense(hidden_units, activation="tanh")

        # Memory
        self.lstm = layers.LSTM(
            lstm_units, return_sequences=True, return_state=True
        )

        # Policy head
        self.mu_head = layers.Dense(1, activation=None)
        # log_std as a trainable scalar (simple + stable)
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones((1,)), trainable=True)

        # Value head
        self.v_head = layers.Dense(1, activation=None)

        self.obs_dim = obs_dim
        self.lstm_units = lstm_units

    def initial_state(self, batch_size: int):
        # LSTM state: (h, c), each [B, lstm_units]
        h = tf.zeros((batch_size, self.lstm_units), dtype=tf.float32)
        c = tf.zeros((batch_size, self.lstm_units), dtype=tf.float32)
        return (h, c)

    def call(self, obs_seq, initial_state=None, training=False):
        """
        obs_seq: [B, T, obs_dim]
        returns: mu [B,T,1], log_std [1], v [B,T,1], next_state (h,c)
        """
        x = self.fc1(obs_seq)
        x = self.fc2(x)

        if initial_state is None:
            lstm_out, h, c = self.lstm(x, training=training)
        else:
            lstm_out, h, c = self.lstm(x, initial_state=initial_state, training=training)

        mu = self.mu_head(lstm_out)
        v = self.v_head(lstm_out)
        return mu, self.log_std, v, (h, c)


def normal_log_prob(x, mu, log_std):
    # x, mu: [..., 1], log_std: [1]
    var = tf.exp(2.0 * log_std)
    return -0.5 * (((x - mu) ** 2) / var + 2.0 * log_std + LOG_2PI)


def log_prob_squashed_gaussian(raw_u, mu, log_std):
    """
    raw_u: pre-sigmoid action, shape [...,1]
    a = sigmoid(raw_u)
    logp(a) = logp(raw_u) - log|d sigmoid(raw_u)/d raw_u|
    """
    logp_u = normal_log_prob(raw_u, mu, log_std)  # [...,1]
    a = tf.sigmoid(raw_u)
    log_det = tf.math.log(a * (1.0 - a) + 1e-8)   # [...,1]
    return tf.squeeze(logp_u - log_det, axis=-1)  # [...]


def sample_action(mu, log_std):
    std = tf.exp(log_std)
    eps = tf.random.normal(shape=tf.shape(mu))
    raw_u = mu + std * eps
    a = tf.sigmoid(raw_u)
    return a, raw_u


@tf.function
def act_step(model: RecurrentActorCritic, obs_t, state, deterministic: bool):
    """
    Single-step action for one environment.
    obs_t: [obs_dim] float32
    state: (h,c) each [1,lstm_units]
    Returns: action [1], raw_u [1], logp scalar, value scalar, next_state
    """
    obs_seq = tf.reshape(obs_t, (1, 1, model.obs_dim))  # [B=1, T=1, obs_dim]
    mu, log_std, v, next_state = model(obs_seq, initial_state=state, training=False)
    mu = mu[:, :, :]   # [1,1,1]
    v = tf.squeeze(v, axis=[0, 1, 2])  # scalar

    if deterministic:
        raw_u = mu
        a = tf.sigmoid(raw_u)
    else:
        a, raw_u = sample_action(mu, log_std)

    logp = log_prob_squashed_gaussian(raw_u, mu, log_std)  # [1,1]
    logp = tf.squeeze(logp, axis=[0, 1])  # scalar

    a = tf.squeeze(a, axis=[0, 1, 2])     # scalar
    raw_u = tf.squeeze(raw_u, axis=[0, 1, 2])  # scalar
    return a, raw_u, logp, v, next_state