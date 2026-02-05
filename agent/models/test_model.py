import numpy as np
import tensorflow as tf
from agent.models.actor_critic_lstm import RecurrentActorCritic, act_step

def main():
    obs_dim = 7
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    state = model.initial_state(batch_size=1)

    obs = tf.constant(np.zeros((obs_dim,), dtype=np.float32))
    a, raw_u, logp, v, next_state = act_step(model, obs, state, deterministic=False)
    print("action:", float(a), "logp:", float(logp), "value:", float(v))

if __name__ == "__main__":
    main()
