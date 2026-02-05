import tensorflow as tf

from env.exec_env import ExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import collect_episodes
from agent.rl.gae import compute_gae
from agent.rl.rollout_buffer import RolloutBuffer
from agent.rl.ppo_update import ppo_update_step


def main():
    env = ExecEnv("/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet")

    obs_dim = env.observation_space.shape[0]
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    rollout, _ = collect_episodes(env, policy, n_episodes=8, deterministic=False)
    adv, ret = compute_gae(rollout["rewards"], rollout["values"], rollout["dones"])

    buf = RolloutBuffer(chunk_len=32)
    arrays = buf.make_training_arrays(rollout, adv, ret)

    # M1/M2: use legacy Adam
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)

    mb = next(buf.iter_minibatches(arrays, batch_size_chunks=4, shuffle=True))

    total_loss, pi_loss, v_loss, entropy, approx_kl, clip_frac, grad_norm = ppo_update_step(
        model, opt,
        mb.obs, mb.raw_u, mb.logp_old, mb.adv, mb.returns, mb.mask, mb.h0, mb.c0,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5
    )

    print("total_loss:", float(total_loss))
    print("pi_loss:", float(pi_loss))
    print("v_loss:", float(v_loss))
    print("entropy:", float(entropy))
    print("approx_kl:", float(approx_kl))
    print("clip_frac:", float(clip_frac))
    print("grad_norm:", float(grad_norm))


if __name__ == "__main__":
    main()

