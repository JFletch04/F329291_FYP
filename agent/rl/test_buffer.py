import numpy as np

from env.exec_env import ExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import collect_episodes
from agent.rl.gae import compute_gae
from agent.rl.rollout_buffer import RolloutBuffer


def main():
    env = ExecEnv("/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet")

    obs_dim = env.observation_space.shape[0]
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    rollout, stats = collect_episodes(env, policy, n_episodes=5, deterministic=False)
    adv, ret = compute_gae(rollout["rewards"], rollout["values"], rollout["dones"])

    buf = RolloutBuffer(chunk_len=32)
    arrays = buf.make_training_arrays(rollout, adv, ret)

    print("Chunks:", arrays["obs"].shape[0])
    print("obs chunk shape:", arrays["obs"].shape)       # [C, 32, 7]
    print("h0 shape:", arrays["h0"].shape)               # [C, 128]
    print("mask example sum:", arrays["mask"][0].sum())

    # Iterate minibatches
    for mb in buf.iter_minibatches(arrays, batch_size_chunks=4):
        print("Minibatch obs:", mb.obs.shape)
        print("Minibatch mask sum:", mb.mask.sum())
        break


if __name__ == "__main__":
    main()
