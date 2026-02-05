import numpy as np

from env.exec_env import ExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import collect_episodes
from agent.rl.gae import compute_gae


def main():
    env = ExecEnv("/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet")

    obs_dim = env.observation_space.shape[0]
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    rollout, stats = collect_episodes(env, policy, n_episodes=5, deterministic=False)
    print("Collected steps:", len(rollout["rewards"]))
    print("Obs shape:", rollout["obs"].shape)
    print("H shape:", rollout["h"].shape)

    adv, ret = compute_gae(rollout["rewards"], rollout["values"], rollout["dones"])
    print("Adv mean/std:", float(np.mean(adv)), float(np.std(adv)))
    print("Return mean/std:", float(np.mean(ret)), float(np.std(ret)))

    print("Episode stats sample:", stats[0])


if __name__ == "__main__":
    main()
