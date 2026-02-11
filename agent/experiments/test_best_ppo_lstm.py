import os
import csv
import numpy as np
import tensorflow as tf

from data.splits import make_time_split
from env.multi_day_env import MultiDayExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy


def run_test_episodes(env, policy, n_episodes=1000, deterministic=True, seed=123):
    """
    Runs episodes on env and returns per-episode metrics.

    We assume your env.info at episode end includes:
      - cost_cash_vs_mid
      - filled_total
      - arrival_mid
      - remaining_qty
      - exec_vwap
    And your reward is already normalized per-step.
    """
    rng = np.random.default_rng(seed)

    rows = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        ep_return = 0.0

        # LSTM state (h, c)
        h, c = policy.initial_state(batch_size=1)

        last_info = None

        while not done:
            # policy expects shape (B, obs_dim)
            obs_b = np.expand_dims(obs, axis=0).astype(np.float32)

            action, logp, value, (h, c) = policy.act(
                obs_b, h, c, deterministic=deterministic
            )
            # action comes out shape (B, act_dim)
            a = action[0]

            obs, r, done, _, info = env.step(a)
            ep_return += float(r)
            last_info = info

        # Pull metrics from final info
        filled = float(last_info.get("filled_total", np.nan))
        arrival_mid = float(last_info.get("arrival_mid", np.nan))
        cost_cash = float(last_info.get("cost_cash_vs_mid", np.nan))
        remaining = float(last_info.get("remaining_qty", np.nan))
        exec_vwap = float(last_info.get("exec_vwap", np.nan))

        # Convert cash cost to IS bps (same normalization style you used)
        denom = filled * arrival_mid
        is_bps = (1e4 * cost_cash / denom) if denom > 0 else np.nan

        # Completion vs target (env should know target; MultiDayExecEnv likely proxies it)
        # If env exposes env.target_qty, use it. Otherwise infer using filled+remaining.
        try:
            target_qty = float(env.target_qty)
        except Exception:
            target_qty = filled + max(0.0, remaining)

        completion = (filled / target_qty) if target_qty > 0 else np.nan

        rows.append({
            "episode": ep,
            "episode_return": ep_return,
            "is_bps": is_bps,
            "cost_cash_vs_mid": cost_cash,
            "filled_total": filled,
            "arrival_mid": arrival_mid,
            "exec_vwap": exec_vwap,
            "remaining_qty": remaining,
            "completion": completion
        })

    return rows


def summarize(rows):
    is_bps = np.array([r["is_bps"] for r in rows if np.isfinite(r["is_bps"])], dtype=float)
    comp = np.array([r["completion"] for r in rows if np.isfinite(r["completion"])], dtype=float)
    rets = np.array([r["episode_return"] for r in rows if np.isfinite(r["episode_return"])], dtype=float)

    def pct(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")

    out = {
        "n_episodes": len(rows),
        "mean_is_bps": float(np.mean(is_bps)) if len(is_bps) else float("nan"),
        "median_is_bps": float(np.median(is_bps)) if len(is_bps) else float("nan"),
        "p95_is_bps": pct(is_bps, 95),
        "mean_completion": float(np.mean(comp)) if len(comp) else float("nan"),
        "min_completion": float(np.min(comp)) if len(comp) else float("nan"),
        "mean_return": float(np.mean(rets)) if len(rets) else float("nan"),
        "std_return": float(np.std(rets)) if len(rets) else float("nan"),
    }
    return out


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    # ---- config ----
    data_root = "/Users/jackfletcher/Desktop/FYP_Data/Replay_5s"  # change to your actual root if different
    ckpt_path = "checkpoints/ppo_lstm/best.weights.h5"

    n_test_episodes = 1000
    deterministic = True

    out_csv = "logs/test_best_ppo_lstm.csv"

    # ---- build split ----
    nov_dir = "/Users/jackfletcher/Desktop/FYP_Data/Replay_5s/November"
    dec_dir = "/Users/jackfletcher/Desktop/FYP_Data/Replay_5s/December"
    jan_dir = "/Users/jackfletcher/Desktop/FYP_Data/Replay_5s/January"

    train_files, val_files, test_files = make_time_split(
        nov_dir=nov_dir,
        dec_dir=dec_dir,
        jan_dir=jan_dir,
    )


    print(f"Train days: {len(train_files)} | Val days: {len(val_files)} | Test days: {len(test_files)}")

    if not test_files:
        raise RuntimeError("No test files found. Check your data_root and make_time_split().")

    # ---- env on TEST days only ----
    env_test = MultiDayExecEnv(test_files, seed=999)

    obs_dim = env_test.observation_space.shape[0]

    # ---- model/policy ----
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    # Build weights by calling once (Keras lazy build)
    dummy_obs, _ = env_test.reset()
    dummy_obs = np.expand_dims(dummy_obs, axis=0).astype(np.float32)
    h, c = policy.initial_state(batch_size=1)
    _ = policy.act(dummy_obs, h, c, deterministic=True)

    # Load best checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_weights(ckpt_path)
    print(f"Loaded: {ckpt_path}")

    # ---- run test ----
    rows = run_test_episodes(env_test, policy, n_episodes=n_test_episodes, deterministic=deterministic)
    summ = summarize(rows)

    print("\n=== TEST RESULTS (PPO-LSTM) ===")
    print(f"Episodes: {summ['n_episodes']}")
    print(f"Mean IS (bps):   {summ['mean_is_bps']:.6f}")
    print(f"Median IS (bps): {summ['median_is_bps']:.6f}")
    print(f"95% IS (bps):    {summ['p95_is_bps']:.6f}")
    print(f"Mean completion: {summ['mean_completion']:.6f}")
    print(f"Min completion:  {summ['min_completion']:.6f}")
    print(f"Mean return:     {summ['mean_return']:.6f} (std {summ['std_return']:.6f})")

    save_csv(rows, out_csv)
    print(f"\nSaved per-episode CSV: {out_csv}")


if __name__ == "__main__":
    # M1/M2 note: legacy optimizer warning is irrelevant here (no training), but keep TF clean:
    tf.get_logger().setLevel("ERROR")
    main()
