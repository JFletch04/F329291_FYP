import os
import csv
import json
import time
import numpy as np
import tensorflow as tf

from data.splits import make_time_split
from env.multi_day_env import MultiDayExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy


OBS_COLS = [
    "spread",
    "trade_vol",
    "signed_vol",
    "imbalance_top5",
    "return_1",
    "remaining_frac",
    "time_remaining_frac",
]

EXECENV_FIELDS = {"horizon_steps", "side", "target_qty", "max_child_qty", "pov_cap", "taker_fee_rate"}


# -----------------------------
# File utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(rows, path):
    ensure_dir(os.path.dirname(path))
    if not rows:
        raise ValueError(f"No rows to save for: {path}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def percentile(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def summarize_episode_rows(rows):
    is_bps = np.array([r["is_bps"] for r in rows if np.isfinite(r["is_bps"])], dtype=float)
    comp = np.array([r["completion"] for r in rows if np.isfinite(r["completion"])], dtype=float)
    rets = np.array([r["episode_return"] for r in rows if np.isfinite(r["episode_return"])], dtype=float)

    return {
        "n_episodes": int(len(rows)),
        "mean_is_bps": float(np.mean(is_bps)) if is_bps.size else float("nan"),
        "std_is_bps": float(np.std(is_bps)) if is_bps.size else float("nan"),
        "median_is_bps": float(np.median(is_bps)) if is_bps.size else float("nan"),
        "p05_is_bps": percentile(is_bps, 5),
        "p25_is_bps": percentile(is_bps, 25),
        "p75_is_bps": percentile(is_bps, 75),
        "p95_is_bps": percentile(is_bps, 95),
        "p99_is_bps": percentile(is_bps, 99),
        "mean_completion": float(np.mean(comp)) if comp.size else float("nan"),
        "min_completion": float(np.min(comp)) if comp.size else float("nan"),
        "completion_rate_99p9": float(np.mean(comp >= 0.999)) if comp.size else float("nan"),
        "mean_return": float(np.mean(rets)) if rets.size else float("nan"),
        "std_return": float(np.std(rets)) if rets.size else float("nan"),
    }


# -----------------------------
# Deterministic episode control
# -----------------------------
def reseed_multiday_for_episode(env: MultiDayExecEnv, episode_seed: int):
    # reproduce MultiDayExecEnv.reset() deterministically so day choice,
    # env_seed and start_idx are all controlled by episode_seed
    env.rng = np.random.default_rng(int(episode_seed))

    path_idx = int(env.rng.integers(0, len(env.parquet_paths)))
    path = env.parquet_paths[path_idx]
    env_seed = int(env.rng.integers(0, 1_000_000))

    from env.exec_env import ExecEnv
    env._env = ExecEnv(path, seed=env_seed, **env.execenv_kwargs)

    # reseed ExecEnv.rng so start_idx is deterministic per episode_seed
    env._env.rng = np.random.default_rng(int(episode_seed))

    return env._env.reset(seed=int(episode_seed))


# -----------------------------
# Scenario handling
# -----------------------------
def make_env_for_scenario(test_files, scen: dict, seed_init: int):
    """
    Best practice with your MultiDayExecEnv:
    pass ExecEnv params via execenv_kwargs at construction time.
    That guarantees every newly created ExecEnv inside reset uses the scenario params.
    """
    execenv_kwargs = {k: v for k, v in scen.items() if k in EXECENV_FIELDS and v is not None}
    env = MultiDayExecEnv(test_files, seed=int(seed_init), **execenv_kwargs)
    return env


# -----------------------------
# Model/policy build/load
# -----------------------------
def build_policy(obs_dim: int, hidden_units=128, lstm_units=128):
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=hidden_units, lstm_units=lstm_units)
    policy = LSTMPolicy(model)
    return model, policy


def warmup_build(policy, obs_dim: int):
    # dummy forward pass to build the model before loading weights
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    h, c = policy.initial_state(batch_size=1)
    _ = policy.act(dummy_obs, h, c, deterministic=True)


# -----------------------------
# Rollout
# -----------------------------
def run_episode(env: MultiDayExecEnv, policy, episode_seed: int, deterministic: bool, ep_index: int, model_tag: str, scenario_tag: str):
    obs, _ = reseed_multiday_for_episode(env, episode_seed)

    h, c = policy.initial_state(batch_size=1)
    ep_return = 0.0
    step_rows = []
    last_info = {}
    prev_filled_total = 0.0
    prev_cost_cash = 0.0
    t = 0
    done = False

    while not done:
        obs_b = np.expand_dims(obs, axis=0).astype(np.float32)
        action, logp, value, (h, c) = policy.act(obs_b, h, c, deterministic=deterministic)
        a = float(np.asarray(action).reshape(-1)[0])

        next_obs, r, done, _, info = env.step(np.array([a], dtype=np.float32))
        ep_return += float(r)
        last_info = info if isinstance(info, dict) else {}

        filled_total = float(last_info.get("filled_total", prev_filled_total))
        cost_cash = float(last_info.get("cost_cash_vs_mid", prev_cost_cash))
        step_filled = filled_total - prev_filled_total
        step_cost_cash = cost_cash - prev_cost_cash
        prev_filled_total = filled_total
        prev_cost_cash = cost_cash

        step_row = {
            "model": model_tag,
            "scenario": scenario_tag,
            "episode": int(ep_index),
            "episode_seed": int(episode_seed),
            "t": int(t),
            "reward": float(r),
            "action": float(a),
            "logp": float(np.asarray(logp).reshape(-1)[0]) if np.asarray(logp).size else float("nan"),
            "value": float(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else float("nan"),
            "step_filled": float(step_filled),
            "step_cost_cash_vs_mid": float(step_cost_cash),
            "filled_total": float(filled_total),
            "cost_cash_vs_mid": float(cost_cash),
        }

        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        for i, col in enumerate(OBS_COLS):
            step_row[col] = float(obs_arr[i]) if i < obs_arr.size else float("nan")

        for k in ["arrival_mid", "exec_vwap", "remaining_qty"]:
            if k in last_info:
                step_row[k] = float(last_info[k])

        # useful for later auditing: which parquet/day was used
        try:
            step_row["parquet_path"] = getattr(env, "_env").df  # not serializable; ignore
        except Exception:
            pass

        step_rows.append(step_row)
        obs = next_obs
        t += 1

    filled = float(last_info.get("filled_total", np.nan))
    arrival_mid = float(last_info.get("arrival_mid", np.nan))
    cost_cash = float(last_info.get("cost_cash_vs_mid", np.nan))
    remaining = float(last_info.get("remaining_qty", np.nan))
    exec_vwap = float(last_info.get("exec_vwap", np.nan))

    denom = filled * arrival_mid
    is_bps = (1e4 * cost_cash / denom) if denom > 0 else np.nan

    # target_qty should come from current ExecEnv instance
    try:
        target_qty = float(env.target_qty)
    except Exception:
        target_qty = filled + max(0.0, remaining)

    completion = (filled / target_qty) if target_qty > 0 else np.nan

    episode_row = {
        "model": model_tag,
        "scenario": scenario_tag,
        "episode": int(ep_index),
        "episode_seed": int(episode_seed),
        "episode_return": float(ep_return),
        "is_bps": float(is_bps),
        "cost_cash_vs_mid": float(cost_cash),
        "filled_total": float(filled),
        "target_qty": float(target_qty),
        "remaining_qty": float(remaining),
        "arrival_mid": float(arrival_mid),
        "exec_vwap": float(exec_vwap),
        "completion": float(completion),
        "n_steps": int(t),
    }
    return episode_row, step_rows


def eval_model(env, policy, episode_seeds, deterministic, model_tag, scenario_tag, log_steps: bool):
    ep_rows, step_rows = [], []
    for ep_i, s in enumerate(episode_seeds):
        ep_row, st = run_episode(
            env, policy,
            episode_seed=int(s),
            deterministic=deterministic,
            ep_index=ep_i,
            model_tag=model_tag,
            scenario_tag=scenario_tag,
        )
        ep_rows.append(ep_row)
        if log_steps:
            step_rows.extend(st)
    return ep_rows, step_rows


# -----------------------------
# Main
# -----------------------------
def main():
    nov_dir = "./data/Replay_5s/November"
    dec_dir = "./data/Replay_5s/December"
    jan_dir = "./data/Replay_5s/January"

    train_files, val_files, test_files = make_time_split(nov_dir=nov_dir, dec_dir=dec_dir, jan_dir=jan_dir)
    print(f"Train days: {len(train_files)} | Val days: {len(val_files)} | Test days: {len(test_files)}")
    if not test_files:
        raise RuntimeError("No test files found. Check your split paths.")

    out_root = "logs/test_runs"
    n_test_episodes = 500
    deterministic = True
    seed_master = 12345
    log_step_traces = True

    # models: add multiple checkpoints here as needed
    models = [
        {"tag": "best", "ckpt_path": "checkpoints/ppo_lstm/best.weights.h5"},
        # {"tag": "model2", "ckpt_path": "..."},
        # {"tag": "model3", "ckpt_path": "..."},
    ]

    # scenarios: absolute values matching your dataset scale
    scenarios = [
        {"name": "default_buy",   "side": "buy",  "horizon_steps": 180, "target_qty": 0.5,  "max_child_qty": 0.05, "pov_cap": 0.10, "taker_fee_rate": 0.0},
        {"name": "small_buy",     "side": "buy",  "horizon_steps": 180, "target_qty": 0.25, "max_child_qty": 0.05, "pov_cap": 0.10, "taker_fee_rate": 0.0},
        {"name": "large_buy",     "side": "buy",  "horizon_steps": 180, "target_qty": 1.0,  "max_child_qty": 0.05, "pov_cap": 0.10, "taker_fee_rate": 0.0},
        {"name": "tight_pov",     "side": "buy",  "horizon_steps": 180, "target_qty": 0.5,  "max_child_qty": 0.05, "pov_cap": 0.05, "taker_fee_rate": 0.0},
        {"name": "short_horizon", "side": "buy",  "horizon_steps": 120, "target_qty": 0.5,  "max_child_qty": 0.05, "pov_cap": 0.10, "taker_fee_rate": 0.0},
        {"name": "default_sell",  "side": "sell", "horizon_steps": 180, "target_qty": 0.5,  "max_child_qty": 0.05, "pov_cap": 0.10, "taker_fee_rate": 0.0},
    ]

    # fixed episode seeds shared across all models and scenarios
    rng = np.random.default_rng(seed_master)
    episode_seeds = rng.integers(0, 1_000_000, size=n_test_episodes, dtype=np.int64).tolist()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"run_{run_ts}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({
            "n_test_episodes": n_test_episodes,
            "deterministic": deterministic,
            "seed_master": seed_master,
            "models": models,
            "scenarios": scenarios,
            "test_files_n": len(test_files),
            "log_step_traces": log_step_traces,
        }, f, indent=2)

    summary_rows = []

    for scen in scenarios:
        scen_tag = scen["name"]
        print(f"\n--- Scenario: {scen_tag} ---")

        # construct env with scenario params passed into execenv_kwargs
        env_test = make_env_for_scenario(test_files, scen, seed_init=999)
        obs_dim = env_test.observation_space.shape[0]

        for m in models:
            tag = m["tag"]
            ckpt_path = m["ckpt_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            model, policy = build_policy(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
            warmup_build(policy, obs_dim=obs_dim)
            model.load_weights(ckpt_path)
            print(f"Loaded model={tag} ckpt={ckpt_path}")

            ep_rows, st_rows = eval_model(
                env_test, policy,
                episode_seeds=episode_seeds,
                deterministic=deterministic,
                model_tag=tag,
                scenario_tag=scen_tag,
                log_steps=log_step_traces,
            )

            out_subdir = os.path.join(run_dir, f"scenario_{scen_tag}", f"model_{tag}")
            ensure_dir(out_subdir)
            save_csv(ep_rows, os.path.join(out_subdir, "episode_metrics.csv"))
            if log_step_traces:
                save_csv(st_rows, os.path.join(out_subdir, "step_traces.csv"))

            summ = summarize_episode_rows(ep_rows)
            summary_rows.append({
                "run_id": run_ts,
                "scenario": scen_tag,
                "model": tag,
                "side": scen.get("side"),
                "horizon_steps": scen.get("horizon_steps"),
                "target_qty": scen.get("target_qty"),
                "max_child_qty": scen.get("max_child_qty"),
                "pov_cap": scen.get("pov_cap"),
                "taker_fee_rate": scen.get("taker_fee_rate"),
                **summ,
            })

            print(
                f"  mean IS bps: {summ['mean_is_bps']:.6f} | median: {summ['median_is_bps']:.6f} "
                f"| p95: {summ['p95_is_bps']:.6f} | compl>=0.999: {summ['completion_rate_99p9']:.3f}"
            )

    save_csv(summary_rows, os.path.join(run_dir, "run_summary.csv"))
    print(f"\nSaved run summary: {os.path.join(run_dir, 'run_summary.csv')}")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()