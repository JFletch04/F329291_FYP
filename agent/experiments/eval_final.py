import os
import math
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from data.splits import make_time_split_from_root
from env.exec_env import ExecEnv


# -----------------------------------------------------------------------------
# Top-model registry
# -----------------------------------------------------------------------------
BTC_MODELS = [
    {
        "policy_name": "BTC_PPO_1",
        "run_name": "BTC_S2B3_lr0.001_clip0.2_ent0.005_ep8_bs16_cl32_gamma0.99_lam0.9_seed2",
        "ckpt": "checkpoints/BTC/BTC_S2B3_lr0.001_clip0.2_ent0.005_ep8_bs16_cl32_gamma0.99_lam0.9_seed2/best.weights.h5",
    },
    {
        "policy_name": "BTC_PPO_2",
        "run_name": "BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.99_lam0.95_seed2",
        "ckpt": "checkpoints/BTC/BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.99_lam0.95_seed2/best.weights.h5",
    },
    {
        "policy_name": "BTC_PPO_3",
        "run_name": "BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.999_lam0.95_seed2",
        "ckpt": "checkpoints/BTC/BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.999_lam0.95_seed2/best.weights.h5",
    },
]

DOGE_MODELS = [
    {
        "policy_name": "DOGE_PPO_1",
        "run_name": "DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0",
        "ckpt": "checkpoints/DOGE/DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
    },
    {
        "policy_name": "DOGE_PPO_2",
        "run_name": "DOGE_S2B2_lr0.0003_clip0.2_ent0.005_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0",
        "ckpt": "checkpoints/DOGE/DOGE_S2B2_lr0.0003_clip0.2_ent0.005_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
    },
    {
        "policy_name": "DOGE_PPO_3",
        "run_name": "DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.99_lam0.9_seed1_H4320_Q3e+06_pov0.1_fee0",
        "ckpt": "checkpoints/DOGE/DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.99_lam0.9_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
    },
]


@dataclass(frozen=True)
class EvalScenario:
    name: str
    target_qty: float
    horizon_steps: int
    max_child_qty: float
    pov_cap: float
    side: str
    taker_fee_rate: float = 0.0


@dataclass(frozen=True)
class EvalEpisode:
    asset: str
    scenario_name: str
    day_path: str
    day_name: str
    start_idx: int


ASSET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "side": "buy",
        "horizon_steps": 4320,
        "max_child_qty": 0.25,
        "pov_cap": 0.05,
        "taker_fee_rate": 0.0,
        "size_grid": [10.0, 25.0, 50.0, 100.0, 200.0],
        "models": BTC_MODELS,
    },
    "DOGE": {
        "side": "buy",
        "horizon_steps": 4320,
        "max_child_qty": 3500.0,
        "pov_cap": 0.10,
        "taker_fee_rate": 0.0,
        "size_grid": [500_000.0, 1_000_000.0, 3_000_000.0, 6_000_000.0, 12_000_000.0],
        "models": DOGE_MODELS,
    },
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class PPOPolicyRunner:
    def __init__(self, ckpt_path: str, obs_dim: int = 7, hidden_units: int = 128, lstm_units: int = 128):
        self.ckpt_path = ckpt_path
        self.model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=hidden_units, lstm_units=lstm_units)
        # build once so load_weights has a created variable structure
        dummy_obs = tf.zeros((1, 1, obs_dim), dtype=tf.float32)
        _ = self.model(dummy_obs, initial_state=self.model.initial_state(batch_size=1), training=False)
        self.model.load_weights(ckpt_path)
        self.policy = LSTMPolicy(self.model)

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: np.ndarray, env: ExecEnv) -> float:
        ps = self.policy.step(obs, deterministic=True)
        return float(np.asarray(ps.action).reshape(-1)[0])


class TWAPPolicyRunner:
    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, env: ExecEnv) -> float:
        remaining_steps = max(int(env.horizon_steps - env.t), 1)
        return float(np.clip(1.0 / remaining_steps, 0.0, 1.0))


class EnvFactory:
    """
    Reuses ExecEnv instances per (day_path, scenario) and force-resets them to a fixed start_idx.
    This avoids modifying your core training env while making evaluation deterministic.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, str], ExecEnv] = {}

    def get_env(self, day_path: str, scenario: EvalScenario, seed: int) -> ExecEnv:
        key = (day_path, scenario.name)
        if key not in self._cache:
            self._cache[key] = ExecEnv(
                replay_parquet_path=day_path,
                horizon_steps=scenario.horizon_steps,
                side=scenario.side,
                target_qty=scenario.target_qty,
                max_child_qty=scenario.max_child_qty,
                pov_cap=scenario.pov_cap,
                taker_fee_rate=scenario.taker_fee_rate,
                seed=seed,
            )
        env = self._cache[key]
        env.horizon_steps = int(scenario.horizon_steps)
        env.side = str(scenario.side).lower()
        env.target_qty = float(scenario.target_qty)
        env.max_child_qty = float(scenario.max_child_qty)
        env.pov_cap = float(scenario.pov_cap)
        env.taker_fee_rate = float(scenario.taker_fee_rate)
        return env

    @staticmethod
    def reset_to_start(env: ExecEnv, start_idx: int, seed: Optional[int] = None):
        env.reset(seed=seed)
        max_start = len(env.df) - env.horizon_steps - 1
        if start_idx < 1 or start_idx >= max_start:
            raise ValueError(
                f"Invalid start_idx={start_idx} for rows={len(env.df)} and horizon_steps={env.horizon_steps}"
            )
        env.start_idx = int(start_idx)
        env.t = 0
        env.remaining_qty = env.target_qty
        env.filled_total = 0.0
        env.notional_total = 0.0
        env.is_cash_total = 0.0
        env.arrival_mid = float(env.df.iloc[env.start_idx]["mid"])
        return env._obs(), {}


def make_scenarios(asset: str) -> List[EvalScenario]:
    asset = asset.upper()
    cfg = ASSET_DEFAULTS[asset]
    scenarios: List[EvalScenario] = []
    for qty in cfg["size_grid"]:
        qty_tag = f"{qty:g}".replace(".", "p")
        scenarios.append(
            EvalScenario(
                name=f"{asset.lower()}_q{qty_tag}",
                target_qty=float(qty),
                horizon_steps=int(cfg["horizon_steps"]),
                max_child_qty=float(cfg["max_child_qty"]),
                pov_cap=float(cfg["pov_cap"]),
                side=str(cfg["side"]),
                taker_fee_rate=float(cfg["taker_fee_rate"]),
            )
        )
    return scenarios


def sample_fixed_episodes(
    asset: str,
    scenario: EvalScenario,
    test_files: List[str],
    n_episodes: int,
    sample_seed: int,
) -> List[EvalEpisode]:
    rng = np.random.default_rng(sample_seed)
    usable: List[Tuple[str, int]] = []
    for path in test_files:
        try:
            n_rows = len(pd.read_parquet(path, columns=["mid"]))
        except Exception:
            n_rows = len(pd.read_parquet(path))
        max_start = n_rows - scenario.horizon_steps - 1
        if max_start > 1:
            usable.append((path, max_start))

    if not usable:
        raise RuntimeError(f"No usable test files for {asset} and horizon={scenario.horizon_steps}")

    episodes: List[EvalEpisode] = []
    for _ in range(n_episodes):
        day_path, max_start = usable[int(rng.integers(0, len(usable)))]
        start_idx = int(rng.integers(1, max_start))
        episodes.append(
            EvalEpisode(
                asset=asset,
                scenario_name=scenario.name,
                day_path=day_path,
                day_name=os.path.basename(day_path),
                start_idx=start_idx,
            )
        )
    return episodes


def _safe_is_bps(cost_cash_vs_mid: float, target_qty: float, arrival_mid: float) -> float:
    denom = float(target_qty) * float(arrival_mid)
    if denom <= 0 or math.isnan(denom):
        return float("nan")
    return 1e4 * float(cost_cash_vs_mid) / denom


def run_episode(
    env_factory: EnvFactory,
    scenario: EvalScenario,
    episode: EvalEpisode,
    policy_name: str,
    policy_runner,
    env_seed: int,
    save_trajectory: bool = False,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    env = env_factory.get_env(episode.day_path, scenario, seed=env_seed)
    obs, _ = env_factory.reset_to_start(env, episode.start_idx, seed=env_seed)
    policy_runner.reset()

    done = False
    ep_ret = 0.0
    info_last: Dict[str, Any] = {}
    step_idx = 0
    traj_rows: List[Dict[str, Any]] = []

    while not done:
        action = float(policy_runner.act(obs, env))
        remaining_before = float(env.remaining_qty)
        t_before = int(env.t)
        row_before = env._get_row(env.start_idx + env.t)
        mid_before = float(row_before["mid"])
        trade_vol_before = float(row_before["trade_vol"])
        spread_before = float(row_before["spread"])

        obs_next, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        done = bool(terminated or truncated)
        info_last = info if info else info_last
        ep_ret += float(reward)

        executed_now = remaining_before - float(env.remaining_qty)
        traj_rows.append(
            {
                "asset": episode.asset,
                "scenario_name": scenario.name,
                "policy_name": policy_name,
                "day_name": episode.day_name,
                "start_idx": episode.start_idx,
                "step": step_idx,
                "env_t_before": t_before,
                "action": action,
                "reward": float(reward),
                "mid": mid_before,
                "spread": spread_before,
                "trade_vol": trade_vol_before,
                "remaining_before": remaining_before,
                "executed_now": executed_now,
                "remaining_after": float(env.remaining_qty),
            }
        )
        obs = obs_next
        step_idx += 1

    filled_total = float(info_last.get("filled_total", np.nan))
    arrival_mid = float(info_last.get("arrival_mid", np.nan))
    cost_cash_vs_mid = float(info_last.get("cost_cash_vs_mid", np.nan))
    remaining_qty = float(info_last.get("remaining_qty", np.nan))
    completion = filled_total / scenario.target_qty if scenario.target_qty > 0 else float("nan")
    is_bps = _safe_is_bps(cost_cash_vs_mid, scenario.target_qty, arrival_mid)

    row = {
        "asset": episode.asset,
        "scenario_name": scenario.name,
        "target_qty": float(scenario.target_qty),
        "horizon_steps": int(scenario.horizon_steps),
        "max_child_qty": float(scenario.max_child_qty),
        "pov_cap": float(scenario.pov_cap),
        "side": scenario.side,
        "policy_name": policy_name,
        "day_name": episode.day_name,
        "day_path": episode.day_path,
        "start_idx": int(episode.start_idx),
        "steps_used": int(step_idx),
        "episode_return": float(ep_ret),
        "filled_total": filled_total,
        "exec_vwap": float(info_last.get("exec_vwap", np.nan)),
        "arrival_mid": arrival_mid,
        "cost_cash_vs_mid": cost_cash_vs_mid,
        "remaining_qty": remaining_qty,
        "completion": completion,
        "completion_100pct": float(completion >= 0.999999),
        "terminated_at_horizon": float(step_idx >= scenario.horizon_steps),
        "is_bps": is_bps,
    }

    traj_df = pd.DataFrame(traj_rows) if save_trajectory else None
    return row, traj_df


def summarise_results(per_episode: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["asset", "scenario_name", "target_qty", "policy_name"]

    for keys, g in per_episode.groupby(group_cols, dropna=False):
        asset, scenario_name, target_qty, policy_name = keys
        is_vals = g["is_bps"].dropna().values
        completion_vals = g["completion"].dropna().values
        row = {
            "asset": asset,
            "scenario_name": scenario_name,
            "target_qty": float(target_qty),
            "policy_name": policy_name,
            "n_episodes": int(len(g)),
            "mean_episode_return": float(g["episode_return"].mean()),
            "mean_steps_used": float(g["steps_used"].mean()),
            "mean_filled_total": float(g["filled_total"].mean()),
            "mean_remaining_qty": float(g["remaining_qty"].mean()),
            "mean_completion": float(np.mean(completion_vals)) if len(completion_vals) else float("nan"),
            "completion_rate_100pct": float(np.mean(g["completion_100pct"].values)) if len(g) else float("nan"),
            "mean_is_bps": float(np.mean(is_vals)) if len(is_vals) else float("nan"),
            "std_is_bps": float(np.std(is_vals, ddof=1)) if len(is_vals) > 1 else 0.0,
            "median_is_bps": float(np.median(is_vals)) if len(is_vals) else float("nan"),
            "p90_is_bps": float(np.percentile(is_vals, 90)) if len(is_vals) else float("nan"),
            "p95_is_bps": float(np.percentile(is_vals, 95)) if len(is_vals) else float("nan"),
            "p99_is_bps": float(np.percentile(is_vals, 99)) if len(is_vals) else float("nan"),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["asset", "target_qty", "mean_is_bps", "policy_name"]).reset_index(drop=True)
    return out


def evaluate_asset(
    asset: str,
    data_root: str,
    output_root: str,
    n_episodes: int,
    eval_seed: int,
    trajectory_episodes: int,
) -> None:
    asset = asset.upper()
    if asset not in ASSET_DEFAULTS:
        raise ValueError(f"Unsupported asset: {asset}")

    set_global_seed(eval_seed)
    train_files, val_files, test_files = make_time_split_from_root(data_root, jan_val_days=7)
    if len(test_files) == 0:
        raise RuntimeError(f"No test files found under {data_root}")

    scenarios = make_scenarios(asset)
    model_specs = ASSET_DEFAULTS[asset]["models"]

    asset_out = os.path.join(output_root, asset)
    os.makedirs(asset_out, exist_ok=True)
    env_factory = EnvFactory()

    all_episode_rows: List[Dict[str, Any]] = []
    all_episode_specs: List[Dict[str, Any]] = []

    for scenario_idx, scenario in enumerate(scenarios):
        scenario_seed = eval_seed + 10_000 * (scenario_idx + 1)
        scenario_out = os.path.join(asset_out, scenario.name)
        os.makedirs(scenario_out, exist_ok=True)

        episodes = sample_fixed_episodes(
            asset=asset,
            scenario=scenario,
            test_files=test_files,
            n_episodes=n_episodes,
            sample_seed=scenario_seed,
        )
        episode_spec_df = pd.DataFrame([asdict(ep) for ep in episodes])
        episode_spec_df["target_qty"] = float(scenario.target_qty)
        episode_spec_df["horizon_steps"] = int(scenario.horizon_steps)
        episode_spec_df.to_csv(os.path.join(scenario_out, "episodes.csv"), index=False)
        all_episode_specs.extend(episode_spec_df.to_dict(orient="records"))

        policies: List[Tuple[str, Any, Optional[Dict[str, str]]]] = [("TWAP", TWAPPolicyRunner(), None)]
        for spec in model_specs:
            policies.append((spec["policy_name"], PPOPolicyRunner(spec["ckpt"]), spec))

        traj_dir = os.path.join(scenario_out, "trajectories")
        if trajectory_episodes > 0:
            os.makedirs(traj_dir, exist_ok=True)

        for policy_name, runner, spec in policies:
            print(f"[{asset}] scenario={scenario.name} policy={policy_name} episodes={len(episodes)}")
            for ep_idx, episode in enumerate(episodes):
                save_traj = ep_idx < trajectory_episodes
                row, traj_df = run_episode(
                    env_factory=env_factory,
                    scenario=scenario,
                    episode=episode,
                    policy_name=policy_name,
                    policy_runner=runner,
                    env_seed=scenario_seed + ep_idx,
                    save_trajectory=save_traj,
                )
                if spec is not None:
                    row["run_name"] = spec["run_name"]
                    row["ckpt"] = spec["ckpt"]
                else:
                    row["run_name"] = policy_name
                    row["ckpt"] = ""
                row["episode_id"] = ep_idx
                all_episode_rows.append(row)

                if traj_df is not None and len(traj_df) > 0:
                    traj_path = os.path.join(traj_dir, f"{policy_name}_episode_{ep_idx:04d}.csv")
                    traj_df.to_csv(traj_path, index=False)

        scenario_episode_df = pd.DataFrame([r for r in all_episode_rows if r["scenario_name"] == scenario.name and r["asset"] == asset])
        scenario_episode_df.to_csv(os.path.join(scenario_out, "per_episode.csv"), index=False)
        scenario_summary = summarise_results(scenario_episode_df)
        scenario_summary.to_csv(os.path.join(scenario_out, "summary.csv"), index=False)

    all_episode_df = pd.DataFrame(all_episode_rows)
    all_episode_df.to_csv(os.path.join(asset_out, "per_episode_all.csv"), index=False)
    all_summary_df = summarise_results(all_episode_df)
    all_summary_df.to_csv(os.path.join(asset_out, "summary_all.csv"), index=False)
    pd.DataFrame(all_episode_specs).to_csv(os.path.join(asset_out, "episodes_all.csv"), index=False)

    manifest = {
        "asset": asset,
        "data_root": data_root,
        "output_root": asset_out,
        "n_test_days": len(test_files),
        "n_episodes_per_scenario": n_episodes,
        "eval_seed": eval_seed,
        "scenarios": [asdict(s) for s in scenarios],
        "policies": ["TWAP"] + [m["policy_name"] for m in model_specs],
    }
    with open(os.path.join(asset_out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    print(f"  {os.path.join(asset_out, 'episodes_all.csv')}")
    print(f"  {os.path.join(asset_out, 'per_episode_all.csv')}")
    print(f"  {os.path.join(asset_out, 'summary_all.csv')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final evaluation runner for PPO-LSTM execution models.")
    p.add_argument("--asset", type=str, required=True, choices=["BTC", "DOGE"], help="Asset to evaluate.")
    p.add_argument("--data_root", type=str, required=True, help="Root folder with November/ December/ January/ parquet folders.")
    p.add_argument("--output_root", type=str, default="results", help="Where to write evaluation outputs.")
    p.add_argument("--n_episodes", type=int, default=100, help="Episodes per size scenario.")
    p.add_argument("--eval_seed", type=int, default=123, help="Global evaluation seed.")
    p.add_argument("--trajectory_episodes", type=int, default=0, help="How many episodes per policy/scenario to save step-level trajectories for.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_asset(
        asset=args.asset,
        data_root=args.data_root,
        output_root=args.output_root,
        n_episodes=args.n_episodes,
        eval_seed=args.eval_seed,
        trajectory_episodes=args.trajectory_episodes,
    )