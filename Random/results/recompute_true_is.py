from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


INPUT_CSV = Path("BTC/per_episode_all.csv")
OUTPUT_EPISODE_CSV = Path("BTC/per_episode_all_with_true_is.csv")
OUTPUT_SUMMARY_CSV = Path("BTC/summary_true_is.csv")


def compute_true_is_bps(row: pd.Series) -> float:
    arrival_mid = float(row["arrival_mid"])
    exec_vwap = float(row["exec_vwap"])
    side = str(row["side"]).lower().strip()

    if arrival_mid <= 0 or np.isnan(arrival_mid) or np.isnan(exec_vwap):
        return np.nan

    if side == "buy":
        return 1e4 * (exec_vwap - arrival_mid) / arrival_mid
    elif side == "sell":
        return 1e4 * (arrival_mid - exec_vwap) / arrival_mid
    else:
        return np.nan


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["asset", "scenario_name", "target_qty", "policy_name"]

    for keys, g in df.groupby(group_cols, dropna=False):
        asset, scenario_name, target_qty, policy_name = keys
        vals = g["true_is_bps"].dropna().values

        row = {
            "asset": asset,
            "scenario_name": scenario_name,
            "target_qty": float(target_qty),
            "policy_name": policy_name,
            "n_episodes": int(len(g)),
            "mean_completion": float(g["completion"].mean()) if len(g) else np.nan,
            "completion_rate_100pct": float(g["completion_100pct"].mean()) if len(g) else np.nan,
            "mean_true_is_bps": float(np.mean(vals)) if len(vals) else np.nan,
            "std_true_is_bps": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median_true_is_bps": float(np.median(vals)) if len(vals) else np.nan,
            "p90_true_is_bps": float(np.percentile(vals, 90)) if len(vals) else np.nan,
            "p95_true_is_bps": float(np.percentile(vals, 95)) if len(vals) else np.nan,
            "p99_true_is_bps": float(np.percentile(vals, 99)) if len(vals) else np.nan,
            # keep old metric too for reference
            "mean_env_is_bps": float(g["is_bps"].mean()) if len(g) else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["asset", "target_qty", "scenario_name", "mean_true_is_bps", "policy_name"]
        ).reset_index(drop=True)
    return out


def main():
    df = pd.read_csv(INPUT_CSV)

    df["true_is_bps"] = df.apply(compute_true_is_bps, axis=1)

    summary = summarize(df)

    df.to_csv(OUTPUT_EPISODE_CSV, index=False)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nSaved:")
    print(f"  {OUTPUT_EPISODE_CSV}")
    print(f"  {OUTPUT_SUMMARY_CSV}")

    print("\nExample summary:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()