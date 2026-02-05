import os
import pandas as pd
import numpy as np

# --- Config ---
BUCKET_MINUTES = 5
BUCKETS_PER_DAY = 24 * 60 // BUCKET_MINUTES  # 288
BUCKET_MS = BUCKET_MINUTES * 60 * 1000       # 300,000
DAY_MS = 24 * 60 * 60 * 1000                 # 86,400,000


def Bin_Weight(csv_path: str) -> list[float]:
    """
    Returns 288 weights for ONE day:
    weight[t] = fraction of that day's traded volume in 5-min bucket t.
    """
    trades = pd.read_csv(csv_path)

    # Basic validation
    if "timestamp" not in trades.columns or "volume" not in trades.columns:
        raise ValueError(f"{csv_path} must include 'timestamp' and 'volume' columns")

    # Determine UTC day start from first timestamp (assumes file is one day)
    first_ts = int(trades["timestamp"].iloc[0])
    day_start = (first_ts // DAY_MS) * DAY_MS

    # Compute bucket index for each trade
    bucket_idx = ((trades["timestamp"] - day_start) // BUCKET_MS).astype(int)

    # Keep only valid [0..287]
    mask = (bucket_idx >= 0) & (bucket_idx < BUCKETS_PER_DAY)

    # Sum volume by bucket
    vols = np.zeros(BUCKETS_PER_DAY, dtype=float)
    grouped = trades.loc[mask].groupby(bucket_idx[mask])["volume"].sum()
    vols[grouped.index.values] = grouped.values

    total = vols.sum()
    if total <= 0:
        # Defensive: if a file is empty or corrupted
        return (np.ones(BUCKETS_PER_DAY) / BUCKETS_PER_DAY).tolist()

    weights = (vols / total).tolist()
    return weights


def Calc_All_Days_Bins_Avg_Weights(folder_path: str, use_median: bool = True) -> list[float]:
    """
    Builds an average 288-bucket curve across all CSVs in folder.
    Median is recommended for crypto (robust to outlier days).
    """
    all_days = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            all_days.append(Bin_Weight(file_path))

    if not all_days:
        raise ValueError(f"No CSVs found in {folder_path}")

    arr = np.array(all_days)  # shape (num_days, 288)

    if use_median:
        avg = np.median(arr, axis=0)
    else:
        avg = np.mean(arr, axis=0)

    # Renormalize (median won't sum perfectly to 1 sometimes)
    avg = avg / avg.sum()

    return avg.tolist()


def cumulative(weights: list[float]) -> list[float]:
    """
    Convert weights into cumulative curve: by bucket t, how much fraction should be done.
    """
    cum = np.cumsum(weights)
    cum[-1] = 1.0
    return cum.tolist()


# ---- quick check ----
if __name__ == "__main__":
    folder_path = "/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_trades/December"
    weights = Calc_All_Days_Bins_Avg_Weights(folder_path, use_median=True)
    print("len(weights) =", len(weights))
    print("sum(weights) =", sum(weights))
    print("min/max =", min(weights), max(weights))

