import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# --- Core fill simulator (market order, walk the book) ---

def walk_book_market(qty: float, prices: List[float], sizes: List[float]) -> Tuple[float, float]:
    """
    Walk the book to fill 'qty' using given ladder (prices asc for asks, desc for bids).
    Returns (filled_qty, avg_fill_price).
    If insufficient depth, fills partially.
    """
    if qty <= 0:
        return 0.0, float("nan")

    filled = 0.0
    notional = 0.0

    for p, s in zip(prices, sizes):
        if filled >= qty:
            break
        take = min(qty - filled, float(s))
        if take > 0:
            filled += take
            notional += take * float(p)

    if filled == 0.0:
        return 0.0, float("nan")

    return filled, notional / filled


@dataclass
class EpisodeResult:
    start_idx: int
    side: str
    target_qty: float
    filled_qty: float
    arrival_mid: float
    exec_vwap: float
    is_cash: float
    is_bps: float
    completion: float


def compute_is(side: str, filled_qty: float, exec_vwap: float, arrival_mid: float) -> float:
    """
    Implementation shortfall in cash units (quote currency).
    For buys:  IS = filled_qty * (exec_vwap - arrival_mid)
    For sells: IS = filled_qty * (arrival_mid - exec_vwap)
    """
    if math.isnan(exec_vwap) or filled_qty <= 0:
        return 0.0

    side = side.lower()
    if side == "buy":
        return filled_qty * (exec_vwap - arrival_mid)
    elif side == "sell":
        return filled_qty * (arrival_mid - exec_vwap)
    else:
        raise ValueError("side must be 'buy' or 'sell'")


# --- TWAP executor ---

def run_twap_episode(
    df: pd.DataFrame,
    start_idx: int,
    horizon_steps: int = 180,          # 15 minutes at 5s
    side: str = "buy",
    target_qty: float = 0.5,           # base units (e.g., BTC)
    taker_fee_rate: float = 0.0,       # set later (e.g., 0.0006)
    require_full_depth: bool = False,  # if True: drop episode if any step can't fill child
) -> EpisodeResult:
    """
    Executes a TWAP schedule: target_qty / horizon_steps each step.
    Uses the replay table ladders for fill simulation.
    """

    end_idx = start_idx + horizon_steps - 1
    if end_idx >= len(df):
        raise IndexError("Episode would run past end of dataframe")

    # Arrival price: mid at episode start
    arrival_mid = float(df.iloc[start_idx]["mid"])

    remaining = float(target_qty)
    filled_total = 0.0
    notional_total = 0.0

    child_qty = float(target_qty) / float(horizon_steps)

    for i in range(start_idx, start_idx + horizon_steps):
        row = df.iloc[i]

        if remaining <= 0:
            break

        q = min(child_qty, remaining)

        if side.lower() == "buy":
            prices = row["ask_prices"]
            sizes = row["ask_sizes"]
        else:
            prices = row["bid_prices"]
            sizes = row["bid_sizes"]

        filled, avg_p = walk_book_market(q, prices, sizes)

        if require_full_depth and filled < q:
            # If you want only episodes where TWAP fully fills each child slice
            break

        if filled > 0:
            filled_total += filled
            notional_total += filled * avg_p
            remaining -= filled

    exec_vwap = (notional_total / filled_total) if filled_total > 0 else float("nan")

    # Fees (taker fee on notional)
    fee_cash = taker_fee_rate * notional_total

    # IS cash (add fees as cost)
    is_cash = compute_is(side, filled_total, exec_vwap, arrival_mid) + fee_cash

    # IS in bps of notional at arrival (use filled_total to avoid division by intended qty when partial)
    denom = filled_total * arrival_mid
    is_bps = (1e4 * is_cash / denom) if denom > 0 else float("nan")

    completion = filled_total / target_qty if target_qty > 0 else float("nan")

    return EpisodeResult(
        start_idx=start_idx,
        side=side,
        target_qty=target_qty,
        filled_qty=filled_total,
        arrival_mid=arrival_mid,
        exec_vwap=exec_vwap,
        is_cash=is_cash,
        is_bps=is_bps,
        completion=completion,
    )


def sample_start_indices(df: pd.DataFrame, horizon_steps: int, n: int, seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    max_start = len(df) - horizon_steps
    return [rng.randint(0, max_start) for _ in range(n)]


def main():
    REPLAY_PARQUET = "/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet"

    df = pd.read_parquet(REPLAY_PARQUET)

    # Quick sanity checks
    assert "ask_prices" in df.columns and "bid_prices" in df.columns, "Missing ladders in replay table"
    assert len(df) >= 180, "Not enough rows for a 15-minute episode"

    # ---- Run a single episode (pick a start idx) ----
    start_idx = 1000  # change this, or set to None to random sample below
    res = run_twap_episode(
        df,
        start_idx=start_idx,
        horizon_steps=180,
        side="buy",
        target_qty=0.5,          # BTC quantity for now (you can switch to notional later)
        taker_fee_rate=0.0,
        require_full_depth=False
    )
    print("\nSingle TWAP episode")
    print(res)

    # ---- Run many random episodes to see distribution ----
    n_episodes = 200
    starts = sample_start_indices(df, horizon_steps=180, n=n_episodes, seed=7)

    results = []
    for s in starts:
        side = "buy" if random.random() < 0.5 else "sell"
        r = run_twap_episode(
            df,
            start_idx=s,
            horizon_steps=180,
            side=side,
            target_qty=0.5,
            taker_fee_rate=0.0,
            require_full_depth=False
        )
        results.append(r)

    # Summarise
    is_bps = np.array([r.is_bps for r in results if not math.isnan(r.is_bps)])
    comp = np.array([r.completion for r in results if not math.isnan(r.completion)])

    print("\nTWAP summary over random episodes")
    print(f"Episodes: {len(results)}")
    print(f"Mean IS (bps): {is_bps.mean():.3f}")
    print(f"Median IS (bps): {np.median(is_bps):.3f}")
    print(f"95th pct IS (bps): {np.percentile(is_bps, 95):.3f}")
    print(f"Mean completion: {comp.mean():.3f}")
    print(f"Min completion: {comp.min():.3f}")


if __name__ == "__main__":
    main()
