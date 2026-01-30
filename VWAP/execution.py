from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from orderbook import replay_orderbook, OrderBook


# ----------------------------
# Config / constants
# ----------------------------
BUCKET_MINUTES = 5
BUCKET_MS = BUCKET_MINUTES * 60 * 1000        # 300,000 ms
BUCKETS_PER_DAY = 24 * 60 // BUCKET_MINUTES   # 288
DAY_MS = 24 * 60 * 60 * 1000                  # 86,400,000 ms


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Fill:
    ts: int
    qty: float
    avg_price: float
    bucket: int
    slice_idx: int


# ----------------------------
# Helpers: time / curve
# ----------------------------
def utc_day_start_ms(ts_ms: int) -> int:
    return (ts_ms // DAY_MS) * DAY_MS


def cumulative(weights: List[float]) -> List[float]:
    cum = np.cumsum(weights).astype(float)
    cum[-1] = 1.0
    return cum.tolist()


# ----------------------------
# Trades utilities
# ----------------------------
def load_trades_day(trade_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trade_csv)
    required = {"timestamp", "price", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{trade_csv} missing required columns: {missing}")
    return df


def market_vwap(df_trades: pd.DataFrame) -> float:
    notional = (df_trades["price"] * df_trades["volume"]).sum()
    vol = df_trades["volume"].sum()
    return float(notional / vol) if vol > 0 else 0.0


def market_volume_by_bucket(
    df_trades: pd.DataFrame,
    day_start: int,
    bucket_ms: int = BUCKET_MS,
    buckets_per_day: int = BUCKETS_PER_DAY,
) -> np.ndarray:
    bucket_idx = ((df_trades["timestamp"] - day_start) // bucket_ms).astype(int)
    mask = (bucket_idx >= 0) & (bucket_idx < buckets_per_day)

    vols = np.zeros(buckets_per_day, dtype=float)
    grouped = df_trades.loc[mask].groupby(bucket_idx[mask])["volume"].sum()
    vols[grouped.index.values] = grouped.values
    return vols


# ----------------------------
# L2 fill model (walk the book)
# ----------------------------
def walk_book_buy(asks: Dict[float, float], qty: float) -> Tuple[float, float]:
    """
    Consume ask depth from best ask upwards.
    Returns (avg_price, filled_qty). Does NOT mutate input dict (works on a copy).
    """
    if qty <= 0:
        return 0.0, 0.0

    # copy so we don't change the live replay book
    local = dict(asks)

    remaining = qty
    cost = 0.0
    filled = 0.0

    for price in sorted(local.keys()):
        if remaining <= 0:
            break
        avail = local[price]
        take = min(avail, remaining)
        if take > 0:
            cost += price * take
            filled += take
            remaining -= take

    avg = (cost / filled) if filled > 0 else 0.0
    return avg, filled


def walk_book_sell(bids: Dict[float, float], qty: float) -> Tuple[float, float]:
    """
    Consume bid depth from best bid downwards.
    Returns (avg_price, filled_qty). Does NOT mutate input dict (works on a copy).
    """
    if qty <= 0:
        return 0.0, 0.0

    local = dict(bids)

    remaining = qty
    proceeds = 0.0
    filled = 0.0

    for price in sorted(local.keys(), reverse=True):
        if remaining <= 0:
            break
        avail = local[price]
        take = min(avail, remaining)
        if take > 0:
            proceeds += price * take
            filled += take
            remaining -= take

    avg = (proceeds / filled) if filled > 0 else 0.0
    return avg, filled


# ----------------------------
# Core VWAP execution simulation
# ----------------------------
def simulate_vwap_execution_day(
    trade_csv: Path,
    book_jsonl: Path,
    avg_curve_weights: List[float],
    Q: float,
    side: str,  # "buy" or "sell"
    participation_rate: Optional[float] = None,  # e.g. 0.1 for 10% cap; None = no cap
    intra_bucket_slices: int = 1,                # execute within each 5-min bucket in N slices
) -> dict:
    """
    VWAP baseline:
    - Uses avg_curve_weights (len=288) to set target schedule
    - Optionally caps execution per bucket by participation_rate * market_volume_in_bucket
    - Prices fills using L2 book depth at execution timestamps (walk the book)
    - Returns metrics + fills

    IMPORTANT:
    - Fill simulation uses a copy of the book at that time (doesn't alter future book replay).
      This is a standard baseline assumption.
    """

    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    if len(avg_curve_weights) != BUCKETS_PER_DAY:
        raise ValueError(f"avg_curve_weights must have length {BUCKETS_PER_DAY}")

    if Q <= 0:
        raise ValueError("Q must be > 0")

    if intra_bucket_slices < 1:
        raise ValueError("intra_bucket_slices must be >= 1")

    # Load trades day
    trades = load_trades_day(trade_csv)
    first_ts = int(trades["timestamp"].iloc[0])
    day_start = utc_day_start_ms(first_ts)

    # Market benchmarks / participation caps
    mkt_vwap = market_vwap(trades)
    mkt_vols = market_volume_by_bucket(trades, day_start)

    # Build cumulative schedule from the average curve
    cum_w = cumulative(avg_curve_weights)  # length 288, ends at 1.0

    # Prepare execution times
    # We'll execute at slice boundaries inside each bucket, e.g. for intra_bucket_slices=5:
    # bucket start + 1/5, 2/5, ... bucket end
    def slice_timestamps_for_bucket(b: int) -> List[int]:
        bucket_start = day_start + b * BUCKET_MS
        return [
            bucket_start + (i + 1) * (BUCKET_MS // intra_bucket_slices)
            for i in range(intra_bucket_slices)
        ]

    # Stream orderbook and keep latest book <= current execution timestamp
    ob_iter = replay_orderbook(book_jsonl)
    current_ts: Optional[int] = None
    current_book: Optional[OrderBook] = None

    def advance_book_to(target_ts: int) -> None:
        """
        Advances the replay stream so that current_book is the latest book with ts <= target_ts.
        """
        nonlocal current_ts, current_book

        while True:
            try:
                ts, book = next(ob_iter)
            except StopIteration:
                break

            if ts <= target_ts:
                current_ts = ts
                current_book = book
                continue
            else:
                # We've gone past target_ts; keep the last one we had (<= target_ts)
                break

    fills: List[Fill] = []
    filled_qty = 0.0
    exec_notional = 0.0

    # Execute bucket by bucket
    for b in range(BUCKETS_PER_DAY):
        # Target cumulative quantity by end of this bucket
        target_cum = Q * cum_w[b]
        deficit = target_cum - filled_qty
        if deficit <= 0:
            continue

        # Participation cap for this bucket
        if participation_rate is not None:
            cap = participation_rate * float(mkt_vols[b])
            # If cap is very small/zero (quiet bucket), you may do nothing in this bucket.
            bucket_qty_allowed = max(0.0, cap)
        else:
            bucket_qty_allowed = deficit  # no cap

        bucket_qty_to_execute = min(deficit, bucket_qty_allowed)
        if bucket_qty_to_execute <= 0:
            continue

        # Split within-bucket
        slice_times = slice_timestamps_for_bucket(b)
        slice_qty = bucket_qty_to_execute / intra_bucket_slices

        for s_idx, ts_exec in enumerate(slice_times):
            if filled_qty >= Q:
                break

            # If last slice and rounding left something, take remainder to hit planned bucket qty
            if s_idx == intra_bucket_slices - 1:
                # remaining planned for this bucket
                already_done_in_bucket = slice_qty * (intra_bucket_slices - 1)
                slice_qty_eff = bucket_qty_to_execute - already_done_in_bucket
            else:
                slice_qty_eff = slice_qty

            # Also don't exceed total Q
            slice_qty_eff = min(slice_qty_eff, Q - filled_qty)
            if slice_qty_eff <= 0:
                continue

            # Advance order book to this exec time
            advance_book_to(ts_exec)
            if current_book is None or (current_book.best_bid() is None) or (current_book.best_ask() is None):
                # no usable book yet; skip
                continue

            # Price + fill from L2
            if side == "buy":
                avg_px, got = walk_book_buy(current_book.asks, slice_qty_eff)
            else:
                avg_px, got = walk_book_sell(current_book.bids, slice_qty_eff)

            if got > 0:
                fills.append(Fill(ts=ts_exec, qty=got, avg_price=avg_px, bucket=b, slice_idx=s_idx))
                filled_qty += got
                exec_notional += got * avg_px

        if filled_qty >= Q:
            break

    exec_avg_price = (exec_notional / filled_qty) if filled_qty > 0 else 0.0

    # Slippage vs market VWAP
    # For buys: paying above VWAP is bad -> positive slippage is bad
    # For sells: receiving below VWAP is bad -> positive slippage is bad (we flip)
    if side == "buy":
        slippage_vs_vwap = exec_avg_price - mkt_vwap
    else:
        slippage_vs_vwap = mkt_vwap - exec_avg_price

    completion_rate = filled_qty / Q

    return {
        "day_start": day_start,
        "side": side,
        "Q": Q,
        "filled_qty": filled_qty,
        "completion_rate": completion_rate,
        "exec_avg_price": exec_avg_price,
        "market_vwap": mkt_vwap,
        "slippage_vs_vwap": slippage_vs_vwap,
        "participation_rate": participation_rate,
        "intra_bucket_slices": intra_bucket_slices,
        "fills": fills,
    }


# ----------------------------
# Output utilities
# ----------------------------
def save_fills_to_csv(fills: List[Fill], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "qty", "avg_price", "bucket", "slice_idx"])
        writer.writeheader()
        for fill in fills:
            writer.writerow(asdict(fill))


# ----------------------------
# Example CLI-ish usage
# ----------------------------
if __name__ == "__main__":
    # Example: adapt these paths
    trade_csv = Path("/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_trades/January/2025-01-15.csv")
    book_jsonl = Path("/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_lob/January/2025-01-15.jsonl")

    # You should load your avg_curve_weights from your curve module output.
    # For quick testing, here's a flat curve (NOT recommended for real use):
    avg_curve_weights = (np.ones(BUCKETS_PER_DAY) / BUCKETS_PER_DAY).tolist()

    result = simulate_vwap_execution_day(
        trade_csv=trade_csv,
        book_jsonl=book_jsonl,
        avg_curve_weights=avg_curve_weights,
        Q=10.0,
        side="buy",
        participation_rate=0.10,   # 10% of market volume per 5-min bucket
        intra_bucket_slices=5,     # execute 5 times within each 5-min bucket
    )

    print("Filled:", result["filled_qty"], "Completion:", result["completion_rate"])
    print("Exec Avg:", result["exec_avg_price"])
    print("Mkt VWAP:", result["market_vwap"])
    print("Slippage vs VWAP:", result["slippage_vs_vwap"])

    save_fills_to_csv(result["fills"], Path("fills.csv"))
    print("Saved fills.csv")
