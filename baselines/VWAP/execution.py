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
    if len(cum) > 0:
        cum[-1] = 1.0
    return cum.tolist()


def renormalize(weights: List[float]) -> List[float]:
    arr = np.asarray(weights, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        if len(arr) == 0:
            return []
        return (np.ones(len(arr), dtype=float) / len(arr)).tolist()
    return (arr / total).tolist()


def infer_arrival_trade_price_at_ts(trades: pd.DataFrame, ts_ms: int) -> float:
    """
    Market trade price proxy at or immediately around ts_ms.
    """
    if len(trades) == 0:
        return 0.0
    before_or_at = trades.loc[trades["timestamp"] <= ts_ms]
    if len(before_or_at) > 0:
        return float(before_or_at["price"].iloc[-1])
    after = trades.loc[trades["timestamp"] > ts_ms]
    if len(after) > 0:
        return float(after["price"].iloc[0])
    return float(trades["price"].iloc[0])


# ----------------------------
# Trades utilities
# ----------------------------
def load_trades_day(trade_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trade_csv)
    required = {"timestamp", "price", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{trade_csv} missing required columns: {missing}")
    return df.sort_values("timestamp").reset_index(drop=True)


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
    Returns (avg_price, filled_qty). Does NOT mutate input dict.
    """
    if qty <= 0:
        return 0.0, 0.0
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
    Returns (avg_price, filled_qty). Does NOT mutate input dict.
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
def simulate_vwap_execution_window(
    trade_csv: Path,
    book_jsonl: Path,
    avg_curve_weights: List[float],
    Q: float,
    side: str,                        # "buy" or "sell"
    start_ts_ms: Optional[int] = None,
    horizon_ms: Optional[int] = None,
    participation_rate: Optional[float] = None,
    intra_bucket_slices: int = 1,
    force_terminal_completion: bool = True,
) -> dict:
    """
    Episode/window-based VWAP baseline.
    Key differences from the old version:
    - Executes only over a chosen window, not necessarily the full day
    - Uses arrival price at window start
    - Can force terminal completion at the horizon
    avg_curve_weights:
    - daily 288-bucket profile
    - sliced to the active window and renormalized
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

    trades = load_trades_day(trade_csv)
    if len(trades) == 0:
        raise ValueError(f"No trades found in {trade_csv}")

    first_ts = int(trades["timestamp"].iloc[0])
    last_ts = int(trades["timestamp"].iloc[-1])
    day_start = utc_day_start_ms(first_ts)
    day_end = day_start + DAY_MS

    if start_ts_ms is None:
        start_ts_ms = first_ts
    start_ts_ms = int(max(first_ts, start_ts_ms))

    if horizon_ms is None:
        end_ts_ms = min(day_end, last_ts)
    else:
        end_ts_ms = min(day_end, start_ts_ms + int(horizon_ms), last_ts)

    if end_ts_ms <= start_ts_ms:
        raise ValueError("Execution window is empty")

    arrival_price = infer_arrival_trade_price_at_ts(trades, start_ts_ms)

    # day-level market data
    mkt_vwap = market_vwap(trades)
    mkt_vols_day = market_volume_by_bucket(trades, day_start)

    # slice the volume curve to the active window and renormalise
    start_bucket = int((start_ts_ms - day_start) // BUCKET_MS)
    end_bucket = int((max(start_ts_ms, end_ts_ms - 1) - day_start) // BUCKET_MS)
    start_bucket = max(0, min(BUCKETS_PER_DAY - 1, start_bucket))
    end_bucket = max(0, min(BUCKETS_PER_DAY - 1, end_bucket))
    window_bucket_ids = list(range(start_bucket, end_bucket + 1))
    window_curve = [avg_curve_weights[b] for b in window_bucket_ids]
    window_curve = renormalize(window_curve)
    window_cum = cumulative(window_curve)

    # stream orderbook
    ob_iter = replay_orderbook(book_jsonl)
    current_ts: Optional[int] = None
    current_book: Optional[OrderBook] = None

    def advance_book_to(target_ts: int) -> None:
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
                break

    def exec_slice(ts_exec: int, qty: float, bucket: int, slice_idx: int) -> Tuple[float, float]:
        nonlocal current_book, current_ts
        if qty <= 0:
            return 0.0, 0.0
        advance_book_to(ts_exec)
        if current_book is None or current_book.best_bid() is None or current_book.best_ask() is None:
            return 0.0, 0.0
        if side == "buy":
            avg_px, got = walk_book_buy(current_book.asks, qty)
        else:
            avg_px, got = walk_book_sell(current_book.bids, qty)
        return avg_px, got

    fills: List[Fill] = []
    filled_qty = 0.0
    exec_notional = 0.0

    # execute within the selected window buckets
    for local_idx, b in enumerate(window_bucket_ids):
        target_cum = Q * window_cum[local_idx]
        deficit = target_cum - filled_qty
        if deficit <= 0:
            continue

        if participation_rate is not None:
            cap = participation_rate * float(mkt_vols_day[b])
            bucket_qty_allowed = max(0.0, cap)
        else:
            bucket_qty_allowed = deficit

        bucket_qty_to_execute = min(deficit, bucket_qty_allowed)
        if bucket_qty_to_execute <= 0:
            continue

        bucket_start = day_start + b * BUCKET_MS
        bucket_end = bucket_start + BUCKET_MS
        active_start = max(bucket_start, start_ts_ms)
        active_end = min(bucket_end, end_ts_ms)
        if active_end <= active_start:
            continue

        step = max((active_end - active_start) // intra_bucket_slices, 1)
        slice_times = [
            min(active_start + (i + 1) * step, active_end)
            for i in range(intra_bucket_slices)
        ]
        slice_qty = bucket_qty_to_execute / intra_bucket_slices

        for s_idx, ts_exec in enumerate(slice_times):
            if filled_qty >= Q:
                break
            if s_idx == intra_bucket_slices - 1:
                # last slice in bucket takes whatever remains to avoid rounding drift
                already_done_in_bucket = slice_qty * (intra_bucket_slices - 1)
                slice_qty_eff = bucket_qty_to_execute - already_done_in_bucket
            else:
                slice_qty_eff = slice_qty
            slice_qty_eff = min(slice_qty_eff, Q - filled_qty)
            if slice_qty_eff <= 0:
                continue
            avg_px, got = exec_slice(ts_exec, slice_qty_eff, b, s_idx)
            if got > 0:
                fills.append(Fill(ts=ts_exec, qty=got, avg_price=avg_px, bucket=b, slice_idx=s_idx))
                filled_qty += got
                exec_notional += got * avg_px
            if filled_qty >= Q:
                break

    # terminal liquidation to match horizon-based RL evaluation
    terminal_liq_qty = max(0.0, Q - filled_qty)
    terminal_liq_avg_px = 0.0
    if force_terminal_completion and terminal_liq_qty > 0:
        avg_px, got = exec_slice(end_ts_ms, terminal_liq_qty, end_bucket, intra_bucket_slices)
        if got > 0:
            fills.append(
                Fill(
                    ts=end_ts_ms,
                    qty=got,
                    avg_price=avg_px,
                    bucket=end_bucket,
                    slice_idx=intra_bucket_slices,
                )
            )
            filled_qty += got
            exec_notional += got * avg_px
            terminal_liq_avg_px = avg_px

    exec_avg_price = (exec_notional / filled_qty) if filled_qty > 0 else 0.0

    if side == "buy":
        slippage_vs_vwap = exec_avg_price - mkt_vwap
        implementation_shortfall = exec_avg_price - arrival_price
    else:
        slippage_vs_vwap = mkt_vwap - exec_avg_price
        implementation_shortfall = arrival_price - exec_avg_price

    completion_rate = filled_qty / Q if Q > 0 else 0.0

    return {
        "day_start": day_start,
        "start_ts_ms": start_ts_ms,
        "end_ts_ms": end_ts_ms,
        "window_horizon_ms": end_ts_ms - start_ts_ms,
        "side": side,
        "Q": Q,
        "filled_qty": filled_qty,
        "completion_rate": completion_rate,
        "exec_avg_price": exec_avg_price,
        "market_vwap": mkt_vwap,
        "arrival_price": arrival_price,
        "implementation_shortfall": implementation_shortfall,
        "slippage_vs_vwap": slippage_vs_vwap,
        "participation_rate": participation_rate,
        "intra_bucket_slices": intra_bucket_slices,
        "start_bucket": start_bucket,
        "end_bucket": end_bucket,
        "force_terminal_completion": force_terminal_completion,
        "terminal_liq_qty": terminal_liq_qty,
        "terminal_liq_avg_px": terminal_liq_avg_px,
        "fills": fills,
    }


# ----------------------------
# Backward-compatible wrapper
# ----------------------------
def simulate_vwap_execution_day(
    trade_csv: Path,
    book_jsonl: Path,
    avg_curve_weights: List[float],
    Q: float,
    side: str,
    participation_rate: Optional[float] = None,
    intra_bucket_slices: int = 1,
) -> dict:
    """
    Original full-day API retained for compatibility.
    """
    return simulate_vwap_execution_window(
        trade_csv=trade_csv,
        book_jsonl=book_jsonl,
        avg_curve_weights=avg_curve_weights,
        Q=Q,
        side=side,
        start_ts_ms=None,
        horizon_ms=None,
        participation_rate=participation_rate,
        intra_bucket_slices=intra_bucket_slices,
        force_terminal_completion=True,
    )


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