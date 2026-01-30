from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Core helpers
# ----------------------------
def safe_div(n: float, d: float) -> float:
    return float(n / d) if d and d != 0 else 0.0


def exec_vwap_from_fills(fills: List[Dict] | List[object]) -> Tuple[float, float, float]:
    """
    Returns (exec_vwap, filled_qty, exec_notional)
    Accepts fills as:
      - list of dicts with keys: qty, avg_price
      - list of objects with attrs: qty, avg_price
    """
    notional = 0.0
    qty = 0.0
    for f in fills:
        if isinstance(f, dict):
            q = float(f["qty"])
            p = float(f["avg_price"])
        else:
            q = float(getattr(f, "qty"))
            p = float(getattr(f, "avg_price"))
        qty += q
        notional += q * p

    vwap = safe_div(notional, qty)
    return vwap, qty, notional


def market_vwap_from_trades(trades: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns (market_vwap, market_vol, market_notional)
    trades needs columns: price, volume
    """
    notional = float((trades["price"] * trades["volume"]).sum())
    vol = float(trades["volume"].sum())
    vwap = safe_div(notional, vol)
    return vwap, vol, notional


def compute_bucket_volumes(
    trades: pd.DataFrame,
    day_start: int,
    bucket_ms: int,
    buckets_per_day: int,
) -> np.ndarray:
    """
    Returns array of market volume per bucket.
    """
    bucket_idx = ((trades["timestamp"] - day_start) // bucket_ms).astype(int)
    mask = (bucket_idx >= 0) & (bucket_idx < buckets_per_day)

    vols = np.zeros(buckets_per_day, dtype=float)
    grouped = trades.loc[mask].groupby(bucket_idx[mask])["volume"].sum()
    vols[grouped.index.values] = grouped.values
    return vols


def fills_by_bucket(
    fills: List[Dict] | List[object],
    buckets_per_day: int,
) -> np.ndarray:
    """
    Returns array of executed qty per bucket based on fill.bucket.
    """
    out = np.zeros(buckets_per_day, dtype=float)
    for f in fills:
        if isinstance(f, dict):
            b = int(f["bucket"])
            q = float(f["qty"])
        else:
            b = int(getattr(f, "bucket"))
            q = float(getattr(f, "qty"))
        if 0 <= b < buckets_per_day:
            out[b] += q
    return out


# ----------------------------
# Main metric computation
# ----------------------------
def compute_execution_metrics(
    *,
    fills: List[Dict] | List[object],
    trades: pd.DataFrame,
    side: str,                    # "buy" or "sell"
    Q: float,                     # parent order size
    day_start: int,
    bucket_ms: int,
    buckets_per_day: int,
    arrival_price: Optional[float] = None,  # if None, use first trade price as a proxy
) -> Dict[str, float | np.ndarray]:
    """
    Produces a dict of standard execution metrics.

    Notes:
    - Slippage vs VWAP is side-aware so + means worse:
        buy: exec - mkt_vwap
        sell: mkt_vwap - exec
    - Implementation shortfall (IS) is also side-aware and uses arrival_price:
        buy: exec - arrival
        sell: arrival - exec
    """

    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    # Market VWAP
    mkt_vwap, mkt_vol, mkt_notional = market_vwap_from_trades(trades)

    # Execution VWAP
    exec_vwap, exec_qty, exec_notional = exec_vwap_from_fills(fills)

    # Arrival price
    if arrival_price is None:
        # simplest proxy: first trade price of the day
        arrival_price = float(trades["price"].iloc[0]) if len(trades) else 0.0

    # Side-aware slippage (positive = worse)
    if side == "buy":
        slippage_vs_vwap = exec_vwap - mkt_vwap
        implementation_shortfall = exec_vwap - arrival_price
    else:
        slippage_vs_vwap = mkt_vwap - exec_vwap
        implementation_shortfall = arrival_price - exec_vwap

    completion_rate = safe_div(exec_qty, Q)

    # Participation overall
    participation_overall = safe_div(exec_qty, mkt_vol)

    # Participation per bucket
    mkt_bucket_vols = compute_bucket_volumes(trades, day_start, bucket_ms, buckets_per_day)
    exec_bucket_qty = fills_by_bucket(fills, buckets_per_day)

    # Avoid division by zero per bucket
    part_per_bucket = np.zeros(buckets_per_day, dtype=float)
    nonzero = mkt_bucket_vols > 0
    part_per_bucket[nonzero] = exec_bucket_qty[nonzero] / mkt_bucket_vols[nonzero]

    return {
        "market_vwap": mkt_vwap,
        "market_volume": mkt_vol,
        "market_notional": mkt_notional,
        "exec_vwap": exec_vwap,
        "exec_qty": exec_qty,
        "exec_notional": exec_notional,
        "completion_rate": completion_rate,
        "slippage_vs_vwap": slippage_vs_vwap,
        "arrival_price": float(arrival_price),
        "implementation_shortfall": implementation_shortfall,
        "participation_overall": participation_overall,
        "market_bucket_vols": mkt_bucket_vols,
        "exec_bucket_qty": exec_bucket_qty,
        "participation_per_bucket": part_per_bucket,
    }


# ----------------------------
# Pretty-print helper (optional)
# ----------------------------
def summarize_metrics(metrics: Dict[str, float | np.ndarray]) -> str:
    """
    Returns a human-readable multi-line summary.
    """
    lines = []
    lines.append(f"Market VWAP:               {metrics['market_vwap']:.6f}")
    lines.append(f"Execution VWAP:            {metrics['exec_vwap']:.6f}")
    lines.append(f"Slippage vs VWAP:          {metrics['slippage_vs_vwap']:.6f} (positive=worse)")
    lines.append(f"Arrival price:             {metrics['arrival_price']:.6f}")
    lines.append(f"Implementation shortfall:  {metrics['implementation_shortfall']:.6f} (positive=worse)")
    lines.append(f"Executed qty:              {metrics['exec_qty']:.6f}")
    lines.append(f"Parent order qty:          {metrics.get('Q', 'n/a')}")
    lines.append(f"Completion rate:           {metrics['completion_rate']:.2%}")
    lines.append(f"Participation overall:     {metrics['participation_overall']:.2%}")
    return "\n".join(lines)
