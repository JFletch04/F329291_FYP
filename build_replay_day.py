# build_replay_day.py
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd


MS = 1000
GRID_MS = 5_000  # 5 seconds


def floor_to_grid(ts_ms: int, grid_ms: int = GRID_MS) -> int:
    return (ts_ms // grid_ms) * grid_ms


def ceil_to_grid(ts_ms: int, grid_ms: int = GRID_MS) -> int:
    return ((ts_ms + grid_ms - 1) // grid_ms) * grid_ms


@dataclass
class TradeAgg:
    trade_vol: float = 0.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    num_trades: int = 0
    pv_sum: float = 0.0  # sum(price * volume) for VWAP

    def add(self, price: float, volume: float, side: str):
        self.trade_vol += volume
        self.num_trades += 1
        self.pv_sum += price * volume
        if side.lower() == "buy":
            self.buy_vol += volume
        elif side.lower() == "sell":
            self.sell_vol += volume

    @property
    def signed_vol(self) -> float:
        return self.buy_vol - self.sell_vol

    @property
    def vwap(self) -> float:
        return self.pv_sum / self.trade_vol if self.trade_vol > 0 else float("nan")


class Book:
    """
    Maintain top-of-book maps from snapshot + delta updates.
    bids: price -> size (prices are floats)
    asks: price -> size
    """
    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.has_snapshot: bool = False

    def apply_snapshot(self, bids: List[List[str]], asks: List[List[str]]):
        self.bids.clear()
        self.asks.clear()
        for p_str, s_str in bids:
            p = float(p_str)
            s = float(s_str)
            if s > 0:
                self.bids[p] = s
        for p_str, s_str in asks:
            p = float(p_str)
            s = float(s_str)
            if s > 0:
                self.asks[p] = s
        self.has_snapshot = True

    def apply_delta(self, bids: List[List[str]], asks: List[List[str]]):
        # bids
        for p_str, s_str in bids:
            p = float(p_str)
            s = float(s_str)
            if s == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        # asks
        for p_str, s_str in asks:
            p = float(p_str)
            s = float(s_str)
            if s == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.bids or not self.asks:
            return None, None
        bb = max(self.bids.keys())
        ba = min(self.asks.keys())
        return bb, ba

    def top_n(self, n: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Return (bid_prices, bid_sizes, ask_prices, ask_sizes), length <= n each.
        """
        bid_items = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        ask_items = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        bid_prices = [p for p, _ in bid_items]
        bid_sizes = [s for _, s in bid_items]
        ask_prices = [p for p, _ in ask_items]
        ask_sizes = [s for _, s in ask_items]
        return bid_prices, bid_sizes, ask_prices, ask_sizes


def load_trades_csv(trades_csv_path: str) -> pd.DataFrame:
    """
    Expect columns: id,timestamp,price,volume,side,rpi (as in your snippet)
    """
    df = pd.read_csv(trades_csv_path)
    # Ensure types
    df["timestamp"] = df["timestamp"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["side"] = df["side"].astype(str)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_replay_day(
    lob_jsonl_path: str,
    trades_csv_path: str,
    out_parquet_path: str,
    top_n_levels: int = 50,
    grid_ms: int = GRID_MS,
    use_ts_field: str = "ts",  # use "ts" from your JSON
):
    trades = load_trades_csv(trades_csv_path)
    trade_i = 0
    n_trades = len(trades)

    # We will aggregate trades into buckets keyed by grid timestamp.
    # We'll do this on the fly using a moving pointer.
    def consume_trades_up_to(bucket_end_ts: int, bucket_start_ts: int) -> TradeAgg:
        nonlocal trade_i
        agg = TradeAgg()
        # Consume trades with (bucket_start_ts, bucket_end_ts]
        while trade_i < n_trades and trades.loc[trade_i, "timestamp"] <= bucket_end_ts:
            ts = int(trades.loc[trade_i, "timestamp"])
            if ts > bucket_start_ts:  # open on left
                agg.add(
                    price=float(trades.loc[trade_i, "price"]),
                    volume=float(trades.loc[trade_i, "volume"]),
                    side=str(trades.loc[trade_i, "side"]),
                )
            trade_i += 1
        return agg

    book = Book()
    rows = []

    next_grid_ts: Optional[int] = None
    last_seen_ts: Optional[int] = None

    with open(lob_jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)

            # timestamp to drive replay (ms)
            ts = int(msg[use_ts_field])
            last_seen_ts = ts

            msg_type = msg.get("type")
            data = msg.get("data", {})
            bids = data.get("b", []) or []
            asks = data.get("a", []) or []

            if msg_type == "snapshot":
                book.apply_snapshot(bids, asks)
                # initialise grid once we have first snapshot
                if next_grid_ts is None:
                    next_grid_ts = ceil_to_grid(ts, grid_ms)
            elif msg_type == "delta":
                if not book.has_snapshot:
                    # Ignore deltas before first snapshot
                    continue
                book.apply_delta(bids, asks)
            else:
                # unknown type
                continue

            if next_grid_ts is None or not book.has_snapshot:
                continue

            # Emit as many grid rows as we've crossed
            while next_grid_ts <= ts:
                bb, ba = book.best_bid_ask()
                if bb is None or ba is None or bb >= ba:
                    # Skip invalid book states (should be rare)
                    next_grid_ts += grid_ms
                    continue

                bid_p, bid_s, ask_p, ask_s = book.top_n(top_n_levels)

                mid = 0.5 * (bb + ba)
                spread = ba - bb

                bucket_end = next_grid_ts
                bucket_start = next_grid_ts - grid_ms
                agg = consume_trades_up_to(bucket_end, bucket_start)

                rows.append(
                    {
                        "ts": next_grid_ts,
                        "mid": mid,
                        "spread": spread,
                        "bid_prices": bid_p,
                        "bid_sizes": bid_s,
                        "ask_prices": ask_p,
                        "ask_sizes": ask_s,
                        "trade_vol": agg.trade_vol,
                        "buy_vol": agg.buy_vol,
                        "sell_vol": agg.sell_vol,
                        "signed_vol": agg.signed_vol,
                        "num_trades": agg.num_trades,
                        "trade_vwap": agg.vwap,
                    }
                )

                next_grid_ts += grid_ms

    if not rows:
        raise RuntimeError("No rows produced. Check file paths, timestamps, and snapshot presence.")

    df_out = pd.DataFrame(rows)
    # Some rows may have empty ladders if book got weird; you can drop those if you want:
    df_out = df_out[df_out["bid_prices"].map(len) > 0]
    df_out = df_out[df_out["ask_prices"].map(len) > 0]

    # Write parquet
    df_out.to_parquet(out_parquet_path, index=False)
    print(f"Wrote {len(df_out):,} rows to {out_parquet_path}")
    print(df_out.head(3))


if __name__ == "__main__":
    # TODO: Set these paths for your Jan 1 files
    LOB_JSONL = "/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_LOB/January/2026-01-01_BTCUSDT_ob200.data"
    TRADES_CSV = "/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_trades/January/BTCUSDT_2026-01-01.csv"
    OUT_PARQUET = "/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet"

    build_replay_day(
        lob_jsonl_path=LOB_JSONL,
        trades_csv_path=TRADES_CSV,
        out_parquet_path=OUT_PARQUET,
        top_n_levels=10,     # start with 50, you can bump to 200 later
        grid_ms=5_000,
        use_ts_field="ts",   # your JSON has "ts"
    )
