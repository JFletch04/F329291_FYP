#!/usr/bin/env python3
"""
Bybit orderbook dump player (orderbook.200.*)

Usage:
  python play_bybit_orderbook.py 2025-12-01_DOGEUSDT_ob200.data --levels 5 --speed 10

Controls:
  - Ctrl+C to stop
"""

import argparse
import json
import sys
import time
from typing import Dict, Tuple, Iterable, Optional


def iter_json_lines(path: str) -> Iterable[dict]:
    """Yield JSON objects from an NDJSON file (handles \n or \r\n)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Some Bybit dumps occasionally contain truncated/odd lines; skip safely.
                continue


def apply_updates(book: Dict[float, float], updates) -> None:
    """
    Apply L2 updates: updates is list of [price(str), size(str)].
    If size == 0 => remove level. Else set/replace.
    """
    for p_str, s_str in updates:
        try:
            p = float(p_str)
            s = float(s_str)
        except (TypeError, ValueError):
            continue
        if s == 0.0:
            book.pop(p, None)
        else:
            book[p] = s


def top_levels(
    bids: Dict[float, float], asks: Dict[float, float], n: int
) -> Tuple[list, list]:
    """Return sorted top-n bids (desc) and asks (asc) as [(price, size), ...]."""
    bid_lvls = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:n]
    ask_lvls = sorted(asks.items(), key=lambda x: x[0])[:n]
    return bid_lvls, ask_lvls


def fmt_num(x: Optional[float], width: int = 12, prec: int = 2) -> str:
    if x is None:
        return " " * width
    return f"{x:>{width},.{prec}f}"


def clear_screen() -> None:
    # ANSI clear + cursor home
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def render(
    bid_lvls: list,
    ask_lvls: list,
    ts_ms: Optional[int],
    levels: int,
    topic: Optional[str] = None,
) -> None:
    best_bid = bid_lvls[0][0] if bid_lvls else None
    best_ask = ask_lvls[0][0] if ask_lvls else None
    mid = (best_bid + best_ask) / 2 if (best_bid is not None and best_ask is not None) else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

    clear_screen()

    # Header
    if topic:
        print(f"{topic}")
    if ts_ms is not None:
        # Show local wall-clock for the event timestamp
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
        print(f"ts: {ts_ms} ({t})")
    print(f"best_bid: {best_bid}   best_ask: {best_ask}   mid: {mid}   spread: {spread}")
    print("-" * 78)

    # Table header
    print(f"{'ASK_SIZE':>14} {'ASK_PX':>14} | {'BID_PX':>14} {'BID_SIZE':>14}")
    print("-" * 78)

    # Align asks/bids by row index
    for i in range(levels):
        a_px = a_sz = b_px = b_sz = None
        if i < len(ask_lvls):
            a_px, a_sz = ask_lvls[i]
        if i < len(bid_lvls):
            b_px, b_sz = bid_lvls[i]

        # sizes often small; show more precision for size
        a_sz_str = fmt_num(a_sz, width=14, prec=6) if a_sz is not None else " " * 14
        b_sz_str = fmt_num(b_sz, width=14, prec=6) if b_sz is not None else " " * 14

        a_px_str = fmt_num(a_px, width=14, prec=5) if a_px is not None else " " * 14
        b_px_str = fmt_num(b_px, width=14, prec=5) if b_px is not None else " " * 14

        print(f"{a_sz_str} {a_px_str} | {b_px_str} {b_sz_str}")

    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to Bybit orderbook dump (.data)")
    ap.add_argument("--levels", type=int, default=20, help="How many levels to display per side")
    ap.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Playback speed multiplier (e.g., 1=real-time, 10=10x faster)",
    )
    ap.add_argument(
        "--max-sleep",
        type=float,
        default=0.25,
        help="Cap sleep per update (seconds) so slow gaps don't stall playback",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="Render every N messages (increase if rendering is too slow)",
    )
    args = ap.parse_args()

    bids: Dict[float, float] = {}
    asks: Dict[float, float] = {}
    last_ts: Optional[int] = None
    topic: Optional[str] = None

    msg_count = 0
    have_snapshot = False

    try:
        for msg in iter_json_lines(args.file):
            # Bybit shape (typical):
            # {"topic":"orderbook.200.BTCUSDT","ts":..., "type":"snapshot"/"delta", "data":{"b":[...],"a":[...]}}
            topic = msg.get("topic") or topic
            ts = msg.get("ts")
            mtype = msg.get("type")
            data = msg.get("data") or {}

            # Some dumps wrap data in a list; handle both
            if isinstance(data, list) and data:
                data = data[0]

            b_updates = data.get("b") or []
            a_updates = data.get("a") or []

            if mtype == "snapshot":
                bids.clear()
                asks.clear()
                apply_updates(bids, b_updates)
                apply_updates(asks, a_updates)
                have_snapshot = True
            elif mtype == "delta":
                # Only apply deltas after a snapshot
                if have_snapshot:
                    apply_updates(bids, b_updates)
                    apply_updates(asks, a_updates)
            else:
                # Unknown message type; skip
                continue

            # Playback timing
            if ts is not None and last_ts is not None and args.speed > 0:
                dt = (ts - last_ts) / 1000.0  # seconds
                if dt > 0:
                    sleep_s = min(dt / args.speed, args.max_sleep)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            if ts is not None:
                last_ts = ts

            msg_count += 1
            if msg_count % args.every == 0 and have_snapshot:
                bid_lvls, ask_lvls = top_levels(bids, asks, args.levels)
                render(bid_lvls, ask_lvls, ts_ms=ts, levels=args.levels, topic=topic)

    except KeyboardInterrupt:
        print("\nStopped.")
        return


if __name__ == "__main__":
    main()
