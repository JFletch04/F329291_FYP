from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from execution import (
    simulate_vwap_execution_window,
    BUCKET_MS,
    BUCKETS_PER_DAY,
)
from metrics import compute_execution_metrics


# ============================================================
# USER CONFIG
# ============================================================

SYMBOL = "BTCUSDT"   # "BTCUSDT" or "DOGEUSDT"
MONTH = "January"

BASE_DIR = Path("./data")

TRADES_DIR = BASE_DIR / f"{SYMBOL}_trades" / MONTH
LOB_DIR = BASE_DIR / f"{SYMBOL}_LOB" / MONTH

TRAIN_DAYS = 25
TEST_DAYS = 6

SIDE = "buy"

# use quantity consistent with your scenario when possible
USE_FIXED_QTY = True
FIXED_QTY = 10000.0

NOTIONAL_USDT = 10_000_000

PARTICIPATION_RATE = 0.05
INTRA_BUCKET_SLICES = 5

# DRL horizon: 4320 steps * 5 seconds = 6 hours
GRID_MS = 5_000
HORIZON_STEPS = 4320
HORIZON_MS = HORIZON_STEPS * GRID_MS

# "day_start" = start at first trade of the day
# "hour_utc"  = start at a fixed UTC hour
WINDOW_START_MODE = "day_start"
START_HOUR_UTC = 0

FORCE_TERMINAL_COMPLETION = True

OUT_RESULTS_CSV = Path(f"vwap_test_results_{SYMBOL}_{MONTH}.csv")
SAVE_FILLS = False
FILLS_DIR = Path("fills")


# ============================================================
# Helpers
# ============================================================

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def extract_date(text: str) -> str:
    m = DATE_RE.search(text)
    if not m:
        raise ValueError(f"Could not find YYYY-MM-DD in: {text}")
    return m.group(0)


def expected_lob_name(day: str) -> str:
    return f"{day}_{SYMBOL}_ob200.data"


def match_lob_file_for_trade(trade_file: Path, lob_dir: Path) -> Optional[Path]:
    day = extract_date(trade_file.stem)

    exact = lob_dir / expected_lob_name(day)
    if exact.exists():
        return exact

    candidates = []
    for p in lob_dir.iterdir():
        name = p.name
        if day in name and SYMBOL in name and "ob200" in name:
            candidates.append(p)

    if not candidates:
        return None

    # prefer shortest filename match if multiple candidates
    candidates.sort(key=lambda x: len(x.name))
    return candidates[0]


def list_trade_files_sorted_by_date(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".csv"]
    files.sort(key=lambda p: extract_date(p.stem))
    return files


def lob_dir_date_range_hint(lob_dir: Path) -> str:
    dates = []
    for p in lob_dir.iterdir():
        m = DATE_RE.search(p.name)
        if m:
            dates.append(m.group(0))
    if not dates:
        return "No dated files found in LOB_DIR."
    dates.sort()
    return f"LOB_DIR date range looks like: {dates[0]} .. {dates[-1]}"


def filter_matched_days(trade_files: List[Path], lob_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for tf in trade_files:
        lf = match_lob_file_for_trade(tf, lob_dir)
        if lf is not None:
            pairs.append((tf, lf))
    pairs.sort(key=lambda pair: extract_date(pair[0].stem))
    return pairs


def choose_start_ts(trades_df: pd.DataFrame) -> int:
    first_ts = int(trades_df["timestamp"].iloc[0])

    if WINDOW_START_MODE == "day_start":
        return first_ts

    if WINDOW_START_MODE == "hour_utc":
        day_start = (first_ts // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)
        target = day_start + START_HOUR_UTC * 60 * 60 * 1000
        return max(first_ts, target)

    raise ValueError(f"Unsupported WINDOW_START_MODE={WINDOW_START_MODE}")


# ============================================================
# Main
# ============================================================

def main():
    print(f"SYMBOL={SYMBOL}  MONTH={MONTH}")
    print(f"TRADES_DIR={TRADES_DIR}")
    print(f"LOB_DIR={LOB_DIR}")
    print(f"SIDE={SIDE}")
    print(f"HORIZON_STEPS={HORIZON_STEPS}  HORIZON_MS={HORIZON_MS}")
    print(f"USE_FIXED_QTY={USE_FIXED_QTY}  FIXED_QTY={FIXED_QTY}")

    if not TRADES_DIR.exists():
        raise ValueError(f"TRADES_DIR does not exist: {TRADES_DIR}")
    if not LOB_DIR.exists():
        raise ValueError(f"LOB_DIR does not exist: {LOB_DIR}")

    print(lob_dir_date_range_hint(LOB_DIR))

    trade_files_all = list_trade_files_sorted_by_date(TRADES_DIR)
    matched_pairs = filter_matched_days(trade_files_all, LOB_DIR)

    print(f"Trade CSV files found: {len(trade_files_all)}")
    print(f"Matched trade+LOB days: {len(matched_pairs)}")

    needed = TRAIN_DAYS + TEST_DAYS
    if len(matched_pairs) < needed:
        raise ValueError(
            f"Not enough matched trade+LOB days. Need {needed}, found {len(matched_pairs)}.\n"
            f"Check that TRADES_DIR and LOB_DIR cover the same dates."
        )

    train_pairs = matched_pairs[:TRAIN_DAYS]
    test_pairs = matched_pairs[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS]

    # build volume curve from train days using median to smooth outliers
    from curve import Bin_Weight
    all_train_weights = [Bin_Weight(str(trade_csv)) for (trade_csv, _) in train_pairs]
    avg_curve_weights = pd.DataFrame(all_train_weights).median(axis=0).to_numpy()
    avg_curve_weights = (avg_curve_weights / avg_curve_weights.sum()).tolist()

    rows = []
    for trade_csv, lob_file in test_pairs:
        day = extract_date(trade_csv.stem)
        trades_df = pd.read_csv(trade_csv).sort_values("timestamp").reset_index(drop=True)

        start_ts_ms = choose_start_ts(trades_df)

        if USE_FIXED_QTY:
            Q_day = float(FIXED_QTY)
        else:
            # convert notional to qty using arrival price
            px0 = float(trades_df["price"].iloc[0])
            Q_day = float(NOTIONAL_USDT / px0)

        sim = simulate_vwap_execution_window(
            trade_csv=trade_csv,
            book_jsonl=lob_file,
            avg_curve_weights=avg_curve_weights,
            Q=Q_day,
            side=SIDE,
            start_ts_ms=start_ts_ms,
            horizon_ms=HORIZON_MS,
            participation_rate=PARTICIPATION_RATE,
            intra_bucket_slices=INTRA_BUCKET_SLICES,
            force_terminal_completion=FORCE_TERMINAL_COMPLETION,
        )

        m = compute_execution_metrics(
            fills=sim["fills"],
            trades=trades_df,
            side=sim["side"],
            Q=sim["Q"],
            day_start=sim["day_start"],
            bucket_ms=BUCKET_MS,
            buckets_per_day=BUCKETS_PER_DAY,
            arrival_price=sim["arrival_price"],
        )

        arrival_price = float(sim["arrival_price"])
        market_vwap = float(m["market_vwap"])

        # express IS and slippage in basis points
        is_bps = (
            10_000 * float(sim["implementation_shortfall"]) / arrival_price
            if arrival_price != 0
            else 0.0
        )

        slippage_bps = (
            10_000 * float(m["slippage_vs_vwap"]) / market_vwap
            if market_vwap != 0
            else 0.0
        )

        row = {
            "day": day,
            "symbol": SYMBOL,
            "side": sim["side"],
            "Q_qty": sim["Q"],
            "filled_qty": sim["filled_qty"],
            "completion_rate": sim["completion_rate"],
            "arrival_price": arrival_price,
            "exec_vwap": float(m["exec_vwap"]),
            "market_vwap": market_vwap,
            "implementation_shortfall": float(sim["implementation_shortfall"]),
            "implementation_shortfall_bps": float(is_bps),
            "slippage_vs_vwap": float(m["slippage_vs_vwap"]),
            "slippage_bps": float(slippage_bps),
            "participation_overall": float(m["participation_overall"]),
            "participation_rate_cap": sim["participation_rate"],
            "intra_bucket_slices": sim["intra_bucket_slices"],
            "start_ts_ms": sim["start_ts_ms"],
            "end_ts_ms": sim["end_ts_ms"],
            "window_horizon_ms": sim["window_horizon_ms"],
            "start_bucket": sim["start_bucket"],
            "end_bucket": sim["end_bucket"],
            "force_terminal_completion": sim["force_terminal_completion"],
            "terminal_liq_qty": sim["terminal_liq_qty"],
            "terminal_liq_avg_px": sim["terminal_liq_avg_px"],
            "lob_file": lob_file.name,
            "trade_file": trade_csv.name,
        }
        rows.append(row)

        if SAVE_FILLS:
            from execution import save_fills_to_csv
            FILLS_DIR.mkdir(parents=True, exist_ok=True)
            out_fills = FILLS_DIR / f"fills_{SYMBOL}_{day}_{SIDE}_Q{Q_day}.csv"
            save_fills_to_csv(sim["fills"], out_fills)

        print(
            f"[DONE] {SYMBOL} {day}  "
            f"IS={row['implementation_shortfall_bps']:.2f} bps  "
            f"slip={row['slippage_bps']:.2f} bps  "
            f"completion={row['completion_rate']:.2%}  "
            f"terminal_liq={row['terminal_liq_qty']:.6f}"
        )

    results = pd.DataFrame(rows)

    print("\n=== Daily Results (Implementation Shortfall bps) ===")
    print(results[["day", "implementation_shortfall_bps", "completion_rate"]].to_string(index=False))

    print("\n=== Summary (Implementation Shortfall bps) ===")
    print(f"Mean IS (bps): {results['implementation_shortfall_bps'].mean():.4f}")
    print(f"Std  IS (bps): {results['implementation_shortfall_bps'].std():.4f}")
    print(f"Min  IS (bps): {results['implementation_shortfall_bps'].min():.4f}")
    print(f"Max  IS (bps): {results['implementation_shortfall_bps'].max():.4f}")

    results.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved daily results to: {OUT_RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()