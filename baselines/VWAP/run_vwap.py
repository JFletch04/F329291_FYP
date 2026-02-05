from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from execution import simulate_vwap_execution_day, BUCKET_MS, BUCKETS_PER_DAY
from metrics import compute_execution_metrics


# ============================================================
# USER CONFIG (change these)
# ============================================================

SYMBOL = "BTCUSDT"   # <-- switch between "BTCUSDT" and "DOGEUSDT"

MONTH = "December"   # folder name you used, e.g. "December"

BASE_DIR = Path("/Users/jackfletcher/Desktop/FYP_Data")

TRADES_DIR = BASE_DIR / f"{SYMBOL}_trades" / MONTH
LOB_DIR    = BASE_DIR / f"{SYMBOL}_LOB" / MONTH

# Train/Test split (uses matched trade+LOB days)
TRAIN_DAYS = 25
TEST_DAYS = 6

SIDE = "sell"  # "buy" or "sell"

# Parent order size (recommended: define as USDT notional)
NOTIONAL_USDT = 10_000_000 # e.g. 1_000_000 means ~$1m notional per day

# Execution knobs
PARTICIPATION_RATE = 0.10     # 0.05 = 5%, 0.10 = 10%, 1.0 = 100% (very aggressive)
INTRA_BUCKET_SLICES = 5       # how many child orders inside each 5-min bucket

OUT_RESULTS_CSV = Path(f"vwap_results_{SYMBOL}_{MONTH}.csv")
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
    # LOB filenames: 2025-12-20_BTCUSDT_ob200.data
    return f"{day}_{SYMBOL}_ob200.data"

def match_lob_file_for_trade(trade_file: Path, lob_dir: Path) -> Optional[Path]:
    """
    Trades: {SYMBOL}_YYYY-MM-DD.csv
    LOB:    YYYY-MM-DD_{SYMBOL}_ob200.data
    """
    day = extract_date(trade_file.stem)

    exact = lob_dir / expected_lob_name(day)
    if exact.exists():
        return exact

    # Fuzzy fallback
    candidates = []
    for p in lob_dir.iterdir():
        name = p.name
        if day in name and SYMBOL in name and "ob200" in name:
            candidates.append(p)

    if not candidates:
        return None

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


# ============================================================
# Main
# ============================================================

def main():
    print(f"SYMBOL={SYMBOL}  MONTH={MONTH}")
    print(f"TRADES_DIR={TRADES_DIR}")
    print(f"LOB_DIR={LOB_DIR}")

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

    # ---- Build curve from TRAIN trades ----
    from curve import Bin_Weight  # must exist in curve.py

    all_train_weights = [Bin_Weight(str(trade_csv)) for (trade_csv, _) in train_pairs]
    avg_curve_weights = pd.DataFrame(all_train_weights).median(axis=0).to_numpy()
    avg_curve_weights = (avg_curve_weights / avg_curve_weights.sum()).tolist()

    # ---- Run VWAP on TEST days ----
    rows = []
    for trade_csv, lob_file in test_pairs:
        day = extract_date(trade_csv.stem)

        # Load trades once (used for Q conversion + metrics)
        trades_df = pd.read_csv(trade_csv)

        # Convert notional to quantity in base asset units
        px0 = float(trades_df["price"].iloc[0])
        Q_day = float(NOTIONAL_USDT / px0)

        sim = simulate_vwap_execution_day(
            trade_csv=trade_csv,
            book_jsonl=lob_file,
            avg_curve_weights=avg_curve_weights,
            Q=Q_day,
            side=SIDE,
            participation_rate=PARTICIPATION_RATE,
            intra_bucket_slices=INTRA_BUCKET_SLICES,
        )

        m = compute_execution_metrics(
            fills=sim["fills"],
            trades=trades_df,
            side=sim["side"],
            Q=sim["Q"],
            day_start=sim["day_start"],
            bucket_ms=BUCKET_MS,
            buckets_per_day=BUCKETS_PER_DAY,
            arrival_price=None,
        )

        # Also compute bps slippage for easy comparison across assets
        slippage_bps = 10_000 * float(m["slippage_vs_vwap"]) / float(m["market_vwap"]) if float(m["market_vwap"]) != 0 else 0.0

        row = {
            "day": day,
            "symbol": SYMBOL,
            "side": sim["side"],
            "notional_usdt": NOTIONAL_USDT,
            "Q_qty": sim["Q"],
            "filled_qty": sim["filled_qty"],
            "completion_rate": sim["completion_rate"],
            "exec_vwap": float(m["exec_vwap"]),
            "market_vwap": float(m["market_vwap"]),
            "slippage_vs_vwap": float(m["slippage_vs_vwap"]),  # in price units (USDT per coin)
            "slippage_bps": float(slippage_bps),
            "implementation_shortfall": float(m["implementation_shortfall"]),
            "participation_overall": float(m["participation_overall"]),
            "participation_rate_cap": sim["participation_rate"],
            "intra_bucket_slices": sim["intra_bucket_slices"],
            "lob_file": lob_file.name,
            "trade_file": trade_csv.name,
        }
        rows.append(row)

        if SAVE_FILLS:
            from execution import save_fills_to_csv
            FILLS_DIR.mkdir(parents=True, exist_ok=True)
            out_fills = FILLS_DIR / f"fills_{SYMBOL}_{day}_{SIDE}_notional{NOTIONAL_USDT}.csv"
            save_fills_to_csv(sim["fills"], out_fills)

        print(
            f"[DONE] {SYMBOL} {day}  "
            f"slip={row['slippage_vs_vwap']:.6f}  ({row['slippage_bps']:.2f} bps)  "
            f"completion={row['completion_rate']:.2%}"
        )

    results = pd.DataFrame(rows)

    print("\n=== Summary (slippage_bps) ===")
    print(results[["day", "slippage_bps", "completion_rate"]].to_string(index=False))
    print("\n=== Describe ===")
    print(results.describe(include="all"))

    results.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved daily results to: {OUT_RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()

