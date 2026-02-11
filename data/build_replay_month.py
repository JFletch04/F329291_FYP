import os
from pathlib import Path
from datetime import datetime

from build_replay_day import build_replay_day  # adjust import if needed


LOB_DIR = Path("/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_LOB/November")
TRADES_DIR = Path("/Users/jackfletcher/Desktop/FYP_Data/BTCUSDT_trades/November")
OUT_DIR = Path("/Users/jackfletcher/Desktop/FYP_Data/replay_5s")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_date_from_lob(fname: str) -> str:
    # "2026-01-01_BTCUSDT_ob200.data" → "2026-01-01"
    return fname.split("_")[0]


def extract_date_from_trades(fname: str) -> str:
    # "BTCUSDT_2026-01-01.csv" → "2026-01-01"
    return fname.split("_")[1].replace(".csv", "")


def main():
    # Build lookup for trades files by date
    trades_by_date = {}
    for f in TRADES_DIR.iterdir():
        if f.suffix == ".csv":
            date = extract_date_from_trades(f.name)
            trades_by_date[date] = f

    # Loop through LOB files
    for lob_file in sorted(LOB_DIR.iterdir()):
        if not lob_file.name.endswith(".data"):
            continue

        date = extract_date_from_lob(lob_file.name)

        if date not in trades_by_date:
            print(f"[SKIP] No trades file for {date}")
            continue

        out_path = OUT_DIR / f"{date}_steps_5s.parquet"
        if out_path.exists():
            print(f"[SKIP] Already built: {out_path.name}")
            continue

        print(f"[BUILD] {date}")

        build_replay_day(
            lob_jsonl_path=str(lob_file),
            trades_csv_path=str(trades_by_date[date]),
            out_parquet_path=str(out_path),
            top_n_levels=10,
            grid_ms=5_000,
            use_ts_field="ts",
        )

        print(f"[DONE] {date}")


if __name__ == "__main__":
    main()
