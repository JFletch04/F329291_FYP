import pandas as pd
import numpy as np

# Change this to one DOGE parquet file
path = "/Users/jackfletcher/Desktop/FYP_Data/replay_5s_BTC/November/2025-11-01_steps_5s.parquet"

df = pd.read_parquet(path)

print("Rows:", len(df))
print()

print("Trade volume stats (BTC units):")
print("Mean trade_vol:     ", df["trade_vol"].mean())
print("Median trade_vol:   ", df["trade_vol"].median())
print("90th pct trade_vol: ", df["trade_vol"].quantile(0.90))
print("99th pct trade_vol: ", df["trade_vol"].quantile(0.99))

print()
print("Non-zero trade_vol rows:", (df["trade_vol"] > 0).sum())