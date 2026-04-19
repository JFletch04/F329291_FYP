import pandas as pd

# Load your parquet file
df = pd.read_parquet("./data/Replay_5s_BTC/December/2025-12-01_steps_5s.parquet")

# Convert timestamp
df["datetime"] = pd.to_datetime(df["ts"], unit="ms")

# Function to expand price/size arrays
def expand_levels(df, col_prices, col_sizes, side, levels=5):
    for i in range(levels):
        df[f"{side}_price_{i+1}"] = df[col_prices].apply(lambda x: x[i] if len(x) > i else None)
        df[f"{side}_size_{i+1}"] = df[col_sizes].apply(lambda x: x[i] if len(x) > i else None)
    return df

# Expand bids and asks
df = expand_levels(df, "bid_prices", "bid_sizes", "bid", levels=5)
df = expand_levels(df, "ask_prices", "ask_sizes", "ask", levels=5)

# Keep only clean report columns
clean = df[
    ["datetime", "mid", "spread"]
    + [f"bid_price_{i}" for i in range(1, 6)]
    + [f"bid_size_{i}" for i in range(1, 6)]
    + [f"ask_price_{i}" for i in range(1, 6)]
    + [f"ask_size_{i}" for i in range(1, 6)]
]

# Export to Excel
clean.to_excel("lob_report.xlsx", index=False)

print("Saved lob_report.xlsx")


pd.read_json("file_path", lines=True)