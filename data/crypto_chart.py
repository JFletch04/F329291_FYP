import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Load your replay dataset
df = pd.read_parquet("/Users/jackfletcher/Desktop/FYP_Data/Replay_5s_BTC/December/2025-12-01_steps_5s.parquet")

# Convert timestamp column to datetime
# Change "ts" if your timestamp column has a different name
df["datetime"] = pd.to_datetime(df["ts"], unit="ms")

# If mid price is not already stored, compute it
# Change these column names if needed
if "mid" in df.columns:
    df["mid_price"] = df["mid"]
else:
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2

# Create figure
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot mid price
ax1.plot(df["datetime"], df["mid_price"], linewidth=0.4)
ax1.set_title("BTC/USDT (01/12/2025)", fontname="Times New Roman")
ax1.set_xlabel("Time", fontname="Times New Roman")
ax1.set_ylabel("Price", fontname="Times New Roman")
ax1.grid(True, linestyle="--", linewidth=0.75, alpha=0.3)

# Clean up borders for a more professional look
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Format x-axis to show cleaner time labels
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))


# Improve layout
plt.tight_layout()

# Show plot
plt.show()


