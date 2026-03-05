import pandas as pd
import matplotlib.pyplot as plt

"""
This plot shows the validation mean implementation shortfall (IS, in bps)
across PPO-LSTM training iterations.

Each point corresponds to one full PPO training iteration
(collect rollouts -> update policy -> evaluate on validation set).

The curve illustrates:
- How execution quality evolves during training
- Whether the policy is improving (lower IS is better)
- The point at which validation performance stabilises or degrades
  (used for early stopping to prevent overfitting)

Downward movement indicates improved execution performance.
Plateauing or rising IS indicates no further generalisation gains.
"""



CSV_PATH = "logs/train_ppo_lstm.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # Clean + enforce numeric
    for col in ["iter", "val_mean_is_bps", "entropy", "approx_kl", "loss_total", "loss_pi", "loss_v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by iteration so the line doesn't scribble
    df = df.sort_values("iter").dropna(subset=["iter", "val_mean_is_bps"])

    # ---- Learning curve: Validation IS (bps) ----
    plt.figure()
    plt.plot(df["iter"], df["val_mean_is_bps"], marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Validation mean IS (bps)")
    plt.title("PPO-LSTM validation IS across training")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()

