import pandas as pd
import matplotlib.pyplot as plt

# Plotting validation implementation shortfall across PPO training iterations

CSV_PATH = "logs/train_ppo_lstm.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # Ensure numeric columns
    for col in ["iter", "val_mean_is_bps", "entropy", "approx_kl", "loss_total", "loss_pi", "loss_v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sorting + removing missing values
    df = df.sort_values("iter").dropna(subset=["iter", "val_mean_is_bps"])

    # Validation IS learning curve
    plt.figure()
    plt.plot(df["iter"], df["val_mean_is_bps"], marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Validation mean IS (bps)")
    plt.title("PPO-LSTM validation IS across training")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()

