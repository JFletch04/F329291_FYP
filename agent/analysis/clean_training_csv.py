import pandas as pd

LOG_PATH = "logs/train_ppo_lstm.csv"
OUT_PATH = "logs/train_ppo_lstm_clean.csv"

def main():
    df = pd.read_csv(LOG_PATH, engine="python", on_bad_lines="skip")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned log to: {OUT_PATH}")
    print("Rows kept:", len(df))

if __name__ == "__main__":
    main()
