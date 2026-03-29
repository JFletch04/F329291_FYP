import pandas as pd
import matplotlib.pyplot as plt

btc = pd.read_csv("logs/BTC/hparam_sweep2_summary.csv")
doge = pd.read_csv("logs/DOGE/hparam_sweep_stage2_gamma_lambda.csv")

plt.figure(figsize=(7,5))
plt.boxplot([btc["best_val_is_bps"], doge["best_val_is_bps"]])
plt.xticks([1,2], ["BTC","DOGE"])
plt.ylabel("Validation Implementation Shortfall (bps)")
plt.title("BTC vs DOGE Hyperparameter Performance")
plt.tight_layout()

plt.savefig("btc_doge_boxplot.png", dpi=300)
plt.show()