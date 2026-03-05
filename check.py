import pandas as pd

df = pd.read_csv("logs/hparam_sweep_summary.csv")

def highest_IS(df):
    df2 = df.sort_values('best_val_is_bps')
    return df2.head(3)

print(highest_IS(df))

