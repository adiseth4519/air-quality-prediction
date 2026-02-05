import numpy as np
import pandas as pd

def add_features(df):
    df = pd.get_dummies(df, columns=["cbwd"], drop_first=True)

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm2.5_lag_{lag}"] = df["pm2.5"].shift(lag)

    return df.dropna()
