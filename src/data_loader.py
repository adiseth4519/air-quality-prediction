import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.drop("No", axis=1)

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime")
    df = df.drop(["year", "month", "day", "hour"], axis=1)

    df["pm2.5"] = df["pm2.5"].ffill().bfill()
    df = df.dropna()

    return df
