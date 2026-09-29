import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data") / "AAPL_historical.csv"
OUTPUT_FILE = Path("data") / "AAPL_processed.csv"


def create_features():
    print("Loading stock data...")

    df = pd.read_csv(INPUT_FILE)

    # Make sure the data is sorted chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Previous day's closing price
    df["Previous_Close"] = df["Close"].shift(1)

    # Moving averages
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_14"] = df["Close"].rolling(window=14).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()

    # Target: next trading day's closing price
    df["Target"] = df["Close"].shift(-1)

    # Remove rows created by rolling/shift operations
    df = df.dropna().reset_index(drop=True)

    # Select the columns we need
    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Previous_Close",
        "MA_7",
        "MA_14",
        "MA_30",
        "Target"
    ]

    df = df[columns]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed data saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    create_features()