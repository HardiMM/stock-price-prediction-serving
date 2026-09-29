import yfinance as yf
from pathlib import Path


TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = None


def download_stock_data():
    print(f"Downloading data for {TICKER}...")

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("No stock data was downloaded.")

    # Handle yfinance's possible MultiIndex columns
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    output_path = Path("data") / f"{TICKER}_historical.csv"
    data.to_csv(output_path, index=False)

    print(f"Downloaded {len(data)} rows.")
    print(f"Saved data to: {output_path}")


if __name__ == "__main__":
    download_stock_data()