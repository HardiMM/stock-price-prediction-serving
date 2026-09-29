import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = Path("data") / "AAPL_processed.csv"
MODEL_FILE = Path("models") / "stock_model.pkl"


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Previous_Close",
    "MA_7",
    "MA_14",
    "MA_30"
]


def train_model():

    print("Loading processed data...")

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate next-day percentage return
    df["Target_Return"] = (df["Target"] - df["Close"]) / df["Close"]

    # Remove any invalid rows
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURES]
    y = df["Target_Return"]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(X_train, y_train)

    # Predict returns
    predicted_returns = model.predict(X_test)

    # Convert predicted returns back to prices
    current_prices = df["Close"].iloc[split_index:].values
    actual_prices = df["Target"].iloc[split_index:].values

    predicted_prices = current_prices * (1 + predicted_returns)

    # Evaluate price predictions
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = mean_squared_error(actual_prices, predicted_prices) ** 0.5
    r2 = r2_score(actual_prices, predicted_prices)

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save model
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()