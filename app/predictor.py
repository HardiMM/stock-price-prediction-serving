import joblib
import pandas as pd

from pathlib import Path


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


class StockPredictor:

    def __init__(self):
        print("Loading model...")
        self.model = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")

    def predict(self, data: dict):

        # Convert input dictionary to DataFrame
        input_data = pd.DataFrame([data])

        # Make sure feature order is correct
        input_data = input_data[FEATURES]

        # Predict next-day return
        predicted_return = self.model.predict(input_data)[0]

        # Convert return to predicted price
        current_price = data["Close"]

        predicted_price = current_price * (1 + predicted_return)

        return {
            "predicted_return": float(predicted_return),
            "predicted_price": float(predicted_price)
        }