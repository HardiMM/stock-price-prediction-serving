from app.predictor import StockPredictor


predictor = StockPredictor()

sample_data = {
    "Open": 250.0,
    "High": 255.0,
    "Low": 248.0,
    "Close": 253.0,
    "Volume": 50000000,
    "Previous_Close": 251.0,
    "MA_7": 250.0,
    "MA_14": 248.0,
    "MA_30": 245.0
}

result = predictor.predict(sample_data)

print("\nPrediction:")
print(result)