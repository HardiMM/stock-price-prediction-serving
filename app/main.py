from fastapi import FastAPI
from pydantic import BaseModel

from app.predictor import StockPredictor


app = FastAPI(
    title="Stock Price Prediction API",
    description="API for next-day stock price prediction",
    version="1.0.0"
)


# Load the model once when the application starts
predictor = StockPredictor()


class StockInput(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Previous_Close: float
    MA_7: float
    MA_14: float
    MA_30: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(stock: StockInput):

    result = predictor.predict(stock.model_dump())

    return {
        "predicted_return": result["predicted_return"],
        "predicted_price": result["predicted_price"]
    }