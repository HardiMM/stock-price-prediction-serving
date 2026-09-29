from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():

    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_return" in data
    assert "predicted_price" in data

    assert isinstance(data["predicted_return"], float)
    assert isinstance(data["predicted_price"], float)

    assert data["predicted_price"] > 0