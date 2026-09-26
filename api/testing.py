from fastapi.testclient import TestClient
from main import app
import pytest 

ENDPOINTS = ["/nbpredict", "/rfpredict"]
client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_predict_validation(endpoint):
    response = client.post(endpoint, json={"sms_message" : "Congratulations! nanalo ka ng $1,000,000.00, claim it via GCash"})
    assert response.status_code == 200
    data = response.json()

    assert data['label'] in ['Legitimate', 'Smishing', 'Spam']
    assert 0.0 <= data['confidence'] <= 1.0

@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_predict_empty_text_rejected(endpoint):
    response = client.post(endpoint, json={"sms_message" : ""})

    assert response.status_code == 422

@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_predict_missing_field_rejected(endpoint):
    response = client.post(endpoint, json={})

    assert response.status_code == 422

@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_predict_oversized_text_rejected(endpoint):
    response = client.post(endpoint, json={"sms_message" : "a" * 600})

    assert response.status_code == 422

@pytest.mark.parametrize(
    "endpoint,text,expected_label",
    [
        ("/nbpredict", "verify your gcash account now or it will be suspended kycupdategcash.online", "Smishing"),
        ("/rfpredict", "verify your gcash account now or it will be suspended kycupdategcash.online", "Smishing"),
        ("/nbpredict", "win big at 789bingo.com free spin now!", "Spam"),
        ("/rfpredict", "win big at 789bingo.com free spin now!", "Spam"),
    ],
)
def test_predict(endpoint, text, expected_label):
    response = client.post(endpoint, json={"sms_message": text})
    assert response.status_code == 200
    data = response.json()
    assert data['label'] == expected_label