import pytest
from fastapi.testclient import TestClient
import api.main as main_module
from api.main import app
from src.preprocessing import extract_inference_features

client = TestClient(app)

VALID_LABELS = {"Legitimate (ham)", "Smishing (scam)"}

PREDICT_ENDPOINTS = [
    ("/mnb/predict", "MultinomialNB"),
    ("/cnb/predict", "ComplementNB"),
    ("/rf/predict", "RandomForestClassifier"),
]

PARAMETER_ENDPOINTS = [
    ("/mnb/parameters", {"alpha": 0.1, "fit_prior": True}),
    ("/cnb/parameters", {"alpha": 0.1, "norm": False}),
    (
        "/rf/parameters",
        {
            "class_weight": None,
            "criterion": "entropy",
            "max_depth": 200,
            "max_features": "log2",
            "min_samples_split": 6,
            "min_samples_leaf": 1,
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
]


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Running"
    assert data["endpoints"]["multinomial_nb"]["predict"] == "/mnb/predict"
    assert data["endpoints"]["complement_nb"]["predict"] == "/cnb/predict"
    assert data["endpoints"]["random_forest"]["predict"] == "/rf/predict"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Running"
    assert set(data["models"]) == {
        "MultinomialNB",
        "ComplementNB",
        "RandomForestClassifier",
    }


@pytest.mark.parametrize("endpoint,expected_params", PARAMETER_ENDPOINTS)
def test_model_parameters(endpoint, expected_params):
    response = client.get(endpoint)
    assert response.status_code == 200
    assert response.json() == expected_params


def test_extract_inference_features_schema():
    sample = "URGENT! Claim your P5,000 GCash reward now at bit.ly/claim123 or call 09171234567!"
    df = extract_inference_features(sample)
    expected_columns = [
        "text",
        "has_url",
        "has_money_mention",
        "has_phone_in_text",
        "exclamation_count",
        "uppercase_ratio",
        "has_otp",
        "urgency",
        "has_leetspeaks",
        "typo_squatting",
        "has_url_shortened",
        "punctuation_count",
        "word_count",
        "text_count",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 1
    assert df.loc[0, "has_url"] == 1
    assert df.loc[0, "has_money_mention"] == 1
    assert df.loc[0, "has_phone_in_text"] == 1
    assert df.loc[0, "has_url_shortened"] == 1
    assert df.loc[0, "urgency"] == 1
    assert df.loc[0, "exclamation_count"] == 2


@pytest.mark.parametrize("endpoint,expected_model", PREDICT_ENDPOINTS)
def test_predict_response_contract(endpoint, expected_model):
    payload = {
        "sms_message": "Congratulations! Nanalo ka ng P1,000,000.00, claim it via GCash at http://claim-prize.online"
    }
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model"] == expected_model
    assert data["label"] in VALID_LABELS
    assert 0.0 <= data["confidence"] <= 1.0

    probs = data["probabilities"]
    assert set(probs.keys()) == VALID_LABELS
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert data["confidence"] == pytest.approx(probs[data["label"]], abs=1e-6)


@pytest.mark.parametrize("endpoint,_", PREDICT_ENDPOINTS)
@pytest.mark.parametrize("length", [1, 500])
def test_predict_valid_length_boundaries(endpoint, _, length):
    response = client.post(endpoint, json={"sms_message": "a" * length})
    assert response.status_code == 200
    assert response.json()["label"] in VALID_LABELS


@pytest.mark.parametrize("endpoint,_", PREDICT_ENDPOINTS)
def test_predict_stopwords_only_handled(endpoint, _):
    response = client.post(endpoint, json={"sms_message": "ang mga ng sa po opo"})
    assert response.status_code == 200
    assert response.json()["label"] in VALID_LABELS


@pytest.mark.parametrize("endpoint,_", PREDICT_ENDPOINTS)
@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"sms_message": ""},
        {},
        {"sms_message": "a" * 501},
        {"sms_message": "a" * 600},
        {"sms_message": None},
    ],
)
def test_predict_invalid_inputs_rejected(endpoint, _, invalid_payload):
    response = client.post(endpoint, json=invalid_payload)
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint,_", PREDICT_ENDPOINTS)
@pytest.mark.parametrize(
    "text,expected_label",
    [
        (
            "Verify your GCash account now or it will be suspended: http://kycupdategcash.online",
            "Smishing (scam)",
        ),
        (
            "URGENT: Your BDO account is disabled. Update OTP via bit.ly/bdo-verify now!",
            "Smishing (scam)",
        ),
        (
            "Nandito na ako sa bahay, ingat sa pag-uwi at kain na tayo pagdating mo.",
            "Legitimate (ham)",
        ),
    ],
)
def test_predict_domain_cases(endpoint, _, text, expected_label):
    response = client.post(endpoint, json={"sms_message": text})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == expected_label
    assert data["confidence"] > 0.5


def test_service_unavailable_when_bundle_fails(monkeypatch):
    monkeypatch.setattr(main_module, "MODEL_LOAD_ERROR", "Simulated load failure")

    health_resp = client.get("/health")
    assert health_resp.status_code == 503

    param_resp = client.get("/mnb/parameters")
    assert param_resp.status_code == 503

    pred_resp = client.post("/mnb/predict", json={"sms_message": "Hello"})
    assert pred_resp.status_code == 503