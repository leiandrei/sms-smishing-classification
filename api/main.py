import os
from pathlib import Path
from typing import Any, Dict
import joblib
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.preprocessing import extract_inference_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "bundled_pipeline.joblib"
MODEL_PATH = os.getenv("MODEL_BUNDLE_PATH", str(DEFAULT_MODEL_PATH))

MNB_PARAM_KEYS = ("alpha", "fit_prior")
CNB_PARAM_KEYS = ("alpha", "norm")
RF_PARAM_KEYS = (
    "class_weight",
    "criterion",
    "max_depth",
    "max_features",
    "min_samples_split",
    "min_samples_leaf",
    "n_estimators",
    "random_state",
    "n_jobs",
)

try:
    bundle = joblib.load(MODEL_PATH)
    preprocessor = bundle["preprocessor"]
    models = bundle["models"]
    mnb_model = models["mnb"]
    cnb_model = models["cnb"]
    rf_model = models["rf"]

    mnb_params = {k: mnb_model.get_params()[k] for k in MNB_PARAM_KEYS}
    cnb_params = {k: cnb_model.get_params()[k] for k in CNB_PARAM_KEYS}
    rf_params = {k: rf_model.get_params()[k] for k in RF_PARAM_KEYS}

    rf_model.n_jobs = 1
    classes = bundle["target_names"]
    MODEL_LOAD_ERROR = None

    saved_version = bundle.get("metadata", {}).get("sklearn_version")
    if saved_version and saved_version != sklearn.__version__:
        MODEL_LOAD_ERROR = (
            f"scikit-learn version mismatch: bundle={saved_version}, runtime={sklearn.__version__}"
        )
except Exception as e:
    preprocessor = mnb_model = cnb_model = rf_model = None
    mnb_params = cnb_params = rf_params = {}
    classes = []
    MODEL_LOAD_ERROR = str(e)

app = FastAPI(title="SMS Scam Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SMSRequest(BaseModel):
    sms_message: str = Field(
        description="Raw SMS Text Message",
        min_length=1,
        max_length=500,
    )


class PredictionResponse(BaseModel):
    model: str
    label: str
    confidence: float
    probabilities: Dict[str, float]


def _ensure_loaded() -> None:
    if MODEL_LOAD_ERROR is not None or preprocessor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")


def _predict(model, model_name: str, raw_text: str) -> PredictionResponse:
    _ensure_loaded()
    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_name} is not available.")

    try:
        features_df = extract_inference_features(raw_text)
        X_transformed = preprocessor.transform(features_df)
        probs = model.predict_proba(X_transformed)[0]
        pred_idx = int(probs.argmax())

        return PredictionResponse(
            model=model_name,
            label=classes[pred_idx],
            confidence=round(float(probs[pred_idx]), 6),
            probabilities={
                cls_name: round(float(prob), 6)
                for cls_name, prob in zip(classes, probs)
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to predict the text: {e}")


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "message": "SMS Scam Classifier",
        "status": "Running" if MODEL_LOAD_ERROR is None else "Model loading failed",
        "endpoints": {
            "multinomial_nb": {"predict": "/mnb/predict", "parameters": "/mnb/parameters"},
            "complement_nb": {"predict": "/cnb/predict", "parameters": "/cnb/parameters"},
            "random_forest": {"predict": "/rf/predict", "parameters": "/rf/parameters"},
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    _ensure_loaded()
    return {
        "status": "Running",
        "models": ["MultinomialNB", "ComplementNB", "RandomForestClassifier"],
    }


@app.get("/mnb/parameters")
def mnb_parameters() -> Dict[str, Any]:
    _ensure_loaded()
    return mnb_params


@app.get("/cnb/parameters")
def cnb_parameters() -> Dict[str, Any]:
    _ensure_loaded()
    return cnb_params


@app.get("/rf/parameters")
def rf_parameters() -> Dict[str, Any]:
    _ensure_loaded()
    return rf_params


@app.post("/mnb/predict", response_model=PredictionResponse)
def mnb_predict(request: SMSRequest) -> PredictionResponse:
    return _predict(mnb_model, "MultinomialNB", request.sms_message)


@app.post("/cnb/predict", response_model=PredictionResponse)
def cnb_predict(request: SMSRequest) -> PredictionResponse:
    return _predict(cnb_model, "ComplementNB", request.sms_message)


@app.post("/rf/predict", response_model=PredictionResponse)
def rf_predict(request: SMSRequest) -> PredictionResponse:
    return _predict(rf_model, "RandomForestClassifier", request.sms_message)