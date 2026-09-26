from pydantic import BaseModel, Field
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import os
import joblib
import numpy as np
from preprocessing import transform_text, remove_stopwords

MODEL_PATH = os.getenv('MODEL_BUNDLE_PATH', '../model/bundled_pipeline.joblib')


try:
    
    bundle = joblib.load(MODEL_PATH)
    naive_model = bundle['nb_pipeline']
    rfc_model = bundle['rf_pipeline']
    classes = bundle['target_names']
    MODEL_LOAD_ERROR = None

except Exception as e:
    naive_model =rfc_model = None
    classes = []
    MODEL_LOAD_ERROR = str(e)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class SMSRequest(BaseModel):
    sms_message: str = Field(description='Raw SMS Text Message', 
                              min_length=1, max_length=500)
    
class PredictionResponse(BaseModel):
    label: str
    confidence: float

def _preprocess(raw_message: str) -> str:
    return remove_stopwords(transform_text(raw_message))

def _predict(pipeline, raw_text: str) -> PredictionResponse:
    if pipeline is None:
         raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

    try:
        clean_text = _preprocess(raw_text)
        probability = pipeline.predict_proba([clean_text])[0]
        prediction = int(probability.argmax())

        return PredictionResponse(
            label=classes[prediction],
            confidence = float(probability[prediction])
        )

    except Exception as e:
        raise ValueError(f"Unable to predict the text: {e}")

@app.get("/")
def home() -> Dict[str, str]:
    return {
        "message" : "SMS Scam Classifier",
        "status" : "Running" if MODEL_LOAD_ERROR is None else "Model Loading failed.",
        "endpoint" : "Send POST request to /nbpredict for NB and /rfpredict for RF"
    }

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status" : "Running",
        "model" : ["MultinomialNB", "RandomForestClassifier"]
    }

@app.post("/nbpredict", response_model=PredictionResponse)
def nb_predict(request: SMSRequest) -> Dict[str, str]:

    try:
        return _predict(naive_model, request.sms_message)
    except HTTPException:
        raise

@app.post("/rfpredict", response_model=PredictionResponse)
def rf_predict(request: SMSRequest) -> Dict[str, str]:

    try:
        return _predict(rfc_model, request.sms_message)
    except HTTPException:
        raise
