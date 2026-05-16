from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os

# Load model from file
model = xgb.Booster()
model.load_model("models/xgboost_model.json")

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount_log: float
    Time_norm: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "fraud-detection-model",
        "version": "1",
        "stage": "Production"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    dmatrix = xgb.DMatrix(data)
    probability = float(model.predict(dmatrix)[0])
    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": bool(probability > 0.5)
    }