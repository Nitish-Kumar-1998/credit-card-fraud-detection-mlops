from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

# Set tracking URI first
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from MLflow registry
model = mlflow.pyfunc.load_model("models:/fraud-detection-model/1")

app = FastAPI(title="Fraud Detection API")

# Define input schema
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

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    probability = model.predict(data)[0]
    return {
        "fraud_probability": round(float(probability), 4),
        "is_fraud": bool(probability > 0.5)
    }