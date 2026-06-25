from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI(title="Fraud Detection API 🚀")

# =========================
# PATHS (FIXED)
# =========================
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
COLUMNS_PATH = "models/train_columns.pkl"

# =========================
# LOAD ARTIFACTS
# =========================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    train_columns = pickle.load(f)


# =========================
# INPUT SCHEMA
# =========================
class Transaction(BaseModel):
    Amount: float
    MerchantCategory: str
    TransactionType: str
    Latitude: float
    Longitude: float
    AvgTransactionAmount: float
    TransactionFrequency: float
    UnusualLocation: int
    UnusualAmount: int
    NewDevice: int
    FailedAttempts: int
    Hour: int
    DayOfWeek: int
    User_Transaction_Count: int
    IP_Frequency: int
    Phone_Usage_Count: int


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(data: dict):
    df = pd.DataFrame([data])

    # encode categorical safely
    for col in ["MerchantCategory", "TransactionType"]:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # align columns
    df = df.reindex(columns=train_columns, fill_value=0)

    return df


# =========================
# RISK ENGINE (IMPORTANT FIX)
# =========================
def compute_risk(prob, data):
    risk = float(prob)

    if data["Amount"] > data["AvgTransactionAmount"] * 3:
        risk += 0.2

    if data["FailedAttempts"] > 2:
        risk += 0.15

    if data["NewDevice"] == 1:
        risk += 0.10

    if data["UnusualLocation"] == 1:
        risk += 0.10

    if data["Hour"] < 5:
        risk += 0.05

    return min(risk, 1.0)


def risk_level(score):
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running 🚀"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(txn: Transaction):

    data = txn.dict()

    X = preprocess(data)

    prob = model.predict_proba(X)[0][1]

    # 🔥 FIXED RISK LOGIC (important)
    final_risk = compute_risk(prob, data)
    pred = final_risk > 0.5

    # =========================
    # EXPLANATION ENGINE
    # =========================
    reasons = []

    if data["Amount"] > data["AvgTransactionAmount"] * 2:
        reasons.append("High transaction amount anomaly")

    if data["FailedAttempts"] > 1:
        reasons.append("Multiple failed login/transaction attempts")

    if data["NewDevice"] == 1:
        reasons.append("Transaction from new device")

    if data["UnusualLocation"] == 1:
        reasons.append("Unusual location detected")

    if data["TransactionFrequency"] > 15:
        reasons.append("Abnormally high transaction frequency")

    # =========================
    # RESPONSE
    # =========================
    return {
        "fraud_probability": round(float(final_risk), 4),
        "raw_model_probability": round(float(prob), 4),
        "is_fraud": bool(pred),
        "risk_level": risk_level(final_risk),
        "reasons": reasons
    }
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")