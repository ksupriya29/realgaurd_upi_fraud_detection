from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load model artifacts
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

FEATURES = [
    "Amount", "MerchantCategory", "TransactionType",
    "Latitude", "Longitude",
    "AvgTransactionAmount", "TransactionFrequency",
    "UnusualLocation", "UnusualAmount",
    "NewDevice", "FailedAttempts",
    "Hour", "DayOfWeek",
    "User_Transaction_Count",
    "IP_Frequency",
    "Phone_Usage_Count"
]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # encode categorical features
        for col, le in encoders.items():
            if col in df:
                df[col] = le.transform(df[col].astype(str))

        df = df[FEATURES]

        X = scaler.transform(df)

        prob = model.predict_proba(X)[0][1]
        pred = prob > 0.5

        return {
            "fraud_probability": float(prob),
            "is_fraud": bool(pred)
        }

    except Exception as e:
        return {"error": str(e)}