"""
RealGuard – UPI Fraud Detection & Risk Analysis
Production FastAPI backend v2.2

Key design decisions:
  - TransactionType and MerchantCategory are validated against the exact
    training-dataset categories (P2M/P2P; 7 merchant categories).
  - predict_proba()[fraud_class_index] is used; fraud class is verified at
    startup from model.classes_.
  - SHAP uses TreeExplainer; values are for the fraud class (class 1).
  - Risk score = model_probability + domain-rule boosts, capped at 1.0.
  - Thresholds: LOW < 0.30, MEDIUM < 0.65, HIGH >= 0.65.
  - is_fraud = final_risk >= 0.50.
"""

import os, uuid, logging, pickle, traceback
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shap

from app.schemas import (
    TransactionInput, HealthResponse,
    VALID_MERCHANT_CATEGORIES, VALID_TRANSACTION_TYPES,
    RISK_LOW, RISK_MEDIUM, FRAUD_THRESHOLD,
)
from app.database import init_db, insert_transaction, fetch_history, clear_history, fetch_analytics

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("realguard")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RealGuard – UPI Fraud Detection",
    description="Explainable AI fraud detection for UPI transactions",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ───────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, "models", "xgb_model.pkl")
ENC_PATH   = os.path.join(BASE, "models", "encoders_v2.pkl")
FEAT_PATH  = os.path.join(BASE, "models", "feature_names.pkl")
METR_PATH  = os.path.join(BASE, "models", "model_metrics.pkl")

# ── Globals ───────────────────────────────────────────────────────────────────
model          = None
encoders       = None
feature_names  = None
model_metrics  = None
shap_explainer = None
FRAUD_CLASS_IDX = 1   # verified at startup from model.classes_


def load_artifacts():
    global model, encoders, feature_names, model_metrics, shap_explainer, FRAUD_CLASS_IDX

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH,   "rb") as f:
        encoders = pickle.load(f)
    with open(FEAT_PATH,  "rb") as f:
        feature_names = pickle.load(f)
    with open(METR_PATH,  "rb") as f:
        model_metrics = pickle.load(f)

    # Verify fraud class index from the trained model — do NOT assume it is 1.
    classes = list(model.classes_)
    if 1 in classes:
        FRAUD_CLASS_IDX = classes.index(1)
    elif True in classes:
        FRAUD_CLASS_IDX = classes.index(True)
    else:
        raise RuntimeError(f"Cannot identify fraud class in model.classes_: {classes}")

    shap_explainer = shap.TreeExplainer(model)

    logger.info(
        "Artifacts loaded — %d features | fraud class index=%d | classes=%s",
        len(feature_names), FRAUD_CLASS_IDX, classes,
    )


@app.on_event("startup")
def startup():
    load_artifacts()
    init_db()
    logger.info("RealGuard v2.2 running")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  — must match training pipeline exactly
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Label-encode categoricals using saved encoders
    for col in ["MerchantCategory", "TransactionType", "TransactionFrequency"]:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: (
                    le.transform([x])[0]
                    if x in le.classes_
                    else le.transform([le.classes_[0]])[0]   # fallback to first class
                )
            )

    # Boolean flags → int
    for col in ["UnusualLocation", "UnusualAmount", "NewDevice"]:
        df[col] = df[col].astype(int)

    # Engineered features — same formulas as training pipeline
    avg = float(data.get("AvgTransactionAmount", 1)) or 1.0
    amount = float(data["Amount"])

    df["AmountRatio"]   = amount / avg
    df["IsLargeAmount"] = int(amount / avg > 3)
    df["IsNightTime"]   = int(int(data.get("Hour", 12)) < 5)
    df["HighFailed"]    = int(int(data.get("FailedAttempts", 0)) > 2)
    df["RiskFlagSum"]   = (
        int(data.get("UnusualLocation", 0)) +
        int(data.get("UnusualAmount", 0)) +
        int(data.get("NewDevice", 0)) +
        df["HighFailed"].iloc[0]
    )
    df["IsHighIPFreq"]  = int(int(data.get("IP_Frequency", 0)) > 10)

    # FreqIsHigh uses the raw string (before encoding) sourced from data dict
    raw_freq = data.get("TransactionFrequency", "1/day")
    df["FreqIsHigh"]    = int(raw_freq == "5/day")

    # Reorder to match feature_names from training (fills 0 for any missing col)
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk(model_prob: float, data: dict) -> float:
    """
    Combine raw XGBoost probability with domain-rule boosts.

    model_prob is predict_proba output for the fraud class (0.0–1.0).
    Returns a final risk score capped at 1.0.
    """
    risk = float(model_prob)
    avg  = float(data.get("AvgTransactionAmount", 1)) or 1.0
    ratio = float(data["Amount"]) / avg

    # Amount anomaly
    if ratio > 3:
        risk += 0.20
    elif ratio > 2:
        risk += 0.10

    # Failed attempts
    fa = int(data.get("FailedAttempts", 0))
    if fa > 2:
        risk += 0.15
    elif fa > 0:
        risk += 0.05

    # Device / location / amount flags
    if int(data.get("NewDevice", 0)) == 1:
        risk += 0.10
    if int(data.get("UnusualLocation", 0)) == 1:
        risk += 0.10
    if int(data.get("UnusualAmount", 0)) == 1:
        risk += 0.05

    # Late-night
    if int(data.get("Hour", 12)) < 5:
        risk += 0.05

    # High frequency
    if str(data.get("TransactionFrequency", "")) == "5/day":
        risk += 0.05

    return min(risk, 1.0)


def risk_label(score: float) -> str:
    """
    Map risk score (0.0–1.0) to label using thresholds from schemas.py.
    LOW < 0.30, MEDIUM < 0.65, HIGH >= 0.65
    """
    if score < RISK_LOW:
        return "LOW"
    if score < RISK_MEDIUM:
        return "MEDIUM"
    return "HIGH"


# ══════════════════════════════════════════════════════════════════════════════
# REASONS — generated from actual submitted data, never hardcoded
# ══════════════════════════════════════════════════════════════════════════════
def build_reasons(data: dict) -> list[str]:
    reasons = []
    avg   = float(data.get("AvgTransactionAmount", 1)) or 1.0
    ratio = float(data["Amount"]) / avg

    if ratio > 3:
        reasons.append(
            f"Amount Rs.{data['Amount']:,.0f} is {ratio:.1f}x your average — highly anomalous"
        )
    elif ratio > 2:
        reasons.append(
            f"Amount Rs.{data['Amount']:,.0f} is {ratio:.1f}x your average transaction"
        )

    fa = int(data.get("FailedAttempts", 0))
    if fa > 2:
        reasons.append(f"{fa} failed attempts detected — possible brute force attack")
    elif fa > 0:
        reasons.append(f"{fa} failed transaction attempt(s) detected")

    if int(data.get("NewDevice", 0)) == 1:
        reasons.append("Transaction from an unrecognised new device")
    if int(data.get("UnusualLocation", 0)) == 1:
        reasons.append("Location flagged as unusual for this account")
    if int(data.get("UnusualAmount", 0)) == 1:
        reasons.append("Transaction amount is outside your normal spending pattern")

    h = int(data.get("Hour", 12))
    if h < 5:
        reasons.append(f"Late-night transaction at {h:02d}:00 — high-risk time window")

    if str(data.get("TransactionFrequency", "")) == "5/day":
        reasons.append("Abnormally high transaction frequency: 5 per day")

    ipf = int(data.get("IP_Frequency", 0))
    if ipf > 10:
        reasons.append(f"IP address used {ipf} times — potentially compromised or shared")

    return reasons


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# SHAP TreeExplainer on XGBoost returns a single ndarray of shape (n_samples, n_features).
# Each value is the SHAP contribution toward the positive class (class 1 = fraud).
# Positive → pushes toward fraud. Negative → pushes away from fraud.
# ══════════════════════════════════════════════════════════════════════════════
def get_shap(X: pd.DataFrame) -> dict:
    """
    Returns a dict with two lists:
      - 'increasing': top positive SHAP contributors (push toward fraud)
      - 'decreasing': top negative SHAP contributors (push away from fraud)
    Each entry: {feature, shap_value, display, positive}
    """
    try:
        sv = shap_explainer.shap_values(X)

        # XGBoost TreeExplainer returns ndarray (n_samples, n_features) for the
        # positive class. If it returns a list (older shap), take index FRAUD_CLASS_IDX.
        if isinstance(sv, list):
            vals = sv[FRAUD_CLASS_IDX][0]
        else:
            vals = sv[0]  # shape (n_features,)

        # Build sorted list of all features
        features = []
        for feat, val in zip(feature_names, vals):
            features.append({
                "feature":    feat,
                "shap_value": round(float(val), 4),
                "display":    round(abs(float(val)), 4),
                "positive":   bool(val > 0),
            })

        # Split into increasing / decreasing, sorted by absolute magnitude
        increasing = sorted(
            [f for f in features if f["positive"]],
            key=lambda x: x["display"], reverse=True
        )[:6]

        decreasing = sorted(
            [f for f in features if not f["positive"]],
            key=lambda x: x["display"], reverse=True
        )[:4]

        # Also produce a flat top-10 list (for API compatibility)
        all_sorted = sorted(features, key=lambda x: x["display"], reverse=True)[:10]

        return {
            "increasing": increasing,
            "decreasing": decreasing,
            "all":        all_sorted,
        }

    except Exception as e:
        logger.warning("SHAP error: %s", e)
        return {"increasing": [], "decreasing": [], "all": []}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"message": "RealGuard v2.2 — visit /static/index.html or /docs"})


@app.get("/health", tags=["System"])
def health():
    """Real health check — tests model load and DB connection."""
    try:
        from app.database import get_connection
        get_connection().execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False

    status = "healthy" if (model is not None and db_ok) else "degraded"
    return {
        "status":       status,
        "model_loaded": bool(model is not None),
        "db_connected": db_ok,
        "fraud_class":  int(FRAUD_CLASS_IDX),
        "features":     len(feature_names) if feature_names else 0,
    }


@app.post("/predict", tags=["Prediction"])
def predict(txn: TransactionInput):
    """
    Analyse a UPI transaction for fraud.

    Steps:
    1. Validate input (Pydantic)
    2. Preprocess + feature engineering
    3. XGBoost predict_proba → raw_model_probability (class 1 = fraud)
    4. Domain-rule risk boost → final_risk (capped at 1.0)
    5. SHAP explanation (TreeExplainer, fraud class)
    6. Store in SQLite history
    7. Return full prediction response
    """
    if model is None:
        raise HTTPException(503, "Model not loaded — server is starting up.")

    try:
        data = txn.model_dump()
        X    = preprocess(data)

        # ── Probability ──
        raw_prob   = float(model.predict_proba(X)[0][FRAUD_CLASS_IDX])
        final_risk = compute_risk(raw_prob, data)
        risk_score = round(final_risk * 100)
        level      = risk_label(final_risk)
        is_fraud   = final_risk >= FRAUD_THRESHOLD

        # ── Explanation ──
        reasons    = build_reasons(data)
        shap_data  = get_shap(X)

        # ── Summary (uses original submitted values, no mapping side-effects) ──
        summary = {
            "amount":          float(data["Amount"]),
            "avg_amount":      float(data["AvgTransactionAmount"]),
            "merchant":        txn.MerchantCategory,
            "txn_type":        txn.TransactionType,
            "hour":            int(data["Hour"]),
            "day_of_week":     int(data["DayOfWeek"]),
            "location":        f"{float(data['Latitude']):.4f}, {float(data['Longitude']):.4f}",
            "new_device":      bool(int(data["NewDevice"])),
            "unusual_loc":     bool(int(data["UnusualLocation"])),
            "unusual_amt":     bool(int(data["UnusualAmount"])),
            "failed_attempts": int(data["FailedAttempts"]),
            "ip_frequency":    int(data["IP_Frequency"]),
            "amount_ratio":    round(float(data["Amount"]) / (float(data["AvgTransactionAmount"]) or 1), 2),
        }

        # ── Persist ──
        txn_id = "TXN-" + str(uuid.uuid4())[:8].upper()
        insert_transaction({
            "txn_id":          txn_id,
            "created_at":      datetime.utcnow().isoformat(),
            "amount":          float(data["Amount"]),
            "avg_amount":      float(data["AvgTransactionAmount"]),
            "merchant":        txn.MerchantCategory,
            "txn_type":        txn.TransactionType,
            "txn_hour":        int(data["Hour"]),
            "day_of_week":     int(data["DayOfWeek"]),
            "unusual_loc":     int(data["UnusualLocation"]),
            "unusual_amt":     int(data["UnusualAmount"]),
            "new_device":      int(data["NewDevice"]),
            "failed_attempts": int(data["FailedAttempts"]),
            "fraud_prob":      round(final_risk, 4),
            "risk_score":      risk_score,
            "risk_level":      level,
            "is_fraud":        int(is_fraud),
        })

        return {
            "txn_id":                txn_id,
            "prediction":            "FRAUD" if is_fraud else "LEGITIMATE",
            "is_fraud":              is_fraud,
            "fraud_probability":     round(final_risk, 4),
            "raw_model_probability": round(raw_prob, 4),
            "risk_score":            risk_score,
            "risk_level":            level,
            "reasons":               reasons,
            "shap_features":         shap_data["all"],        # flat top-10 (backward compat)
            "shap_increasing":       shap_data["increasing"], # new: split view
            "shap_decreasing":       shap_data["decreasing"], # new: split view
            "transaction_summary":   summary,
        }

    except HTTPException:
        raise
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Unable to analyse this transaction. Please check your input and try again.")


@app.get("/history", tags=["History"])
def history(
    limit:      int  = Query(100, ge=1, le=500),
    fraud_only: bool = False,
    legit_only: bool = False,
    search:     str  = "",
):
    return fetch_history(limit=limit, fraud_only=fraud_only, legit_only=legit_only, search=search)


@app.delete("/history", tags=["History"])
def delete_history():
    clear_history()
    return {"message": "History cleared."}


@app.get("/analytics", tags=["Analytics"])
def analytics():
    return fetch_analytics()


@app.get("/model-metrics", tags=["Model"])
def get_model_metrics():
    if not model_metrics:
        raise HTTPException(503, "Metrics not available — model_metrics.pkl missing.")
    return model_metrics


@app.get("/categories", tags=["Meta"])
def get_categories():
    """Return valid category lists so the frontend can stay in sync with the backend."""
    return {
        "merchant_categories":  VALID_MERCHANT_CATEGORIES,
        "transaction_types":    VALID_TRANSACTION_TYPES,
        "transaction_frequencies": ["1/day", "3/day", "5/day"],
    }
