"""
Pydantic schemas with input validation for RealGuard API.

Source-of-truth categories come directly from the training dataset:
  - TransactionType: ['P2M', 'P2P']
  - MerchantCategory: ['Clothing', 'Electronics', 'Entertainment',
                        'Groceries', 'Restaurants', 'Travel', 'Utilities']
  - TransactionFrequency: ['1/day', '3/day', '5/day']
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# ── Canonical categories (must match training data exactly) ────────────────
VALID_MERCHANT_CATEGORIES = [
    "Clothing", "Electronics", "Entertainment",
    "Groceries", "Restaurants", "Travel", "Utilities",
]

VALID_TRANSACTION_TYPES = ["P2M", "P2P"]

VALID_FREQUENCIES = ["1/day", "3/day", "5/day"]

# Risk thresholds (single source of truth — used by main.py too)
RISK_LOW    = 0.30   # < 0.30 → LOW
RISK_MEDIUM = 0.65   # 0.30–0.65 → MEDIUM, ≥ 0.65 → HIGH
FRAUD_THRESHOLD = 0.50   # final_risk ≥ 0.50 → is_fraud = True


class TransactionInput(BaseModel):
    Amount: float = Field(..., gt=0, description="Transaction amount in INR (must be > 0)")
    MerchantCategory: str = Field(..., description="Merchant category")
    TransactionType: str = Field(..., description="Transaction type: P2M or P2P")
    Latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    Longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    AvgTransactionAmount: float = Field(..., gt=0, description="Average transaction amount (> 0)")
    TransactionFrequency: str = Field(default="1/day", description="1/day, 3/day, or 5/day")
    UnusualLocation: int = Field(..., ge=0, le=1, description="0 or 1")
    UnusualAmount: int = Field(..., ge=0, le=1, description="0 or 1")
    NewDevice: int = Field(..., ge=0, le=1, description="0 or 1")
    FailedAttempts: int = Field(..., ge=0, le=100, description="Number of failed attempts (0–100)")
    Hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    DayOfWeek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday … 6=Sunday)")
    User_Transaction_Count: int = Field(..., ge=0, description="User's total transaction count")
    IP_Frequency: int = Field(..., ge=0, description="How many times this IP was used")
    Phone_Usage_Count: int = Field(..., ge=0, description="Phone usage count")

    @field_validator("MerchantCategory")
    @classmethod
    def validate_merchant(cls, v: str) -> str:
        if v not in VALID_MERCHANT_CATEGORIES:
            raise ValueError(
                f"Invalid MerchantCategory '{v}'. "
                f"Must be one of: {', '.join(VALID_MERCHANT_CATEGORIES)}"
            )
        return v

    @field_validator("TransactionType")
    @classmethod
    def validate_txn_type(cls, v: str) -> str:
        if v not in VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Invalid TransactionType '{v}'. "
                f"Must be one of: {', '.join(VALID_TRANSACTION_TYPES)}"
            )
        return v

    @field_validator("TransactionFrequency")
    @classmethod
    def validate_freq(cls, v: str) -> str:
        if v not in VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid TransactionFrequency '{v}'. "
                f"Must be one of: {', '.join(VALID_FREQUENCIES)}"
            )
        return v


class PredictionResponse(BaseModel):
    txn_id: str
    prediction: str                  # "FRAUD" | "LEGITIMATE"
    is_fraud: bool
    fraud_probability: float         # final risk score 0.0–1.0
    raw_model_probability: float     # direct XGBoost predict_proba output
    risk_score: int                  # 0–100 (fraud_probability × 100, capped)
    risk_level: str                  # LOW | MEDIUM | HIGH
    reasons: List[str]               # rule-based human-readable flags
    shap_features: List[dict]        # [{feature, shap_value, direction, display, positive}]
    transaction_summary: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool


class AnalyticsResponse(BaseModel):
    total: int
    fraud: int
    legit: int
    fraud_rate: float
    avg_amount: float
    avg_risk: float
    by_merchant: list
    by_type: list
    by_hour: list
    by_day: list
    amounts: list


class ModelMetricsResponse(BaseModel):
    xgboost: dict
    logistic_regression: dict
    random_forest: dict
    roc_curve: dict
    pr_curve: dict
    feature_names: list
