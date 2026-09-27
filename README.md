# 🔐 RealGuard — UPI Fraud Detection & Risk Analysis

A production-quality, end-to-end fraud detection platform for UPI transactions powered by **XGBoost**, **SHAP (Explainable AI)**, and **FastAPI**.

---

## Project Overview

RealGuard analyses UPI transactions in real time and tells you:

- Whether the transaction is **FRAUD** or **LEGITIMATE**
- The **raw XGBoost fraud probability** (0–100%)
- The **final risk score** (model probability + domain-rule boosts)
- **Why** the model made that decision (SHAP feature contributions, split into factors increasing vs reducing risk)
- A full **transaction summary** and **dynamic risk indicators**

---

## Features

| Feature | Description |
|---|---|
| Analyse | Real-time fraud prediction with SHAP explanations |
| History | SQLite-backed transaction history with search and filter |
| Analytics | Live dashboard with 7 charts built from real stored data |
| Model Performance | Actual metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Explainable AI | SHAP TreeExplainer — real values split into increasing/decreasing fraud risk |
| Validation | Pydantic v2 — categories enforced against training dataset |
| Deployment ready | FastAPI, CORS, Uvicorn, Render-compatible, .gitignore, render.yaml |

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **ML**: XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **XAI**: SHAP (TreeExplainer)
- **Database**: SQLite (stdlib sqlite3)
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js
- **Python**: 3.11+

---

## Architecture

```
User Transaction Input
        |
        v
Frontend (index.html)
        |
        v
FastAPI /predict
        |
        v
Pydantic Validation
(categories enforced against training dataset)
        |
        v
Preprocessing + Feature Engineering
(LabelEncoders from encoders_v2.pkl, matches training exactly)
        |
        v
XGBoost Model (xgb_model.pkl)
predict_proba()[fraud_class_index]
fraud class verified from model.classes_ at startup
        |
        v
Domain-Rule Risk Boost
(amount anomaly + device + location + failed attempts + frequency)
        |
        v
SHAP TreeExplainer
(values for fraud class, split into increasing/decreasing)
        |
        v
SQLite History (realguard.db)
        |
        v
JSON Response → Frontend
```

---

## Project Structure

```
rtrp/
├── app/
│   ├── __init__.py
│   ├── main.py            FastAPI backend v2.2
│   ├── database.py        SQLite layer
│   ├── schemas.py         Pydantic validation + canonical categories
│   └── static/
│       └── index.html     Production frontend
│
├── models/
│   ├── xgb_model.pkl      Trained XGBoost model
│   ├── encoders_v2.pkl    LabelEncoders (MerchantCategory, TransactionType, TransactionFrequency)
│   ├── feature_names.pkl  23 feature names in training order
│   └── model_metrics.pkl  Actual metrics (accuracy, F1, ROC-AUC, PR-AUC, curves)
│
├── data/
│   ├── raw/               Original synthetic dataset
│   └── processed/         clean_data.csv
│
├── src/
│   ├── preprocess.py      Data preprocessing
│   ├── train.py           Original training script
│   ├── rebuild_encoders.py
│   └── explain.py         SHAP explanation script
│
├── tests/
│   └── run_tests.py       73-assertion test suite
│
├── requirements.txt
├── render.yaml            Render deployment config
├── .gitignore
└── README.md
```

---

## Dataset

- **Source**: Synthetic Indian UPI Fraud Data
- **Records**: 10,000 transactions
- **Fraud Rate**: ~9.65% (965 fraud / 10,000)
- **TransactionType**: P2M, P2P (only these two — enforced)
- **MerchantCategory**: Clothing, Electronics, Entertainment, Groceries, Restaurants, Travel, Utilities (7 categories — enforced)
- **TransactionFrequency**: 1/day, 3/day, 5/day

> This is a synthetic dataset. Real-world fraud detection requires genuine transaction data.

---

## Model Details

| Aspect | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Class imbalance | SMOTE applied to training set only |
| Train/Test split | 80% / 20% (stratified) |
| Feature encoding | LabelEncoder for 3 categorical columns |
| Feature count | 23 (16 raw + 7 engineered) |
| Fraud class | Verified from model.classes_ at startup (class 1) |

---

## Risk Score Calculation

```
raw_model_probability = model.predict_proba(X)[0][fraud_class_index]
                        (fraud_class_index verified from model.classes_)

final_risk = raw_model_probability
           + 0.20  if Amount > AvgAmount x 3
           + 0.10  if Amount > AvgAmount x 2
           + 0.15  if FailedAttempts > 2
           + 0.05  if FailedAttempts > 0
           + 0.10  if NewDevice = 1
           + 0.10  if UnusualLocation = 1
           + 0.05  if UnusualAmount = 1
           + 0.05  if Hour < 5  (late night)
           + 0.05  if TransactionFrequency = 5/day

risk_score  = min(final_risk, 1.0) x 100   (0–100)
```

**Risk level thresholds:**

| Score | Level |
|---|---|
| 0–29 | LOW |
| 30–64 | MEDIUM |
| 65–100 | HIGH |

`is_fraud = True` when `final_risk >= 0.50`

---

## SHAP Integration

```python
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
# Returns ndarray shape (1, 23) for XGBoost — values for the fraud class
vals = shap_values[0]   # shape (23,)
```

- Positive SHAP value → pushes prediction toward fraud
- Negative SHAP value → pushes prediction toward legitimate
- UI shows two sections: **Factors Increasing Fraud Risk** and **Factors Reducing Fraud Risk**

---

## Engineered Features

All computed in backend from submitted transaction, matching training exactly:

| Feature | Formula |
|---|---|
| AmountRatio | Amount / AvgTransactionAmount |
| IsLargeAmount | 1 if AmountRatio > 3 |
| IsNightTime | 1 if Hour < 5 |
| HighFailed | 1 if FailedAttempts > 2 |
| RiskFlagSum | UnusualLocation + UnusualAmount + NewDevice + HighFailed |
| IsHighIPFreq | 1 if IP_Frequency > 10 |
| FreqIsHigh | 1 if TransactionFrequency == "5/day" |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/health` | Real health check (model + DB) |
| POST | `/predict` | Fraud prediction + SHAP |
| GET | `/history` | Transaction history |
| DELETE | `/history` | Clear history |
| GET | `/analytics` | Dashboard data |
| GET | `/model-metrics` | Model evaluation metrics |
| GET | `/categories` | Valid categories (for frontend sync) |
| GET | `/docs` | Swagger UI |
| GET | `/static/index.html` | Frontend |

### POST /predict — Request

```json
{
  "Amount": 25000,
  "MerchantCategory": "Electronics",
  "TransactionType": "P2M",
  "Latitude": 19.2,
  "Longitude": 72.8,
  "AvgTransactionAmount": 2000,
  "TransactionFrequency": "5/day",
  "UnusualLocation": 1,
  "UnusualAmount": 1,
  "NewDevice": 1,
  "FailedAttempts": 5,
  "Hour": 2,
  "DayOfWeek": 6,
  "User_Transaction_Count": 60,
  "IP_Frequency": 15,
  "Phone_Usage_Count": 10
}
```

**Valid values:**
- `MerchantCategory`: Clothing, Electronics, Entertainment, Groceries, Restaurants, Travel, Utilities
- `TransactionType`: P2M, P2P
- `TransactionFrequency`: 1/day, 3/day, 5/day

### POST /predict — Response

```json
{
  "txn_id": "TXN-1A2B3C4D",
  "prediction": "FRAUD",
  "is_fraud": true,
  "fraud_probability": 1.0,
  "raw_model_probability": 0.9999,
  "risk_score": 100,
  "risk_level": "HIGH",
  "reasons": ["Amount Rs.25,000 is 12.5x your average..."],
  "shap_features": [...],
  "shap_increasing": [...],
  "shap_decreasing": [...],
  "transaction_summary": {...}
}
```

---

## Model Performance

All metrics are real — computed on a held-out 20% test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 92.6% | 97.4% | 82.1% | 89.1% | 91.0% | 92.2% |

Confusion matrix (XGBoost, test set):
- True Negative: 1245
- False Positive: 16
- False Negative: 132
- True Positive: 607

---

## Local Setup

```bash
# 1. Enter project directory
cd rtrp

# 2. Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload --port 8000

# 5. Open in browser
#   UI:         http://127.0.0.1:8000/static/index.html
#   API docs:   http://127.0.0.1:8000/docs
#   Health:     http://127.0.0.1:8000/health
```

---

## Deployment (Render)

A `render.yaml` is included. Steps:

1. Push to GitHub (models/*.pkl must be included — they are not in .gitignore by default)
2. Create a **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repository
4. Render will use the `render.yaml` automatically, or set manually:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. SQLite `realguard.db` is auto-created at startup

> Note: Render's free tier uses ephemeral storage — transaction history resets on redeploy.
> For persistent history, use PostgreSQL.

---

## Validation Rules

| Field | Rule |
|---|---|
| Amount | > 0 |
| AvgTransactionAmount | > 0 |
| MerchantCategory | Must be one of 7 training categories |
| TransactionType | Must be P2M or P2P |
| TransactionFrequency | Must be 1/day, 3/day, or 5/day |
| Hour | 0–23 |
| DayOfWeek | 0–6 |
| Latitude | -90 to 90 |
| Longitude | -180 to 180 |
| FailedAttempts | >= 0 |

---

## Testing

```bash
# Run the test suite (73 assertions)
python tests/run_tests.py
# Expected: 73 PASSED | 0 FAILED
```

Tests cover: health check, legitimate transaction, suspicious transaction, high-risk transaction, input validation (7 cases), SHAP correctness, history, analytics, model metrics, category consistency, risk level consistency.

---

## Limitations

- Synthetic dataset — real performance would differ on genuine UPI data
- SQLite is not suitable for high-concurrency production (use PostgreSQL)
- SHAP computation adds ~50–200ms per request
- No user authentication (add JWT/OAuth2 for production)
- Render free tier: ephemeral SQLite storage

---

## Security Notes

- All inputs validated with Pydantic before reaching the model
- No raw stack traces exposed to users
- No API keys or secrets in code
- CORS enabled — restrict `allow_origins` for production

---

## Author

Developed by **Supriya**
