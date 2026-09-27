# 🔐 RealGuard — UPI Fraud Detection & Risk Analysis

**RealGuard** is an end-to-end UPI fraud detection and risk analysis application that combines **Machine Learning, Explainable AI (SHAP), rule-based risk scoring, FastAPI, Streamlit, and SQLite** to analyze suspicious transactions.

The application accepts transaction details, predicts the probability of fraud using an **XGBoost classifier**, applies additional risk rules, generates a final risk score, explains the prediction using **SHAP**, and stores transaction results for historical analysis.



## 🚀 Live Demo

### 🌐 Try RealGuard Online

👉 **Live Demo:**  
https://realguard-upi-fraud-supriya.streamlit.app/

The deployed application allows users to:

- Enter transaction details
- Analyze transactions
- Detect potential fraud
- View fraud probability
- View risk score and risk level
- View fraud risk indicators
- Understand model decisions using SHAP
- View transaction history
- View analytics
- View model performance

---

## 💻 GitHub Repository

👉 https://github.com/ksupriya29/realgaurd_upi_fraud_detection

---

# 📌 Project Overview

UPI transactions are fast and convenient, but fraudulent transactions can occur due to unusual transaction amounts, new devices, repeated failed attempts, unusual locations, high transaction frequency, and other suspicious patterns.

RealGuard provides a machine-learning-based system for analyzing these transaction characteristics.

The system combines:

- **XGBoost** for fraud prediction
- **SMOTE** for handling class imbalance during training
- **Feature engineering** for additional transaction signals
- **Rule-based risk scoring** for domain-specific risk factors
- **SHAP** for explainable predictions
- **FastAPI** for backend API services
- **Streamlit** for the interactive dashboard
- **SQLite** for transaction history and analytics

> RealGuard is an academic/portfolio project and is not intended to replace production-grade banking fraud detection systems.

---

# ✨ Features

| Feature | Description |
|---|---|
| 🔍 Transaction Analysis | Analyze a UPI transaction using machine learning |
| 🤖 XGBoost Prediction | Predict fraud probability using an XGBoost classifier |
| 📊 Risk Score | Calculate a final risk score using model probability and additional rules |
| 🚨 Fraud Classification | Classify transactions as FRAUD or LEGITIMATE |
| ⚠️ Risk Level | LOW, MEDIUM, or HIGH risk classification |
| 🧠 SHAP Explainability | Show features that increase or decrease fraud risk |
| 📋 Transaction History | Store and view analyzed transactions |
| 📈 Analytics Dashboard | View fraud rate, average amount, risk and transaction distribution |
| 📊 Model Performance | Display model evaluation metrics and confusion matrix |
| 🌐 FastAPI | REST API for transaction prediction |
| 🖥️ Streamlit | Interactive web dashboard |
| 🗄️ SQLite | Store transaction history locally |
| 📚 API Documentation | Swagger UI and ReDoc documentation |

---

# 🏗️ System Architecture

```text
                         REALGUARD
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       Streamlit Dashboard             FastAPI API
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    Input Validation
                             │
                             ▼
                    Feature Engineering
                             │
                             ▼
                      XGBoost Model
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          Fraud Probability          SHAP Explainer
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                       Risk Engine
                             │
                             ▼
                   Final Risk Assessment
                             │
                             ▼
                          SQLite
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          Transaction History        Analytics
🧰 Tech Stack
Backend
Python
FastAPI
Uvicorn
Pydantic
SQLite
Machine Learning
XGBoost
Scikit-learn
Pandas
NumPy
SMOTE / imbalanced-learn
Explainable AI
SHAP
Frontend / Dashboard
Streamlit
HTML
CSS
JavaScript
Deployment
Streamlit Community Cloud
Render-compatible FastAPI deployment
📁 Project Structure
realgaurd_upi_fraud_detection/
│
├── app/
│   ├── main.py
│   ├── database.py
│   ├── schemas.py
│   │
│   └── static/
│       └── index.html
│
├── models/
│   ├── xgb_model.pkl
│   ├── encoders_v2.pkl
│   ├── feature_names.pkl
│   └── model_metrics.pkl
│
├── data/
│
├── src/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── realguard.db

realguard.db is a SQLite database created at runtime when transaction data is stored.

📊 Dataset

The project uses transaction-level data for fraud detection.

The dataset contains transaction characteristics such as:

Transaction amount
Merchant category
Transaction type
Location
Transaction frequency
Device information
Failed attempts
Time information
User transaction count
IP frequency
Phone usage count
Dataset Information
Property	Value
Dataset Size	10,000 transactions
Input Fields	16
Engineered Features	7
Total Model Features	23
Target Variable	FraudFlag

The dataset is used for academic and demonstration purposes.

🧩 Input Features

RealGuard accepts 16 transaction-level input fields.

Amount
MerchantCategory
TransactionType
Latitude
Longitude
AvgTransactionAmount
TransactionFrequency
UnusualLocation
UnusualAmount
NewDevice
FailedAttempts
Hour
DayOfWeek
User_Transaction_Count
IP_Frequency
Phone_Usage_Count
⚙️ Feature Engineering

Before prediction, RealGuard derives additional features from the input transaction data.

The application generates 7 engineered features:

AmountRatio
IsLargeAmount
IsNightTime
HighFailed
RiskFlagSum
IsHighIPFreq
FreqIsHigh

Therefore:

16 Input Features
        +
7 Engineered Features
        =
23 Model Features

These engineered features help the model capture additional transaction-level risk patterns.

🤖 Machine Learning Model

RealGuard uses an XGBoost classifier for fraud detection.

The trained model is stored as:

models/xgb_model.pkl

The application uses:

model.predict_proba(...)

to obtain the model's fraud probability.

⚖️ Class Imbalance Handling

Fraud datasets can contain significantly fewer fraudulent transactions than legitimate transactions.

During model training, SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in the training data.

This helps provide the model with a more balanced representation of the minority fraud class during training.

🔬 Model Features

The final model uses 23 features:

Original Input Features
Amount
MerchantCategory
TransactionType
Latitude
Longitude
AvgTransactionAmount
TransactionFrequency
UnusualLocation
UnusualAmount
NewDevice
FailedAttempts
Hour
DayOfWeek
User_Transaction_Count
IP_Frequency
Phone_Usage_Count
Engineered Features
AmountRatio
IsLargeAmount
IsNightTime
HighFailed
RiskFlagSum
IsHighIPFreq
FreqIsHigh
🧠 Explainable AI with SHAP

RealGuard uses SHAP (SHapley Additive exPlanations) to explain model predictions.

The application uses:

shap.TreeExplainer(model)

to calculate feature contributions.

The dashboard separates the important factors into:

Increasing Risk

Features that contribute toward a higher fraud risk.

Decreasing Risk

Features that contribute toward a lower fraud risk.

All Important Features

The application also displays the most important SHAP contributors.

This makes the prediction more interpretable instead of showing only:

FRAUD

or

LEGITIMATE
📈 Risk Scoring

RealGuard combines the machine-learning probability with additional domain-specific risk indicators.

The basic risk calculation is:

risk = model_probability

Additional risk adjustments can be applied:

+ 0.20 if Amount > AvgAmount × 3
+ 0.10 if Amount > AvgAmount × 2
+ 0.15 if FailedAttempts > 2
+ 0.10 if NewDevice = True
+ 0.10 if UnusualLocation = True
+ 0.05 if UnusualAmount = True
+ 0.05 if Hour < 5
+ 0.05 if TransactionFrequency = "5/day"

The final value is bounded to the application's supported risk range.

🚨 Fraud Classification

The final risk value is used to classify the transaction.

Final Risk >= 0.50
        ↓
      FRAUD
Final Risk < 0.50
        ↓
   LEGITIMATE
⚠️ Risk Levels

Risk levels are evaluated separately from the fraud classification threshold.

Final Risk	Risk Level
< 0.30	LOW
0.30 – < 0.65	MEDIUM
>= 0.65	HIGH

Therefore, a transaction can have a MEDIUM risk level while still being classified as FRAUD when its final risk is at least 0.50.

🚩 Risk Indicators

RealGuard generates human-readable reasons for suspicious transactions.

Potential indicators include:

Unusual transaction amount
Multiple failed attempts
New device
Unusual location
Unusual amount
Late-night transaction
High transaction frequency
High IP frequency

These indicators are shown alongside the prediction to provide additional context.

🔌 REST API

FastAPI provides the backend REST API.

Available Endpoints
Method	Endpoint	Description
GET	/	API information
GET	/health	Health check
POST	/predict	Analyze a transaction
GET	/history	Retrieve transaction history
DELETE	/history	Clear transaction history
GET	/analytics	Transaction analytics
GET	/model-metrics	Model evaluation metrics
GET	/categories	Valid transaction categories and types
GET	/docs	Swagger API documentation
GET	/redoc	ReDoc API documentation
GET	/static/index.html	FastAPI-served web UI
📡 Prediction API
Endpoint
POST /predict

The endpoint accepts transaction information and returns:

Transaction ID
Prediction
Fraud probability
Raw model probability
Risk score
Risk level
Risk reasons
SHAP feature contributions
Transaction summary
📝 Example Request
{
  "Amount": 5000,
  "MerchantCategory": "Electronics",
  "TransactionType": "P2M",
  "Latitude": 17.385,
  "Longitude": 78.4867,
  "AvgTransactionAmount": 2500,
  "TransactionFrequency": "2/day",
  "UnusualLocation": false,
  "UnusualAmount": true,
  "NewDevice": true,
  "FailedAttempts": 2,
  "Hour": 14,
  "DayOfWeek": "Monday",
  "User_Transaction_Count": 10,
  "IP_Frequency": 3,
  "Phone_Usage_Count": 5
}

The exact categorical values accepted by the API should be obtained from the application's /categories endpoint.

📤 Example Response

A simplified response looks like:

{
  "prediction": "FRAUD",
  "fraud_probability": 0.98,
  "risk_score": 98,
  "risk_level": "HIGH"
}

The actual API response also provides additional information such as:

txn_id
is_fraud
raw_model_probability
reasons
shap_features
shap_increasing
shap_decreasing
transaction_summary
📚 API Documentation

After starting the FastAPI application locally:

Swagger UI
http://127.0.0.1:8000/docs
ReDoc
http://127.0.0.1:8000/redoc
📋 Transaction History

Every analyzed transaction can be stored in the SQLite database.

The history section can be used to review previously analyzed transactions.

Stored information can include:

Transaction ID
Amount
Prediction
Fraud probability
Risk score
Risk level
Transaction details
Timestamp
📊 Analytics Dashboard

The Streamlit dashboard provides transaction analytics such as:

Total Transactions
Fraud Transactions
Legitimate Transactions
Fraud Rate
Average Transaction Amount
Average Risk Score
Transaction Distribution

Example dashboard metrics:

Total Transactions
Fraud Transactions
Legitimate Transactions
Fraud Rate
Average Amount
Average Risk
📈 Model Performance

The application provides model evaluation information such as:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix

The model metrics are stored in:

models/model_metrics.pkl

The Streamlit application provides a dedicated Model Performance section for viewing the available evaluation results.

Since the project uses a synthetic/academic dataset, model metrics should not be interpreted as real-world banking fraud detection performance.

🖥️ Streamlit Dashboard

The Streamlit application contains the following main sections:

Analyse
History
Analytics
Model Performance
Analyse

Allows the user to:

Enter transaction information
Run fraud detection
View fraud probability
View risk score
View risk level
View risk indicators
View SHAP explanations
View transaction summary
History

Displays previously analyzed transactions.

Analytics

Displays:

Transaction counts
Fraud rate
Average transaction amount
Average risk
Transaction distribution
Model Performance

Displays available model evaluation metrics and performance information.

🚀 Run Locally
1. Clone the Repository
git clone https://github.com/ksupriya29/realgaurd_upi_fraud_detection.git
cd realgaurd_upi_fraud_detection
🐍 2. Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1
📦 3. Install Dependencies
pip install -r requirements.txt

If required:

python -m pip install -r requirements.txt
▶️ 4. Run FastAPI
uvicorn app.main:app --reload --port 8000

Open:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
🖥️ 5. Run Streamlit

Use:

python -m streamlit run streamlit_app.py

On Windows with the project virtual environment:

.\venv\Scripts\python.exe -m streamlit run streamlit_app.py

Streamlit will provide a local URL such as:

http://localhost:8501
☁️ Streamlit Cloud Deployment

The current live Streamlit deployment uses:

Repository:
ksupriya29/realgaurd_upi_fraud_detection

Branch:
main

Main file:
streamlit_app.py

Live application:

https://realguard-upi-fraud-supriya.streamlit.app/
🌐 Render Deployment

The FastAPI backend can also be deployed as a web service on Render.

A typical start command is:

uvicorn app.main:app --host 0.0.0.0 --port $PORT

The FastAPI backend can then expose:

/docs

for API documentation.

Render deployment is an available deployment option for the FastAPI backend; the currently documented public demo is the Streamlit Cloud application above.

🔐 Security & Validation

The project includes validation and application-level checks for transaction input.

The application also includes:

Pydantic request validation
CORS middleware
Input preprocessing
Model artifact validation
Database storage
Error handling
Prediction validation

For a real financial deployment, additional controls would be required.

⚠️ Limitations

RealGuard is a portfolio/academic fraud detection project.

Current limitations include:

Uses an academic/synthetic dataset
No real banking transaction integration
No user authentication
SQLite is not intended for high-concurrency production workloads
No real-time banking infrastructure
Model performance depends on the training dataset
Rule-based risk adjustments are application-specific
No production-grade monitoring infrastructure
No guaranteed detection of previously unseen fraud patterns

Therefore, the system should not be used as a real banking security system without substantial additional validation and engineering.

🔮 Future Improvements

Potential improvements include:

Real-time transaction streaming
Larger real-world datasets
Advanced anomaly detection
Deep learning fraud models
Graph-based fraud detection
User authentication and authorization
PostgreSQL or other production database
Redis caching
Kafka-based transaction streaming
Model monitoring
Drift detection
Automated model retraining
Alert notifications
Role-based access control
Secure cloud deployment
Containerization with Docker
CI/CD pipelines
🏭 Potential Production Architecture

A larger-scale production architecture could use:

UPI Transaction
       │
       ▼
API Gateway
       │
       ▼
Authentication
       │
       ▼
Fraud Detection Service
       │
 ┌─────┴──────────┐
 ▼                ▼
ML Model       Rule Engine
 │                │
 └──────┬─────────┘
        ▼
 Risk Aggregation
        │
 ┌──────┴─────────┐
 ▼                ▼
Fraud Alert     Database
        │
        ▼
 Monitoring & Analytics

Possible technologies include:

FastAPI
PostgreSQL
Redis
Kafka
Docker
Kubernetes
Cloud monitoring
Model monitoring systems

These are potential future production components and are not required by the current implementation.

📚 Learning Outcomes

This project provided practical experience with:

Machine Learning
XGBoost
Imbalanced classification
SMOTE
Feature engineering
Explainable AI
SHAP
FastAPI
REST APIs
Streamlit
SQLite
Data preprocessing
Model evaluation
Risk scoring
Dashboard development
Cloud deployment
⭐ Project Highlights
Machine Learning
XGBoost
+
SMOTE
+
Feature Engineering
Explainability
SHAP
+
Risk Indicators
+
Transaction Summary
Application
FastAPI
+
Streamlit
+
SQLite
Deployment
Streamlit Community Cloud
🔗 Project Links
🌐 Live Demo

https://realguard-upi-fraud-supriya.streamlit.app/

💻 GitHub Repository

https://github.com/ksupriya29/realgaurd_upi_fraud_detection

👩‍💻 Author

Supriya

B.Tech – Computer Science and Engineering
BVRIT Hyderabad College of Engineering for Women

GitHub:

https://github.com/ksupriya29

📄 License

This project is intended for educational and portfolio purposes.

If you reuse or extend this project, please provide appropriate attribution.

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

🔐 RealGuard

Machine Learning + Explainable AI + Risk Analysis for UPI Fraud Detection


### One important note

Before you run:

```powershell
git add README.md
git commit -m "Finalize RealGuard README"
git push origin main