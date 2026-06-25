import joblib
import pandas as pd

# Load model
model = joblib.load("models/fraud_model.pkl")

# Load data
raw_df = pd.read_csv("data/raw/synthetic_indian_upi_fraud_data.csv")
df = pd.read_csv("data/processed/clean_data.csv")

# Features
X = df.drop("FraudFlag", axis=1)

# Predictions
probs = model.predict_proba(X)[:, 1]
preds = model.predict(X)

# Add results
raw_df["Fraud_Probability"] = probs
raw_df["Prediction"] = preds

# -----------------------------
# OVERALL FRAUD STATS
# -----------------------------
total = len(raw_df)
fraud_count = (preds == 1).sum()
normal_count = total - fraud_count

fraud_percentage = (fraud_count / total) * 100

print("\n==============================")
print("📊 FRAUD SUMMARY REPORT")
print("==============================")

print(f"Total Transactions: {total}")
print(f"Fraud Transactions: {fraud_count}")
print(f"Normal Transactions: {normal_count}")
print(f"Overall Fraud Percentage: {fraud_percentage:.2f}%")

# -----------------------------
# Show Fraud Transaction IDs
# -----------------------------
fraud_ids = raw_df[raw_df["Prediction"] == 1]["TransactionID"]

print("\n🚨 Fraud Transaction IDs (Top 10):")
print(fraud_ids.head(10).to_list())

print("\n==============================")