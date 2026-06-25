import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/fraud_model.pkl")

# Load data
raw_df = pd.read_csv("data/raw/synthetic_indian_upi_fraud_data.csv")
df = pd.read_csv("data/processed/clean_data.csv")

X = df.drop("FraudFlag", axis=1)

# Predictions
probs = model.predict_proba(X)[:, 1]
preds = model.predict(X)

# Add results
raw_df["Fraud_Probability"] = probs
raw_df["Prediction"] = preds

# -----------------------------
# 1. Fraud vs Normal Count
# -----------------------------
counts = raw_df["Prediction"].value_counts()

plt.figure()
counts.plot(kind="bar")
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.savefig("models/fraud_vs_normal.png", bbox_inches='tight')

# -----------------------------
# 2. Fraud Probability Distribution
# -----------------------------
plt.figure()
plt.hist(probs, bins=30)
plt.title("Fraud Probability Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Number of Transactions")
plt.savefig("models/fraud_probability_hist.png", bbox_inches='tight')

# -----------------------------
# 3. Top Fraud Transactions
# -----------------------------
top_fraud = raw_df.sort_values(by="Fraud_Probability", ascending=False).head(10)

plt.figure()
plt.bar(top_fraud["TransactionID"].astype(str), top_fraud["Fraud_Probability"])
plt.xticks(rotation=45)
plt.title("Top 10 High-Risk Transactions")
plt.xlabel("Transaction ID")
plt.ylabel("Fraud Probability")
plt.savefig("models/top_fraud_transactions.png", bbox_inches='tight')

print("✅ Report images saved in models/ folder!")