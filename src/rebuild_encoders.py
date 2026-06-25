import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# -------------------------
# Load processed data
# -------------------------
df = pd.read_csv("data/processed/clean_data.csv")

categorical_cols = ["MerchantCategory", "TransactionType"]

encoders = {}

# -------------------------
# Fit encoders
# -------------------------
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])

    encoders[col] = le   # IMPORTANT: store FULL LabelEncoder object

# -------------------------
# Save encoders
# -------------------------
os.makedirs("models", exist_ok=True)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ encoders.pkl created successfully")