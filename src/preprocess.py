import pandas as pd
import os

RAW_PATH = "data/raw/synthetic_indian_upi_fraud_data.csv"
OUT_PATH = "data/processed/clean_data.csv"

df = pd.read_csv(RAW_PATH)

print("Data Loaded:", df.shape)

# Convert timestamp → features
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

# Create behavioral features
df["User_Transaction_Count"] = df.groupby("UserID")["TransactionID"].transform("count")
df["IP_Frequency"] = df.groupby("IPAddress")["TransactionID"].transform("count")
df["Phone_Usage_Count"] = 1

# Drop unused columns
drop_cols = [
    "TransactionID",
    "UserID",
    "DeviceID",
    "IPAddress",
    "PhoneNumber",
    "BankName",
    "Timestamp"
]

df = df.drop(columns=drop_cols)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("✅ Clean data saved")