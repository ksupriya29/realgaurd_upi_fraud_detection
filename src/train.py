import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/processed/clean_data.csv")

X = df.drop("FraudFlag", axis=1)
y = df["FraudFlag"]

encoders = {}

# Encode categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature importance (EXPLANATION REPLACEMENT FOR SHAP)
importances = model.feature_importances_
feature_names = list(X.columns)

top_idx = np.argsort(importances)[::-1][:5]

top_features = [
    {"feature": feature_names[i], "importance": float(importances[i])}
    for i in top_idx
]

# Save artifacts
os.makedirs("artifacts", exist_ok=True)

pickle.dump(model, open("artifacts/model.pkl", "wb"))
pickle.dump(encoders, open("artifacts/encoders.pkl", "wb"))
pickle.dump(feature_names, open("artifacts/columns.pkl", "wb"))
pickle.dump(top_features, open("artifacts/top_features.pkl", "wb"))

print("✅ Model trained successfully")