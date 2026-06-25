import shap
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("models/fraud_model.pkl")

# -----------------------------
# Load data
# -----------------------------
raw_df = pd.read_csv("data/raw/synthetic_indian_upi_fraud_data.csv")
df = pd.read_csv("data/processed/clean_data.csv")

X = df.drop("FraudFlag", axis=1)

# -----------------------------
# Given Fraud Transaction IDs
# -----------------------------
fraud_ids = [
    373481869464, 874207772966, 285614658293, 313793292686,
    616665746189, 799767300764, 517016864757, 342861071692,
    977913017472, 420305987588
]

print("\n==============================")
print("🚨 SELECTED FRAUD TRANSACTIONS EXPLANATION")
print("==============================\n")

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.Explainer(model)

# -----------------------------
# Feature explanation mapping
# -----------------------------
def explain_feature(feature, impact_positive):
    if feature == "UnusualAmount":
        return "The transaction amount is unusual compared to normal behavior, increasing fraud risk." if impact_positive else \
               "The transaction amount appears normal and does not indicate fraud."

    elif feature == "UnusualLocation":
        return "The transaction occurred in an unusual location, which is a strong indicator of fraud." if impact_positive else \
               "The location matches normal user behavior."

    elif feature == "TransactionFrequency":
        return "The user is performing transactions very frequently, which may indicate suspicious activity." if impact_positive else \
               "Transaction frequency appears normal."

    elif feature == "FailedAttempts":
        return "Multiple failed login or transaction attempts suggest possible unauthorized access." if impact_positive else \
               "No unusual failed attempts detected."

    elif feature == "NewDevice":
        return "The transaction is made from a new or unknown device, increasing fraud likelihood." if impact_positive else \
               "The device used is recognized and safe."

    elif feature == "Hour":
        return "The transaction occurred at an unusual time, which may indicate fraud." if impact_positive else \
               "The transaction time is typical."

    elif feature == "DayOfWeek":
        return "The transaction occurred on an unusual day pattern for this user." if impact_positive else \
               "The transaction day is consistent with normal behavior."

    elif feature == "IP_Frequency":
        return "This IP address is used very frequently, which may indicate suspicious activity." if impact_positive else \
               "IP usage appears normal."

    elif feature == "Phone_Usage_Count":
        return "This phone number is linked to multiple accounts, indicating possible fraud." if impact_positive else \
               "Phone number usage is normal."

    elif feature == "User_Transaction_Count":
        return "The user has an unusually high number of transactions, which is suspicious." if impact_positive else \
               "User transaction count is normal."

    else:
        return f"The feature '{feature}' influences the prediction."

# -----------------------------
# Loop through selected IDs
# -----------------------------
for txn_id in fraud_ids:

    # Find row index
    matches = raw_df[raw_df["TransactionID"] == txn_id]

    if matches.empty:
        print(f"\n❌ Transaction ID {txn_id} not found in dataset\n")
        continue

    idx = matches.index[0]

    x_input = X.iloc[idx:idx+1]

    # Prediction
    prob = model.predict_proba(x_input)[0][1]

    # SHAP explanation
    shap_values = explainer(x_input)

    contributions = shap_values.values[0]
    features = X.columns

    explain_df = pd.DataFrame({
        "Feature": features,
        "Impact": contributions
    })

    explain_df["AbsImpact"] = np.abs(explain_df["Impact"])
    explain_df = explain_df.sort_values(by="AbsImpact", ascending=False)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("--------------------------------------------------")
    print(f"Transaction ID: {txn_id}")
    print(f"Fraud Probability: {prob*100:.2f}%")

    print("\nDetailed Reasons:")

    for i in range(5):
        feature = explain_df.iloc[i]["Feature"]
        impact = explain_df.iloc[i]["Impact"]

        sentence = explain_feature(feature, impact > 0)
        print(f"- {sentence}")

    print("\nConclusion:")
    print("This transaction is classified as fraud due to multiple suspicious behavioral indicators.\n")

print("==============================")
print("✅ Completed explanation for selected fraud transactions")