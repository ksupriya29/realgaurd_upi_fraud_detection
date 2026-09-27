import streamlit as st
import pandas as pd
import numpy as np
import app.main as realguard


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="REALGUARD - UPI Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .realguard-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0;
    }

    .realguard-subtitle {
        font-size: 24px;
        font-weight: 500;
        margin-top: 8px;
        margin-bottom: 22px;
    }

    .live-box {
        background: #123e2b;
        border-radius: 8px;
        padding: 15px 20px;
        color: #4ade80;
        font-size: 17px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .fraud-card {
        background: #4a1717;
        border: 1px solid #ef4444;
    }

    .legit-card {
        background: #123e2b;
        border: 1px solid #22c55e;
    }

    .risk-number {
        font-size: 42px;
        font-weight: 800;
    }

    .risk-label {
        font-size: 18px;
        font-weight: 600;
    }

    .reason-card {
        background: #171b23;
        padding: 15px 18px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #ef4444;
    }

    .shap-positive {
        background: #2a1616;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #ef4444;
    }

    .shap-negative {
        background: #14281d;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #22c55e;
    }

    .summary-card {
        background: #171b23;
        padding: 18px;
        border-radius: 10px;
        margin-top: 10px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE BACKEND
# ============================================================

@st.cache_resource
def initialize_realguard():

    try:
        realguard.load_artifacts()
    except Exception:
        pass

    try:
        realguard.init_db()
    except Exception:
        pass

    return True


initialize_realguard()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="realguard-title">🔐 REALGUARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="realguard-subtitle">UPI Fraud Detection & Risk Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="live-box">● MODEL LIVE</div>',
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

analyse_tab, history_tab, analytics_tab, model_tab = st.tabs(
    [
        "🔍 Analyse",
        "📋 History",
        "📊 Analytics",
        "🤖 Model Performance"
    ]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    """
    Convert a scalar numeric value safely to float.

    Prevents Streamlit metric errors when backend values are
    numpy numbers, Decimal values, strings, or unexpected lists.
    """

    try:

        if isinstance(value, (list, tuple, dict, set)):
            return float(default)

        if value is None:
            return float(default)

        if isinstance(value, np.ndarray):

            if value.size == 1:
                return float(value.reshape(-1)[0])

            return float(default)

        return float(value)

    except Exception:
        return float(default)


def safe_int(value, default=0):
    try:

        if isinstance(value, (list, tuple, dict, set)):
            return int(default)

        if value is None:
            return int(default)

        if isinstance(value, np.ndarray):

            if value.size == 1:
                return int(value.reshape(-1)[0])

            return int(default)

        return int(float(value))

    except Exception:
        return int(default)


def scalar_value(data, keys, default=0.0):
    """
    Find the first usable scalar value from a dictionary.

    This is the main protection against the Analytics
    'Provided type: list' error.
    """

    if not isinstance(data, dict):
        return default

    for key in keys:

        if key not in data:
            continue

        value = data[key]

        if isinstance(value, (list, tuple, dict, set)):
            continue

        if isinstance(value, np.ndarray):

            if value.size != 1:
                continue

            value = value.reshape(-1)[0]

        try:
            return float(value)
        except Exception:
            continue

    return default


def normalize_records(records):
    """
    Convert history returned by the backend into a DataFrame.
    Handles list/dict/tuple responses safely.
    """

    if records is None:
        return pd.DataFrame()

    if isinstance(records, pd.DataFrame):
        return records.copy()

    if isinstance(records, dict):

        # Common API response formats
        for key in ["data", "history", "transactions", "records", "items"]:

            value = records.get(key)

            if isinstance(value, list):
                return pd.DataFrame(value)

        # Single record
        return pd.DataFrame([records])

    if isinstance(records, list):
        return pd.DataFrame(records)

    if isinstance(records, tuple):
        return pd.DataFrame(list(records))

    return pd.DataFrame()


def clean_dataframe_for_streamlit(df):
    """
    Convert numpy/object values into Streamlit-safe values.
    """

    if df.empty:
        return df

    result = df.copy()

    for column in result.columns:

        result[column] = result[column].apply(
            lambda value:
                value.item()
                if isinstance(value, np.generic)
                else value
        )

    return result


def extract_history_analytics(history_data):
    """
    Fallback analytics calculated directly from transaction history.

    This prevents Analytics from breaking if fetch_analytics()
    contains chart arrays/lists.
    """

    df = normalize_records(history_data)

    if df.empty:

        return {
            "total": 0,
            "fraud": 0,
            "legit": 0,
            "avg_amount": 0.0,
            "avg_risk": 0.0,
            "fraud_rate": 0.0
        }

    # -----------------------------
    # Amount
    # -----------------------------

    amount_column = None

    for column in ["amount", "Amount", "transaction_amount"]:

        if column in df.columns:
            amount_column = column
            break

    if amount_column is not None:

        amounts = pd.to_numeric(
            df[amount_column],
            errors="coerce"
        )

        avg_amount = float(amounts.mean()) if amounts.notna().any() else 0.0

    else:

        avg_amount = 0.0

    # -----------------------------
    # Fraud
    # -----------------------------

    fraud_column = None

    for column in [
        "is_fraud",
        "fraud",
        "FraudFlag",
        "fraud_flag",
        "prediction"
    ]:

        if column in df.columns:
            fraud_column = column
            break

    fraud_count = 0

    if fraud_column is not None:

        values = df[fraud_column]

        if fraud_column == "prediction":

            fraud_count = int(
                values.astype(str)
                .str.upper()
                .eq("FRAUD")
                .sum()
            )

        else:

            fraud_count = int(
                pd.to_numeric(
                    values,
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
                .eq(1)
                .sum()
            )

    total = len(df)

    legit_count = max(total - fraud_count, 0)

    # -----------------------------
    # Risk
    # -----------------------------

    risk_column = None

    for column in [
        "risk_score",
        "risk",
        "RiskScore"
    ]:

        if column in df.columns:
            risk_column = column
            break

    if risk_column is not None:

        risks = pd.to_numeric(
            df[risk_column],
            errors="coerce"
        )

        avg_risk = float(risks.mean()) if risks.notna().any() else 0.0

    else:

        avg_risk = 0.0

    fraud_rate = (
        (fraud_count / total) * 100
        if total > 0
        else 0.0
    )

    return {
        "total": int(total),
        "fraud": int(fraud_count),
        "legit": int(legit_count),
        "avg_amount": float(avg_amount),
        "avg_risk": float(avg_risk),
        "fraud_rate": float(fraud_rate)
    }


# ============================================================
# ANALYSE TAB
# ============================================================

with analyse_tab:

    st.markdown(
        '<div class="section-title">🔍 Analyse Transaction</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=1.0,
            value=500.0,
            step=100.0
        )

        avg_amount = st.number_input(
            "Average Transaction Amount (₹)",
            min_value=1.0,
            value=650.0,
            step=50.0
        )

        merchant = st.selectbox(
            "Merchant Category",
            [
                "Groceries",
                "Clothing",
                "Electronics",
                "Food",
                "Entertainment",
                "Travel",
                "Utilities"
            ]
        )

        txn_type = st.selectbox(
            "Transaction Type",
            ["P2M", "P2P"]
        )

        frequency = st.selectbox(
            "Transaction Frequency",
            ["1/day", "3/day", "5/day"]
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            value=17.3850,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=78.4867,
            format="%.6f"
        )

        transaction_frequency = st.number_input(
            "Transaction Count",
            min_value=0,
            value=5
        )

        user_transaction_count = st.number_input(
            "User Transaction Count",
            min_value=0,
            value=5
        )

        ip_frequency = st.number_input(
            "IP Frequency",
            min_value=0,
            value=2
        )

    with col3:

        phone_usage_count = st.number_input(
            "Phone Usage Count",
            min_value=0,
            value=5
        )

        hour = st.number_input(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=14
        )

        day_of_week = st.number_input(
            "Day of Week",
            min_value=0,
            max_value=6,
            value=2
        )

        failed_attempts = st.number_input(
            "Failed Attempts",
            min_value=0,
            value=0
        )

        st.markdown("### Risk Indicators")

        unusual_location = st.checkbox(
            "Unusual Location"
        )

        unusual_amount = st.checkbox(
            "Unusual Amount"
        )

        new_device = st.checkbox(
            "New Device"
        )

    st.markdown("---")

    analyse_button = st.button(
        "🔍 Analyse Transaction",
        type="primary",
        use_container_width=True
    )

    if analyse_button:

        transaction_data = {

            "Amount": float(amount),

            "MerchantCategory": merchant,

            "TransactionType": txn_type,

            "Latitude": float(latitude),

            "Longitude": float(longitude),

            "AvgTransactionAmount": float(avg_amount),

            "TransactionFrequency": frequency,

            "UnusualLocation": int(unusual_location),

            "UnusualAmount": int(unusual_amount),

            "NewDevice": int(new_device),

            "FailedAttempts": int(failed_attempts),

            "Hour": int(hour),

            "DayOfWeek": int(day_of_week),

            "User_Transaction_Count": int(user_transaction_count),

            "IP_Frequency": int(ip_frequency),

            "Phone_Usage_Count": int(phone_usage_count)
        }

        try:

            # -----------------------------------------
            # PREPROCESS
            # -----------------------------------------

            X = realguard.preprocess(transaction_data)

            # -----------------------------------------
            # MODEL PROBABILITY
            # -----------------------------------------

            probabilities = realguard.model.predict_proba(X)[0]

            fraud_index = int(realguard.FRAUD_CLASS_IDX)

            raw_probability = float(
                probabilities[fraud_index]
            )

            # -----------------------------------------
            # RISK
            # -----------------------------------------

            final_risk = float(
                realguard.compute_risk(
                    raw_probability,
                    transaction_data
                )
            )

            risk_score = int(
                round(final_risk * 100)
            )

            risk_level = realguard.risk_label(
                final_risk
            )

            fraud_threshold = float(
                realguard.FRAUD_THRESHOLD
            )

            is_fraud = (
                final_risk >= fraud_threshold
            )

            # -----------------------------------------
            # REASONS
            # -----------------------------------------

            reasons = realguard.build_reasons(
                transaction_data
            )

            # -----------------------------------------
            # SHAP
            # -----------------------------------------

            shap_data = realguard.get_shap(X)

            # -----------------------------------------
            # STORE
            # -----------------------------------------

            try:

                import uuid
                from datetime import datetime

                txn_id = (
                    "TXN-"
                    + str(uuid.uuid4())[:8].upper()
                )

                realguard.insert_transaction(
                    {
                        "txn_id": txn_id,

                        "created_at":
                            datetime.utcnow().isoformat(),

                        "amount":
                            float(amount),

                        "avg_amount":
                            float(avg_amount),

                        "merchant":
                            merchant,

                        "txn_type":
                            txn_type,

                        "txn_hour":
                            int(hour),

                        "day_of_week":
                            int(day_of_week),

                        "unusual_loc":
                            int(unusual_location),

                        "unusual_amt":
                            int(unusual_amount),

                        "new_device":
                            int(new_device),

                        "failed_attempts":
                            int(failed_attempts),

                        "fraud_prob":
                            round(final_risk, 4),

                        "risk_score":
                            int(risk_score),

                        "risk_level":
                            risk_level,

                        "is_fraud":
                            int(is_fraud)
                    }
                )

            except Exception:
                # Do not stop the prediction UI if DB persistence
                # encounters an issue.
                txn_id = "Not stored"

            # -----------------------------------------
            # RESULT
            # -----------------------------------------

            if is_fraud:

                st.markdown(
                    f"""
                    <div class="risk-card fraud-card">
                        <div class="risk-number">
                            🚨 FRAUD DETECTED
                        </div>
                        <div class="risk-label">
                            High-risk transaction
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-card legit-card">
                        <div class="risk-number">
                            ✅ LEGITIMATE
                        </div>
                        <div class="risk-label">
                            Transaction appears legitimate
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -----------------------------------------
            # MAIN METRICS
            # -----------------------------------------

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Fraud Probability",
                    f"{final_risk * 100:.2f}%"
                )

            with m2:

                st.metric(
                    "Risk Score",
                    f"{risk_score}/100"
                )

            with m3:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            # -----------------------------------------
            # RAW MODEL PROBABILITY
            # -----------------------------------------

            st.caption(
                f"Raw XGBoost fraud probability: "
                f"{raw_probability * 100:.2f}%"
            )

            # -----------------------------------------
            # REASONS
            # -----------------------------------------

            st.markdown(
                "### 🚨 Risk Factors"
            )

            if reasons:

                for reason in reasons:

                    st.markdown(
                        f"""
                        <div class="reason-card">
                            ⚠️ {reason}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.success(
                    "No major rule-based risk factors detected."
                )

            # -----------------------------------------
            # SHAP
            # -----------------------------------------

            st.markdown(
                "### 🧠 SHAP Explanation"
            )

            shap_col1, shap_col2 = st.columns(2)

            with shap_col1:

                st.markdown(
                    "#### 🔴 Increasing Fraud Risk"
                )

                increasing = shap_data.get(
                    "increasing",
                    []
                )

                if increasing:

                    for item in increasing:

                        feature = item.get(
                            "feature",
                            "Unknown"
                        )

                        value = safe_float(
                            item.get(
                                "shap_value",
                                0
                            )
                        )

                        st.markdown(
                            f"""
                            <div class="shap-positive">
                                <b>{feature}</b><br>
                                SHAP contribution:
                                +{abs(value):.4f}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No positive SHAP contributors."
                    )

            with shap_col2:

                st.markdown(
                    "#### 🟢 Decreasing Fraud Risk"
                )

                decreasing = shap_data.get(
                    "decreasing",
                    []
                )

                if decreasing:

                    for item in decreasing:

                        feature = item.get(
                            "feature",
                            "Unknown"
                        )

                        value = safe_float(
                            item.get(
                                "shap_value",
                                0
                            )
                        )

                        st.markdown(
                            f"""
                            <div class="shap-negative">
                                <b>{feature}</b><br>
                                SHAP contribution:
                                {value:.4f}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No negative SHAP contributors."
                    )

            # -----------------------------------------
            # TRANSACTION SUMMARY
            # -----------------------------------------

            st.markdown(
                "### 📋 Transaction Summary"
            )

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:

                st.metric(
                    "Amount",
                    f"₹{amount:,.2f}"
                )

                st.metric(
                    "Average Amount",
                    f"₹{avg_amount:,.2f}"
                )

                st.metric(
                    "Amount Ratio",
                    f"{amount / max(avg_amount, 1):.2f}x"
                )

            with summary_col2:

                st.metric(
                    "Merchant",
                    merchant
                )

                st.metric(
                    "Transaction Type",
                    txn_type
                )

                st.metric(
                    "Failed Attempts",
                    int(failed_attempts)
                )

            with summary_col3:

                st.metric(
                    "Hour",
                    int(hour)
                )

                st.metric(
                    "IP Frequency",
                    int(ip_frequency)
                )

                st.metric(
                    "New Device",
                    "Yes" if new_device else "No"
                )

        except Exception as error:

            st.error(
                f"Unable to analyse transaction: {error}"
            )


# ============================================================
# HISTORY TAB
# ============================================================

with history_tab:

    st.markdown(
        '<div class="section-title">📋 Transaction History</div>',
        unsafe_allow_html=True
    )

    try:

        history_data = realguard.fetch_history(
            limit=100,
            fraud_only=False,
            legit_only=False,
            search=""
        )

        history_df = normalize_records(
            history_data
        )

        history_df = clean_dataframe_for_streamlit(
            history_df
        )

        if history_df.empty:

            st.info(
                "No transaction history available."
            )

        else:

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            f"Unable to load history: {error}"
        )


# ============================================================
# ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.markdown(
        '<div class="section-title">📊 Analytics</div>',
        unsafe_allow_html=True
    )

    try:

        # ----------------------------------------------------
        # Get backend analytics
        # ----------------------------------------------------

        backend_analytics = None

        try:

            backend_analytics = (
                realguard.fetch_analytics()
            )

        except Exception:

            backend_analytics = None

        # ----------------------------------------------------
        # Safely extract scalar metrics
        # ----------------------------------------------------

        total = safe_int(
            scalar_value(
                backend_analytics,
                [
                    "total",
                    "total_transactions",
                    "count"
                ],
                0
            )
        )

        fraud = safe_int(
            scalar_value(
                backend_analytics,
                [
                    "fraud",
                    "fraud_count",
                    "fraud_transactions"
                ],
                0
            )
        )

        legit = safe_int(
            scalar_value(
                backend_analytics,
                [
                    "legit",
                    "legitimate",
                    "legit_count",
                    "legitimate_transactions"
                ],
                0
            )
        )

        avg_amount = safe_float(
            scalar_value(
                backend_analytics,
                [
                    "avg_amount",
                    "average_amount",
                    "mean_amount"
                ],
                0
            )
        )

        avg_risk = safe_float(
            scalar_value(
                backend_analytics,
                [
                    "avg_risk",
                    "average_risk",
                    "mean_risk"
                ],
                0
            )
        )

        fraud_rate = safe_float(
            scalar_value(
                backend_analytics,
                [
                    "fraud_rate",
                    "fraud_percentage",
                    "fraud_percent"
                ],
                0
            )
        )

        # ----------------------------------------------------
        # If backend scalar values are missing, calculate
        # everything directly from history.
        # ----------------------------------------------------

        if total == 0:

            history_data = realguard.fetch_history(
                limit=500,
                fraud_only=False,
                legit_only=False,
                search=""
            )

            calculated = extract_history_analytics(
                history_data
            )

            total = calculated["total"]
            fraud = calculated["fraud"]
            legit = calculated["legit"]
            avg_amount = calculated["avg_amount"]
            avg_risk = calculated["avg_risk"]
            fraud_rate = calculated["fraud_rate"]

        else:

            # If legitimate count is absent, derive it.
            if legit == 0 and total >= fraud:

                legit = total - fraud

            # If fraud rate is absent/invalid, derive it.
            if fraud_rate == 0 and total > 0:

                fraud_rate = (
                    fraud / total
                ) * 100

        # ----------------------------------------------------
        # Normalize values
        # ----------------------------------------------------

        total = int(total)

        fraud = int(fraud)

        legit = int(legit)

        avg_amount = float(avg_amount)

        avg_risk = float(avg_risk)

        fraud_rate = float(fraud_rate)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Total",
                total
            )

        with c2:

            st.metric(
                "Fraud",
                fraud
            )

        with c3:

            st.metric(
                "Legit",
                legit
            )

        with c4:

            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%"
            )

        c5, c6 = st.columns(2)

        with c5:

            st.metric(
                "Avg Amount",
                f"₹{avg_amount:,.2f}"
            )

        with c6:

            st.metric(
                "Avg Risk",
                f"{avg_risk:.2f}"
            )

        # ----------------------------------------------------
        # Transaction distribution
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Transaction Distribution"
        )

        distribution_df = pd.DataFrame(
            {
                "Type": [
                    "Fraud",
                    "Legitimate"
                ],

                "Transactions": [
                    int(fraud),
                    int(legit)
                ]
            }
        )

        st.bar_chart(
            distribution_df.set_index("Type")
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT pass backend ROC/PR lists to st.metric().
        # They are arrays used for charts only.
        # ----------------------------------------------------

        st.success(
            "Analytics loaded successfully."
        )

    except Exception as error:

        st.error(
            f"Unable to load analytics: {error}"
        )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with model_tab:

    st.markdown(
        '<div class="section-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    try:

        metrics = realguard.model_metrics

        if metrics is None:

            st.warning(
                "Model metrics are not available."
            )

        else:

            # ------------------------------------------------
            # XGBOOST
            # ------------------------------------------------

            xgb_metrics = metrics.get(
                "xgboost",
                {}
            )

            st.markdown(
                "### XGBoost"
            )

            x1, x2, x3 = st.columns(3)

            with x1:

                st.metric(
                    "Accuracy",
                    f"{safe_float(xgb_metrics.get('accuracy')) * 100:.2f}%"
                )

                st.metric(
                    "Precision",
                    f"{safe_float(xgb_metrics.get('precision')) * 100:.2f}%"
                )

            with x2:

                st.metric(
                    "Recall",
                    f"{safe_float(xgb_metrics.get('recall')) * 100:.2f}%"
                )

                st.metric(
                    "F1 Score",
                    f"{safe_float(xgb_metrics.get('f1')) * 100:.2f}%"
                )

            with x3:

                st.metric(
                    "ROC-AUC",
                    f"{safe_float(xgb_metrics.get('roc_auc')):.4f}"
                )

                st.metric(
                    "PR-AUC",
                    f"{safe_float(xgb_metrics.get('pr_auc')):.4f}"
                )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            confusion_matrix = (
                xgb_metrics.get(
                    "confusion_matrix"
                )
            )

            if (
                isinstance(
                    confusion_matrix,
                    list
                )
                and len(confusion_matrix) == 2
            ):

                st.markdown(
                    "### Confusion Matrix"
                )

                cm_df = pd.DataFrame(
                    confusion_matrix,
                    index=[
                        "Actual Legit",
                        "Actual Fraud"
                    ],
                    columns=[
                        "Predicted Legit",
                        "Predicted Fraud"
                    ]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

            # ------------------------------------------------
            # OTHER MODELS
            # ------------------------------------------------

            st.markdown(
                "### Model Comparison"
            )

            comparison_rows = []

            for model_name, values in metrics.items():

                if not isinstance(values, dict):
                    continue

                if "accuracy" not in values:
                    continue

                comparison_rows.append(
                    {
                        "Model":
                            model_name.replace(
                                "_",
                                " "
                            ).title(),

                        "Accuracy":
                            round(
                                safe_float(
                                    values.get(
                                        "accuracy"
                                    )
                                ) * 100,
                                2
                            ),

                        "Precision":
                            round(
                                safe_float(
                                    values.get(
                                        "precision"
                                    )
                                ) * 100,
                                2
                            ),

                        "Recall":
                            round(
                                safe_float(
                                    values.get(
                                        "recall"
                                    )
                                ) * 100,
                                2
                            ),

                        "F1":
                            round(
                                safe_float(
                                    values.get(
                                        "f1"
                                    )
                                ) * 100,
                                2
                            ),

                        "ROC-AUC":
                            round(
                                safe_float(
                                    values.get(
                                        "roc_auc"
                                    )
                                ),
                                4
                            ),

                        "PR-AUC":
                            round(
                                safe_float(
                                    values.get(
                                        "pr_auc"
                                    )
                                ),
                                4
                            )
                    }
                )

            if comparison_rows:

                comparison_df = pd.DataFrame(
                    comparison_rows
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # FEATURE NAMES
            # ------------------------------------------------

            feature_names = getattr(
                realguard,
                "feature_names",
                None
            )

            if feature_names:

                st.markdown(
                    "### Feature Names"
                )

                feature_df = pd.DataFrame(
                    {
                        "Feature":
                            list(feature_names)
                    }
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:

        st.error(
            f"Unable to load model performance: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#888; padding:15px;">
        REALGUARD — Explainable AI for UPI Fraud Detection |
        XGBoost + SHAP
    </div>
    """,
    unsafe_allow_html=True
)