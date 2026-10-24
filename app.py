
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.features.engineering import engineer_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "v2_feature_model.joblib"
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"

THRESHOLD = 0.984

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]

st.set_page_config(
    page_title="Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Explainable Fraud Risk Intelligence")
st.caption(
    "Explainable fraud scoring | V2 model | "
    "Decision-support prototype"
)


# --------------------------------------------------
# 2. Load model and prepare SHAP explainer
# --------------------------------------------------
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load(MODEL_PATH)

    # Use validation data only to establish the SHAP background.
    validation = pd.read_csv(VALIDATION_PATH)

    background = validation.sample(
        n=min(100, len(validation)),
        random_state=42,
    )

    background = engineer_features(background)
    background = background[FEATURES]

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_background = preprocessor.transform(background)

    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()

    feature_names = preprocessor.get_feature_names_out()

    transformed_background = pd.DataFrame(
        transformed_background,
        columns=feature_names,
    )

    explainer = shap.LinearExplainer(
        classifier,
        transformed_background,
    )

    return model, preprocessor, explainer, list(feature_names)


# --------------------------------------------------
# 3. Sidebar transaction input
# --------------------------------------------------
st.sidebar.header("Transaction Details")
st.sidebar.caption("Enter the information available for scoring.")

transaction_type = st.sidebar.selectbox(
    "Transaction type",
    ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
)

amount = st.sidebar.number_input(
    "Transaction amount",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
)

sender_balance = st.sidebar.number_input(
    "Sender's old balance",
    min_value=0.0,
    value=30000.0,
    step=1000.0,
)

receiver_balance = st.sidebar.number_input(
    "Receiver's old balance",
    min_value=0.0,
    value=0.0,
    step=1000.0,
)

score_button = st.sidebar.button(
    "Analyze Transaction",
    type="primary",
    width="stretch",
)


# --------------------------------------------------
# 4. Score transaction and explain the result
# --------------------------------------------------
if score_button:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not VALIDATION_PATH.exists():
        st.error(f"Validation file not found: {VALIDATION_PATH}")
        st.stop()

    try:
        model, preprocessor, explainer, feature_names = (
            load_model_and_explainer()
        )

        transaction = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": sender_balance,
            "oldbalanceDest": receiver_balance,
        }])

        transaction = engineer_features(transaction)
        transaction = transaction[FEATURES]

        risk_score = float(model.predict_proba(transaction)[0, 1])
        is_alert = risk_score >= THRESHOLD

        transformed = preprocessor.transform(transaction)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = pd.DataFrame(
            transformed,
            columns=feature_names,
        )

        values = np.asarray(explainer.shap_values(transformed))

        if values.ndim == 3:
            values = values[:, :, -1]

        contributions = values.reshape(-1)

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP contribution": contributions,
        })

        explanation["Direction"] = np.where(
            explanation["SHAP contribution"] > 0,
            "Raises model score",
            np.where(
                explanation["SHAP contribution"] < 0,
                "Lowers model score",
                "No contribution",
            ),
        )

        explanation["Magnitude"] = explanation[
            "SHAP contribution"
        ].abs()


# Convert technical feature names into business-friendly labels.
        FEATURE_LABELS = {
         "categorical__type_CASH_IN": "Transaction type: CASH_IN",
         "categorical__type_CASH_OUT": "Transaction type: CASH_OUT",
         "categorical__type_DEBIT": "Transaction type: DEBIT",
            "categorical__type_PAYMENT": "Transaction type: PAYMENT",
            "categorical__type_TRANSFER": "Transaction type: TRANSFER",
         "numeric__amount": "Transaction amount",
         "numeric__oldbalanceOrg": "Sender's previous balance",
         "numeric__oldbalanceDest": "Receiver's previous balance",
            "numeric__log_amount": "Log of transaction amount",
            "numeric__sender_balance_utilization": "Amount relative to sender balance",
            "numeric__receiver_balance_relative_amount": (
              "Amount relative to receiver balance"
            ),
            "numeric__sender_balance_insufficient": (
              "Amount exceeds sender's previous balance"
            ),
        }

        explanation["Feature"] = explanation["Feature"].map(
            lambda name: FEATURE_LABELS.get(name, name)
        )


        explanation = explanation.sort_values(
            "Magnitude",
            ascending=False,
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
             "Model fraud score",
             f"{risk_score:.8f}",
             help="Raw model output for class 1 (fraud); not a calibrated real-world probability.",)
        st.caption(f"Raw score as percentage: {risk_score * 100:.6f}%")
        col2.metric("Selected threshold", f"{THRESHOLD:.3f}")
        col3.metric(
            "Alert status",
            "REVIEW" if is_alert else "BELOW THRESHOLD",
        )

        st.progress(risk_score)

        if is_alert:
            st.warning(
                "This transaction crossed the selected threshold. "
                "Review it using additional evidence."
            )
        else:
            st.info(
                "This transaction is below the selected threshold. "
                "That does not guarantee the transaction is legitimate."
            )

        st.subheader("Transaction Summary")

        summary = pd.DataFrame([{
            "Transaction type": transaction_type,
            "Amount": amount,
            "Sender old balance": sender_balance,
            "Receiver old balance": receiver_balance,
            "Amount exceeds sender balance": amount > sender_balance,
        }])

        st.dataframe(summary, width="stretch", hide_index=True)

        st.subheader("Why did the model produce this score?")

        st.caption(
            "SHAP contributions are in log-odds space, not percentage points. "
            "Positive values push the model output upward; negative values "
            "push it downward relative to the background reference."
        )

        display_explanation = explanation[
            ["Feature", "SHAP contribution", "Direction"]
        ].copy()


        display_explanation["SHAP contribution"] = (
            display_explanation["SHAP contribution"].round(5)
        )

        display_explanation = display_explanation.rename(
         columns={
                "Feature": "Risk factor",
                "SHAP contribution": "Contribution (log-odds)",
                "Direction": "Effect on model output",
         }
        )


        st.dataframe(
            display_explanation,
            width="stretch",
            hide_index=True,
        )

        st.subheader("Contribution chart")


        chart_data = explanation.set_index("Feature")[
           "SHAP contribution"
        ].sort_values()

        st.caption(
          "Negative values lower the model output; positive values raise it. "
           "Longer bars indicate larger contributions in log-odds space."
        )

        st.bar_chart(chart_data, horizontal=True)


        csv_bytes = explanation.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download explanation CSV",
            data=csv_bytes,
            file_name="transaction_explanation.csv",
            mime="text/csv",
        )

        st.info(
            "Important: this is a decision-support prototype. "
            "The score is not guaranteed to be a calibrated fraud probability. "
            "Synthetic-data patterns may not generalize to real transactions."
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)

else:
    st.info(
        "Enter transaction details in the sidebar and click "
        "'Analyze Transaction' to view the score and explanation."
    )

    st.subheader("What this dashboard provides")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        "**Risk scoring**\n\n"
        "Scores a transaction with the saved V2 model."
    )

    col2.markdown(
        "**Threshold-based alert**\n\n"
        "Compares the score with the selected threshold."
    )

    col3.markdown(
        "**Explainability**\n\n"
        "Shows features pushing the model score up or down."
    )
