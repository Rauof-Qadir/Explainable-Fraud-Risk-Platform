
from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.features.engineering import engineer_features


# =========================================================
# 1. CONFIGURATION
# =========================================================
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

FEATURE_LABELS = {
    "categorical__type_CASH_IN": "Transaction type: CASH_IN",
    "categorical__type_CASH_OUT": "Transaction type: CASH_OUT",
    "categorical__type_DEBIT": "Transaction type: DEBIT",
    "categorical__type_PAYMENT": "Transaction type: PAYMENT",
    "categorical__type_TRANSFER": "Transaction type: TRANSFER",
    "numeric__amount": "Transaction amount",
    "numeric__oldbalanceOrg": "Sender's previous balance",
    "numeric__oldbalanceDest": "Receiver's previous balance",
    "numeric__log_amount": "Log transaction amount",
    "numeric__sender_balance_utilization": "Amount relative to sender balance",
    "numeric__receiver_balance_relative_amount": (
        "Amount relative to receiver balance"
    ),
    "numeric__sender_balance_insufficient": (
        "Amount exceeds sender's previous balance"
    ),
}

st.set_page_config(
    page_title="Explainable Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. FUTURISTIC UI THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --bg: #070B14;
        --panel: #101827;
        --panel-light: #151F32;
        --cyan: #22D3EE;
        --violet: #A78BFA;
        --text: #F1F5F9;
        --muted: #94A3B8;
        --border: rgba(148, 163, 184, 0.16);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(34,211,238,.10), transparent 34%),
            radial-gradient(ellipse at 94% 10%, rgba(139,92,246,.11), transparent 32%),
            #070B14;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 11, 20, 0.78);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1524 0%, #090E19 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    p, label, .stMarkdown {
        color: #CBD5E1;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #131E30, #0D1523);
        border: 1px solid var(--border);
        padding: 18px 20px;
        border-radius: 15px;
        transition: border-color .2s ease, transform .2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(34,211,238,.55);
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-family: 'Space Mono', monospace;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(
            115deg,
            rgba(17, 35, 54, .97),
            rgba(22, 20, 48, .96)
        );
        border: 1px solid rgba(34,211,238,.28);
        border-radius: 22px;
        padding: 30px 32px;
        margin-bottom: 24px;
        box-shadow: 0 0 35px rgba(34,211,238,.055);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -100px;
        top: -145px;
        border-radius: 50%;
        border: 1px solid rgba(34,211,238,.22);
        box-shadow: 0 0 0 25px rgba(34,211,238,.025),
                    0 0 0 55px rgba(167,139,250,.025);
        pointer-events: none;
    }

    .eyebrow {
        color: #67E8F9;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #F8FAFC;
        font-size: clamp(27px, 3.5vw, 43px);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1.8px;
        margin: 0;
    }

    .hero-subtitle {
        color: #AAB8CB;
        font-size: 14px;
        margin-top: 12px;
        max-width: 700px;
        line-height: 1.8;
    }

    .system-pill {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 11px;
        border: 1px solid rgba(34,211,238,.28);
        border-radius: 999px;
        color: #67E8F9;
        background: rgba(34,211,238,.07);
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
    }

    .section-kicker {
        color: #67E8F9;
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .panel {
        background: linear-gradient(145deg, #121D2E, #0C1422);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 21px;
        height: 100%;
    }

    .panel-title {
        color: #F1F5F9;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .panel-subtitle {
        color: #94A3B8;
        font-size: 12px;
        line-height: 1.7;
    }

    .score-number {
        font-family: 'Space Mono', monospace;
        font-size: clamp(25px, 3vw, 39px);
        font-weight: 700;
        letter-spacing: -2px;
        margin: 10px 0;
        overflow-wrap: anywhere;
    }

    .score-caption {
        color: #94A3B8;
        font-size: 11px;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 1px;
    }

    .status-alert {
        color: #FDA4AF;
        border: 1px solid rgba(251,113,133,.4);
        background: rgba(244,63,94,.10);
    }

    .status-normal {
        color: #67E8F9;
        border: 1px solid rgba(34,211,238,.35);
        background: rgba(34,211,238,.08);
    }

    .mini-label {
        color: #94A3B8;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .feature-positive {
        color: #FDA4AF;
        font-weight: 600;
    }

    .feature-negative {
        color: #67E8F9;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(100deg, #0891B2, #6366F1);
        color: white;
        border: 1px solid rgba(103,232,249,.38);
        border-radius: 11px;
        min-height: 46px;
        font-weight: 700;
        letter-spacing: .2px;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        border-color: #67E8F9;
        box-shadow: 0 0 22px rgba(34,211,238,.22);
        transform: translateY(-1px);
        color: white;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(34,211,238,.4);
        background: rgba(34,211,238,.07);
        color: #A5F3FC;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #111B2B;
        border-color: #2A3B51;
        border-radius: 9px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #06B6D4, #8B5CF6);
    }

    .footer {
        color: #64748B;
        font-size: 11px;
        line-height: 1.8;
        text-align: center;
        padding: 24px 0 8px;
        border-top: 1px solid var(--border);
        margin-top: 28px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero { padding: 22px 20px; }
        .block-container { padding-top: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. REUSABLE UI HELPERS
# =========================================================
def section_heading(kicker, title, subtitle=None):
    st.markdown(
        f"""
        <div style="margin: 26px 0 15px;">
            <div class="section-kicker">{html.escape(kicker)}</div>
            <div style="font-size:23px;font-weight:800;color:#F1F5F9;
                        letter-spacing:-.7px;">
                {html.escape(title)}
            </div>
            {
                f'<div class="panel-subtitle" style="margin-top:5px;">{html.escape(subtitle)}</div>'
                if subtitle else ''
            }
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(value):
    return f"{value:,.2f}"


def render_score_panel(score, threshold, is_alert):
    status_class = "status-alert" if is_alert else "status-normal"
    status_text = "REVIEW REQUIRED" if is_alert else "BELOW THRESHOLD"
    score_color = "#FDA4AF" if is_alert else "#67E8F9"

    st.markdown(
        f"""
        <div class="panel">
            <div class="mini-label">MODEL OUTPUT / CLASS 1</div>
            <div class="score-number" style="color:{score_color};">
                {score:.8f}
            </div>
            <div class="score-caption">
                Raw model score — not a calibrated real-world probability.
            </div>
            <div style="margin-top:18px;">
                <span class="status-pill {status_class}">
                    {status_text}
                </span>
            </div>
            <div style="margin-top:18px;" class="mini-label">
                SELECTED THRESHOLD: {threshold:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 4. HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">FINTECH / RISK ANALYTICS / AI</div>
        <div class="hero-title">
            Explainable FRAUD RISK<br>INTELLIGENCE
        </div>
        <div class="hero-subtitle">
            An explainable transaction risk console combining machine
            learning, decision thresholds and SHAP-powered insights.
        </div>
        <div class="system-pill">
            ● V2 FEATURE MODEL &nbsp; / &nbsp; EXPLAINABILITY ENABLED
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. LOAD MODEL + SHAP EXPLAINER
# =========================================================
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    if validation.empty:
        raise ValueError("Validation dataset is empty.")

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

    feature_names = list(preprocessor.get_feature_names_out())

    transformed_background = pd.DataFrame(
        transformed_background,
        columns=feature_names,
    )

    explainer = shap.LinearExplainer(
        classifier,
        transformed_background,
    )

    return model, preprocessor, explainer, feature_names


# =========================================================
# 6. SIDEBAR / TRANSACTION CONTROLS
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:5px 0 20px;">
            <div style="font-size:11px;color:#67E8F9;
                        letter-spacing:2px;font-weight:700;">
                CONTROL CENTER
            </div>
            <div style="font-size:23px;font-weight:800;color:#F8FAFC;
                        margin-top:8px;">
                Transaction Lab
            </div>
            <div style="font-size:12px;color:#94A3B8;
                        line-height:1.7;margin-top:6px;">
                Configure a transaction and inspect its model output.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Transaction parameters")

    transaction_type = st.selectbox(
        "Transaction type",
        ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
    )

    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
    )

    sender_balance = st.number_input(
        "Sender's old balance",
        min_value=0.0,
        value=30000.0,
        step=1000.0,
        format="%.2f",
    )

    receiver_balance = st.number_input(
        "Receiver's old balance",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        format="%.2f",
    )

    st.markdown("---")

    score_button = st.button(
        "⚡  ANALYZE TRANSACTION",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="margin-top:22px;padding:14px;
                    background:rgba(34,211,238,.045);
                    border:1px solid rgba(34,211,238,.15);
                    border-radius:12px;">
            <div class="mini-label">MODEL CONFIGURATION</div>
            <div style="font-size:12px;color:#CBD5E1;line-height:2;
                        margin-top:8px;">
                Model: V2 feature model<br>
                Threshold: 0.984<br>
                Explainer: SHAP LinearExplainer<br>
                Mode: Decision support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. ANALYZE TRANSACTION
# =========================================================
if score_button:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not VALIDATION_PATH.exists():
        st.error(f"Validation file not found: {VALIDATION_PATH}")
        st.stop()

    if not np.isfinite([amount, sender_balance, receiver_balance]).all():
        st.error("All numeric inputs must be finite numbers.")
        st.stop()

    try:
        with st.spinner("Running model inference and generating explanations..."):
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

            shap_values = np.asarray(
                explainer.shap_values(transformed)
            )

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

            contributions = shap_values.reshape(-1)

            if len(contributions) != len(feature_names):
                raise ValueError(
                    "SHAP output does not match transformed feature names."
                )

            explanation = pd.DataFrame({
                "Feature": feature_names,
                "SHAP contribution": contributions,
            })

            explanation["Direction"] = np.where(
                explanation["SHAP contribution"] > 0,
                "Raises model output",
                np.where(
                    explanation["SHAP contribution"] < 0,
                    "Lowers model output",
                    "No contribution",
                ),
            )

            explanation["Magnitude"] = (
                explanation["SHAP contribution"].abs()
            )

            explanation["Feature"] = explanation["Feature"].map(
                lambda name: FEATURE_LABELS.get(name, name)
            )

            explanation = explanation.sort_values(
                "Magnitude",
                ascending=False,
            )

        # -------------------------------------------------
        # RESULT OVERVIEW
        # -------------------------------------------------
        section_heading(
            "01 / INFERENCE",
            "Transaction intelligence overview",
            "Model output and decision status for the transaction you entered.",
        )

        left, middle, right = st.columns([1.2, 1, 1])

        with left:
            render_score_panel(risk_score, THRESHOLD, is_alert)

        with middle:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="mini-label">TRANSACTION VALUE</div>
                    <div class="score-number" style="color:#F8FAFC;
                         font-size:29px;">
                        {money(amount)}
                    </div>
                    <div class="score-caption">
                        Input amount in dataset currency units.
                        Currency is not independently verified.
                    </div>
                    <div style="margin-top:20px;" class="mini-label">
                        TRANSACTION TYPE
                    </div>
                    <div style="font-size:18px;font-weight:700;
                         color:#A78BFA;margin-top:8px;">
                        {html.escape(transaction_type)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="mini-label">DECISION ENGINE</div>
                    <div style="font-size:20px;font-weight:800;
                         color:#F8FAFC;margin-top:15px;">
                        {"Manual review" if is_alert else "Below threshold"}
                    </div>
                    <div class="score-caption" style="margin-top:10px;">
                        {"The score meets or exceeds the configured threshold."
                         if is_alert else
                         "The score is below the configured threshold."}
                    </div>
                    <div style="margin-top:22px;" class="mini-label">
                        THRESHOLD COMPARISON
                    </div>
                    <div style="font-family:'Space Mono',monospace;
                         color:#67E8F9;font-size:14px;margin-top:8px;">
                        {risk_score:.6f} / {THRESHOLD:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(
            min(max(risk_score, 0.0), 1.0),
            text="Raw model score position on a 0–1 scale (not a calibration guarantee)",
        )

        if is_alert:
            st.warning(
                "REVIEW SUGGESTED — the score crossed the configured threshold. "
                "Use additional evidence before making a decision."
            )
        else:
            st.info(
                "BELOW THRESHOLD — this transaction was not flagged by the "
                "configured rule. This does not guarantee that it is legitimate."
            )

        # -------------------------------------------------
        # TRANSACTION SUMMARY
        # -------------------------------------------------
        section_heading(
            "02 / TRANSACTION",
            "Transaction profile",
            "A structured view of the submitted input features.",
        )

        summary = pd.DataFrame([{
            "Transaction type": transaction_type,
            "Amount": amount,
            "Sender's previous balance": sender_balance,
            "Receiver's previous balance": receiver_balance,
            "Amount exceeds sender balance": amount > sender_balance,
        }])

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # SHAP EXPLANATIONS
        # -------------------------------------------------
        section_heading(
            "03 / EXPLAINABLE AI",
            "Why did the model produce this output?",
            "Feature contributions are measured relative to the SHAP background reference.",
        )

        positive_count = int(
            (explanation["SHAP contribution"] > 0).sum()
        )
        negative_count = int(
            (explanation["SHAP contribution"] < 0).sum()
        )

        a, b, c = st.columns(3)

        a.metric("Factors raising output", positive_count)
        b.metric("Factors lowering output", negative_count)
        c.metric("Factors explained", len(explanation))

        st.markdown(
            """
            <div class="panel" style="margin:10px 0 18px;">
                <div class="panel-title">Contribution guide</div>
                <div class="panel-subtitle">
                    <span style="color:#FDA4AF;font-weight:700;">
                    Positive contribution
                    </span>
                    — pushes model output upward relative to the background.<br>
                    <span style="color:#67E8F9;font-weight:700;">
                    Negative contribution
                    </span>
                    — pushes model output downward relative to the background.<br><br>
                    SHAP values are in log-odds space for this linear model,
                    not percentage points. They describe model behavior,
                    not causation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_explanation = explanation[
            ["Feature", "SHAP contribution", "Direction"]
        ].copy()

        display_explanation["SHAP contribution"] = (
            display_explanation["SHAP contribution"].round(6)
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
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # CONTRIBUTION CHART
        # -------------------------------------------------
        section_heading(
            "04 / FEATURE SIGNALS",
            "Model contribution map",
            "Features are ordered by their absolute SHAP contribution.",
        )

        chart_data = explanation.set_index("Feature")[
            "SHAP contribution"
        ].sort_values()

        st.bar_chart(
            chart_data,
            horizontal=True,
            height=440,
            color="#22D3EE",
        )

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------
        section_heading(
            "05 / EXPORT CENTER",
            "Download the explanation",
            "Export the transaction's SHAP contributions for further inspection.",
        )

        csv_bytes = explanation.to_csv(index=False).encode("utf-8")

        col_download, col_note = st.columns([1, 2])

        with col_download:
            st.download_button(
                "⬇  Download explanation CSV",
                data=csv_bytes,
                file_name="transaction_explanation.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_note:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">Export contents</div>
                    <div class="panel-subtitle">
                        Feature names, SHAP contribution values, direction
                        of contribution, and contribution magnitude.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # RESPONSIBLE USE
        # -------------------------------------------------
        st.warning(
            "Research prototype: this dashboard uses synthetic-data patterns. "
            "The raw model score is not guaranteed to be a calibrated fraud "
            "probability. Do not use this output alone to block transactions "
            "or make consequential decisions."
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        with st.expander("Technical error details"):
            st.exception(exc)


# =========================================================
# 8. LANDING STATE
# =========================================================
else:
    section_heading(
        "SYSTEM / READY",
        "Your risk analysis workspace",
        "Enter transaction information in the left control panel to begin.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:25px;">◈</div>
                <div class="panel-title">Risk scoring</div>
                <div class="panel-subtitle">
                    Evaluate a transaction using the saved V2 feature model
                    and configured decision threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:25px;">⌁</div>
                <div class="panel-title">Explainable AI</div>
                <div class="panel-subtitle">
                    Inspect feature contributions that push model output
                    upward or downward relative to the reference.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:25px;">↗</div>
                <div class="panel-title">Decision support</div>
                <div class="panel-subtitle">
                    Compare model output with the selected threshold and
                    export explanations for further analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel" style="margin-top:20px;">
            <div class="section-kicker">HOW IT WORKS</div>
            <div class="panel-title">From transaction input to explanation</div>
            <div class="panel-subtitle">
                01 &nbsp; Enter transaction details<br>
                02 &nbsp; Engineer the expected model features<br>
                03 &nbsp; Calculate the raw fraud-class score<br>
                04 &nbsp; Compare it with the selected threshold<br>
                05 &nbsp; Inspect SHAP contributions and export the results
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        FRAUD RISK INTELLIGENCE &nbsp; · &nbsp; V2 FEATURE MODEL<br>
        Explainable ML research prototype &nbsp; · &nbsp;
        Synthetic data &nbsp; · &nbsp; Human review required
    </div>
    """,
    unsafe_allow_html=True,
)
