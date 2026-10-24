
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from src.features.engineering import engineer_features


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
TRAIN_FILE = Path("data/processed/splits/train.csv")
MODEL_FILE = Path("models/v2_feature_model.joblib")
REPORT_FILE = Path("reports/v2_training_report.json")

BASE_NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

ENGINEERED_FEATURES = [
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]

CATEGORICAL_FEATURES = ["type"]

FEATURES = (
    CATEGORICAL_FEATURES
    + BASE_NUMERIC_FEATURES
    + ENGINEERED_FEATURES
)


# --------------------------------------------------
# 2. Load TRAINING data only
# --------------------------------------------------
print("Loading training split only...")
df = pd.read_csv(TRAIN_FILE)

if df.empty:
    raise ValueError("Training dataset is empty.")

if "isFraud" not in df.columns:
    raise ValueError("Training dataset is missing isFraud.")

print(f"Training rows: {len(df):,}")
print(f"Training fraud cases: {int(df['isFraud'].sum()):,}")

# The target is kept separate from the model features.
y = df["isFraud"].astype("int8")

# Create features using the same reusable feature module.
featured_df = engineer_features(df)

X = featured_df[FEATURES]

if X.isna().any().any():
    raise ValueError("Features contain missing values.")

print(f"Features used: {len(FEATURES)}")
print("Feature columns:", FEATURES)


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            StandardScaler(),
            BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 4. Model
# --------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=42,
                max_iter=20,
                tol=1e-3,
            ),
        ),
    ]
)


# --------------------------------------------------
# 5. Train on TRAINING data only
# --------------------------------------------------
print("\nTraining V2 model...")
model.fit(X, y)

# Create output directories if needed.
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

report = {
    "model_name": "V2 engineered-feature baseline",
    "training_data": str(TRAIN_FILE),
    "training_rows": int(len(df)),
    "training_fraud_cases": int(y.sum()),
    "feature_columns": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES,
    "target": "isFraud",
    "model": "SGDClassifier with log_loss",
    "class_weight": "balanced",
    "max_iter": 20,
    "random_state": 42,
    "validation_used_for_fitting": False,
    "test_used": False,
    "note": (
        "This report documents training configuration only. "
        "Model quality must be assessed on validation data."
    ),
}

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nV2 model training complete.")
print("Saved model:", MODEL_FILE.resolve())
print("Saved training report:", REPORT_FILE.resolve())
print("Validation data was not used for fitting.")
print("Test data was not loaded.")
