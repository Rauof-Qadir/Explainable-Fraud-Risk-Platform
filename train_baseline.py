
from pathlib import Path
import gc
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_FILE = Path("data/processed/splits/train.csv")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")
MODEL_FILE = Path("models/baseline_model.joblib")
REPORT_FILE = Path("reports/baseline_model_report.json")

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "isFraud"

THRESHOLD = 0.5

# Step 1: Training data load karo.
print("Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype("int8")

print("Training rows:", len(train_df))
print("Training frauds:", int(y_train.sum()))
print("Training fraud rate (%):", round(100 * y_train.mean(), 6))

# Step 2: Features ko model ke liye prepare karo.
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
            NUMERIC_FEATURES,
        ),
    ],
)

# SGDClassifier with log_loss ek scalable logistic model hai.
# class_weight balanced rare fraud class ko training mein
# zyada importance deta hai.
classifier = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    random_state=42,
    max_iter=20,
    tol=1e-3,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

# Step 3: Model train karo.
print("\nTraining baseline logistic model...")
model.fit(X_train, y_train)
print("Training complete.")

del train_df, X_train, y_train
gc.collect()

# Step 4: Validation data par evaluate karo.
print("\nLoading validation data...")
validation_df = pd.read_csv(VALIDATION_FILE)

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET].astype("int8")

print("Validation rows:", len(validation_df))
print("Validation frauds:", int(y_val.sum()))

print("\nScoring validation transactions...")
scores = model.predict_proba(X_val)[:, 1]
predictions = (scores >= THRESHOLD).astype("int8")

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

report = {
    "model": "SGDClassifier logistic loss",
    "class_weight": "balanced",
    "threshold": THRESHOLD,
    "features": FEATURES,
    "training_rows": 4_433_703,
    "validation_rows": int(len(y_val)),
    "validation_frauds": int(y_val.sum()),
    "metrics": {
        "precision": float(precision_score(
            y_val, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_val, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_val, predictions, zero_division=0
        )),
        "pr_auc_average_precision": float(
            average_precision_score(y_val, scores)
        ),
        "roc_auc": float(roc_auc_score(y_val, scores)),
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
}

# Step 5: Model aur report save karo.
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- BASELINE MODEL REPORT ---")
print(json.dumps(report, indent=2))
print("\nModel saved to:", MODEL_FILE.resolve())
print("Report saved to:", REPORT_FILE.resolve())
