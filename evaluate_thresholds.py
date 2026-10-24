
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

MODEL_FILE = Path("models/baseline_model.joblib")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")
CSV_REPORT = Path("reports/threshold_comparison.csv")
JSON_REPORT = Path("reports/threshold_selection_report.json")

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
TARGET = "isFraud"

FP_COST = 5.0
FN_COST = 150.0

THRESHOLDS = (
    [round(0.90 + i * 0.001, 3) for i in range(101)]
    + [1.01]
)

print("Loading model...")
model = joblib.load(MODEL_FILE)

print("Loading validation data...")
df = pd.read_csv(VALIDATION_FILE)

X_val = df[FEATURES]
y_val = df[TARGET].astype("int8")

print("Calculating fraud scores...")
scores = model.predict_proba(X_val)[:, 1]

rows = []

for threshold in THRESHOLDS:
    predictions = (scores >= threshold).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    total_cost = fp * FP_COST + fn * FN_COST

    rows.append({
        "threshold": threshold,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "alerts": int(tp + fp),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
        "missed_fraud_cost": float(fn * FN_COST),
        "false_alert_cost": float(fp * FP_COST),
        "total_cost": float(total_cost),
    })

results = pd.DataFrame(rows)
results = results.sort_values(
    ["total_cost", "threshold"],
    ascending=[True, False],
).reset_index(drop=True)

best = results.iloc[0].to_dict()

CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(CSV_REPORT, index=False)

report = {
    "model": "SGDClassifier logistic loss",
    "evaluation_data": "validation only",
    "validation_rows": int(len(y_val)),
    "actual_frauds": int(y_val.sum()),
    "false_positive_cost": FP_COST,
    "false_negative_cost": FN_COST,
    "average_precision": float(
        average_precision_score(y_val, scores)
    ),
    "thresholds_tested": THRESHOLDS,
    "best_threshold_by_estimated_cost": best,
    "comparison_csv": str(CSV_REPORT),
}

JSON_REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- THRESHOLD COMPARISON ---")
print(
    results[
        [
            "threshold",
            "alerts",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "total_cost",
        ]
    ].sort_values("threshold").to_string(index=False)
)

print("\n--- LOWEST ESTIMATED COST ---")
print(json.dumps(best, indent=2))
print("\nCSV report:", CSV_REPORT.resolve())
print("JSON report:", JSON_REPORT.resolve())
