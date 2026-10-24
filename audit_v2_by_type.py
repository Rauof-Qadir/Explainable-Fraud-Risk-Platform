
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import engineer_features

def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    return value


VALIDATION_FILE = Path("data/processed/splits/validation.csv")
MODEL_FILE = Path("models/v2_feature_model.joblib")

CSV_REPORT = Path("reports/v2_by_type_audit.csv")
JSON_REPORT = Path("reports/v2_by_type_audit.json")

BASE_FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

V2_FEATURES = BASE_FEATURES + [
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]

THRESHOLDS = [0.5, 0.87, 0.984]


def main():
    print("Loading validation data only...")
    df = pd.read_csv(VALIDATION_FILE)

    print("Loading V2 model...")
    model = joblib.load(MODEL_FILE)

    featured = engineer_features(df)
    X = featured[V2_FEATURES]
    y = featured["isFraud"].to_numpy(dtype=np.int8)

    print("Scoring validation transactions...")
    scores = model.predict_proba(X)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores.")

    scored = pd.DataFrame({
        "type": featured["type"].to_numpy(),
        "isFraud": y,
        "score": scores,
    })

    # Overall score distributions by transaction type.
    score_summary = (
        scored.groupby("type", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            actual_frauds=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            score_min=("score", "min"),
            score_median=("score", "median"),
            score_p95=("score", lambda s: s.quantile(0.95)),
            score_p99=("score", lambda s: s.quantile(0.99)),
            score_max=("score", "max"),
        )
        .reset_index()
    )

    rows = []

    for threshold in THRESHOLDS:
        scored["predicted_fraud"] = (
            scored["score"] >= threshold
        ).astype(np.int8)

        for transaction_type, group in scored.groupby(
            "type", observed=True
        ):
            actual = group["isFraud"].to_numpy(dtype=np.int8)
            pred = group["predicted_fraud"].to_numpy(dtype=np.int8)

            tp = int(((actual == 1) & (pred == 1)).sum())
            fp = int(((actual == 0) & (pred == 1)).sum())
            fn = int(((actual == 1) & (pred == 0)).sum())
            tn = int(((actual == 0) & (pred == 0)).sum())

            actual_frauds = tp + fn
            alerts = tp + fp

            rows.append({
                "threshold": threshold,
                "type": transaction_type,
                "transactions": int(len(group)),
                "actual_frauds": int(actual_frauds),
                "alerts": alerts,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": tp / alerts if alerts else 0.0,
                "recall": (
                    tp / actual_frauds if actual_frauds else None
                ),
                "alert_rate": alerts / len(group),
            })

    threshold_report = pd.DataFrame(rows)

    # Save reports.
    CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    threshold_report.to_csv(CSV_REPORT, index=False)

    report = {
        "evaluation_data": "validation only",
        "validation_rows": int(len(scored)),
        "validation_frauds": int(scored["isFraud"].sum()),
        "thresholds_checked": THRESHOLDS,
        "score_summary_by_type": score_summary.to_dict(
            orient="records"
        ),
        "threshold_metrics_by_type": threshold_report.to_dict(
            orient="records"
        ),
        "notes": [
            "This is a diagnostic breakdown, not a causal analysis.",
            "Types with zero actual fraud have undefined recall.",
            "Test data was not loaded.",
        ],
    }

    JSON_REPORT.write_text(
    json.dumps(json_safe(report), indent=2, allow_nan=False),
    encoding="utf-8",
    )
    print("\n--- SCORE DISTRIBUTION BY TYPE ---")
    print(score_summary.to_string(index=False))

    print("\n--- METRICS BY TYPE AND THRESHOLD ---")
    print(
        threshold_report[
            [
                "threshold",
                "type",
                "transactions",
                "actual_frauds",
                "alerts",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "alert_rate",
            ]
        ].to_string(index=False)
    )

    print("\nCSV report:", CSV_REPORT.resolve())
    print("JSON report:", JSON_REPORT.resolve())
    print("Test data was not loaded.")


if __name__ == "__main__":
    main()
