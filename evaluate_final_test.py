
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.features.engineering import engineer_features


TEST_FILE = Path("data/processed/splits/test.csv")
MODEL_FILE = Path("models/v2_feature_model.joblib")
REPORT_FILE = Path("reports/final_test_evaluation.csv")

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


def calculate_metrics(y, scores, threshold):
    predicted = scores >= threshold

    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())

    alerts = tp + fp
    total_fraud = tp + fn

    return {
        "threshold": threshold,
        "transactions": len(y),
        "actual_frauds": total_fraud,
        "fraud_rate": float(y.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": alerts,
        "alert_rate": alerts / len(y),
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / total_fraud if total_fraud else 0.0,
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "roc_auc": float(roc_auc_score(y, scores)),
        "estimated_cost_fp5_fn150": fp * 5 + fn * 150,
    }


def main():
    print("Loading final holdout test data...")
    df = pd.read_csv(TEST_FILE)
    df = engineer_features(df)

    y = df["isFraud"].astype(int).to_numpy()

    print("Loading frozen V2 model...")
    model = joblib.load(MODEL_FILE)

    print("Generating predictions...")
    scores = model.predict_proba(df[FEATURES])[:, 1]

    if not (scores == scores).all():
        raise ValueError("Invalid prediction scores detected.")

    result = calculate_metrics(y, scores, THRESHOLD)

    print("\n--- FINAL TEST RESULTS ---")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n--- RESULTS BY TRANSACTION TYPE ---")
    df["score"] = scores
    df["predicted_fraud"] = (scores >= THRESHOLD).astype(int)

    rows = []

    for transaction_type, group in df.groupby(
        "type", observed=True
    ):
        actual = group["isFraud"].astype(int)
        predicted = group["predicted_fraud"].astype(int)

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        alerts = tp + fp

        rows.append({
            "type": transaction_type,
            "transactions": int(len(group)),
            "actual_frauds": int(actual.sum()),
            "alerts": alerts,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / alerts if alerts else 0.0,
            "recall": (
                tp / int(actual.sum()) if actual.sum() else None
            ),
            "alert_rate": alerts / len(group),
        })

    by_type = pd.DataFrame(rows)
    print(by_type.to_string(index=False))

    report = pd.DataFrame([result])
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_FILE, index=False)

    print("\nSaved:", REPORT_FILE.resolve())
    print("Threshold remained fixed at", THRESHOLD)
    print("Do not tune the model using these test results.")


if __name__ == "__main__":
    main()
