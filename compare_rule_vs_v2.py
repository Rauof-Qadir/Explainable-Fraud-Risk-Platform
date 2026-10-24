
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.engineering import engineer_features

DATA = Path("data/processed/splits/validation.csv")
MODEL = Path("models/v2_feature_model.joblib")
OUTPUT = Path("reports/rule_vs_v2_comparison.csv")

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


def evaluate(name, y, predicted, scores=None):
    y = y.to_numpy(dtype=int)
    predicted = pd.Series(predicted).to_numpy(dtype=bool)

    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    alerts = tp + fp

    result = {
        "method": name,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "alert_rate": alerts / len(y),
    }

    if scores is not None:
        result["average_precision"] = average_precision_score(y, scores)
        result["roc_auc"] = roc_auc_score(y, scores)

    return result


def main():
    print("Loading validation split only...")
    df = pd.read_csv(DATA)
    df = engineer_features(df)

    y = df["isFraud"].astype(int)

    model = joblib.load(MODEL)
    scores = model.predict_proba(df[FEATURES])[:, 1]

    # Simple rule A: only TRANSFER transactions with zero receiver balance.
    rule_a = (
        df["type"].eq("TRANSFER")
        & df["oldbalanceDest"].eq(0)
    )

    # Simple rule B: zero receiver balance for TRANSFER or CASH_OUT.
    rule_b = (
        df["type"].isin(["TRANSFER", "CASH_OUT"])
        & df["oldbalanceDest"].eq(0)
    )

    results = [
        evaluate("rule_transfer_receiver_zero", y, rule_a),
        evaluate("rule_transfer_or_cashout_receiver_zero", y, rule_b),
    ]

    for threshold in [0.87, 0.984]:
        results.append(
            evaluate(
                f"v2_threshold_{threshold}",
                y,
                scores >= threshold,
                scores=scores,
            )
        )

    report = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("\n--- RULE VS V2 ---")
    print(report.to_string(index=False))
    print("\nSaved:", OUTPUT.resolve())
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
