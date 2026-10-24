
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.engineering import engineer_features


TRAIN_FILE = Path("data/processed/splits/train.csv")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")

CSV_REPORT = Path("reports/v2_feature_signal_audit.csv")
JSON_REPORT = Path("reports/v2_feature_signal_audit.json")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]


def load_and_engineer(path):
    df = pd.read_csv(path)
    featured = engineer_features(df)

    if "isFraud" not in featured.columns:
        raise ValueError(f"Missing isFraud in {path}")

    return featured


def describe_split(name, df):
    y = df["isFraud"].astype(int)

    print(f"\n--- {name} ---")
    print(f"Rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")

    print("\nFraud rate by transaction type:")
    by_type = (
        df.groupby("type", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
    )
    print(by_type.to_string())


def audit_features(name, df):
    y = df["isFraud"].astype(int).to_numpy()
    rows = []

    for feature in FEATURES:
        values = df[feature].astype(float)

        nonfraud = values[y == 0]
        fraud = values[y == 1]

        # A single feature is only a diagnostic, not a model.
        if values.nunique(dropna=True) > 1:
            try:
                ap = float(average_precision_score(y, values))
                auc = float(roc_auc_score(y, values))
                # AUC below 0.5 can indicate inverse direction.
            except ValueError:
                ap = None
                auc = None
        else:
            ap = None
            auc = None

        rows.append({
            "split": name,
            "feature": feature,
            "missing_values": int(values.isna().sum()),
            "unique_values": int(values.nunique(dropna=True)),
            "nonfraud_median": float(nonfraud.median()),
            "fraud_median": float(fraud.median()) if len(fraud) else None,
            "nonfraud_mean": float(nonfraud.mean()),
            "fraud_mean": float(fraud.mean()) if len(fraud) else None,
            "single_feature_average_precision": ap,
            "single_feature_roc_auc": auc,
        })

    return rows


def main():
    print("Loading training and validation splits only...")
    train = load_and_engineer(TRAIN_FILE)
    validation = load_and_engineer(VALIDATION_FILE)

    # Basic target sanity checks.
    for name, df in [("train", train), ("validation", validation)]:
        if not set(df["isFraud"].unique()).issubset({0, 1}):
            raise ValueError(f"Invalid target labels in {name}")

        if df[FEATURES].isna().any().any():
            raise ValueError(f"Missing feature values in {name}")

    describe_split("TRAIN", train)
    describe_split("VALIDATION", validation)

    rows = []
    rows.extend(audit_features("train", train))
    rows.extend(audit_features("validation", validation))

    report_df = pd.DataFrame(rows)
    CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(CSV_REPORT, index=False)

    summary = {
        "data_used": ["training split", "validation split"],
        "test_split_loaded": False,
        "train_rows": int(len(train)),
        "train_fraud_cases": int(train["isFraud"].sum()),
        "validation_rows": int(len(validation)),
        "validation_fraud_cases": int(validation["isFraud"].sum()),
        "features_audited": FEATURES,
        "feature_metrics": rows,
        "interpretation_notes": [
            "Single-feature metrics are diagnostic, not causal evidence.",
            "High predictive power alone does not prove leakage.",
            "Check feature availability at decision time.",
            "Investigate suspiciously strong features before production.",
            "No test data was loaded.",
        ],
    }

    JSON_REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n--- FEATURE SIGNAL AUDIT ---")
    print(
        report_df[
            [
                "split",
                "feature",
                "nonfraud_median",
                "fraud_median",
                "single_feature_average_precision",
                "single_feature_roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nCSV report:", CSV_REPORT.resolve())
    print("JSON report:", JSON_REPORT.resolve())
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
