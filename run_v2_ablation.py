
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import engineer_features


TRAIN_FILE = Path("data/processed/splits/train.csv")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")
REPORT_FILE = Path("reports/v2_ablation_comparison.csv")

NUMERIC_BASE = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

ENGINEERED = [
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]

EXPERIMENTS = {
    "type_only": ["type"],
    "v1_baseline_features": ["type"] + NUMERIC_BASE,
    "v2_full": ["type"] + NUMERIC_BASE + ENGINEERED,
    "v2_without_receiver_ratio": [
        "type",
        *NUMERIC_BASE,
        "log_amount",
        "sender_balance_utilization",
        "sender_balance_insufficient",
    ],
}


def make_model(features):
    categorical = ["type"] if "type" in features else []
    numeric = [f for f in features if f != "type"]

    transformers = []

    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            )
        )

    if numeric:
        transformers.append(
            ("numeric", StandardScaler(), numeric)
        )

    preprocessing = ColumnTransformer(transformers)

    classifier = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        random_state=42,
        max_iter=20,
        tol=1e-3,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    print("Loading training and validation splits only...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    train = engineer_features(train)
    validation = engineer_features(validation)

    y_train = train["isFraud"].astype(int)
    y_val = validation["isFraud"].astype(int)

    results = []

    for name, features in EXPERIMENTS.items():
        print(f"\nTraining experiment: {name}")
        print("Features:", features)

        model = make_model(features)
        model.fit(train[features], y_train)

        scores = model.predict_proba(validation[features])[:, 1]

        ap = average_precision_score(y_val, scores)
        auc = roc_auc_score(y_val, scores)

        row = {
            "experiment": name,
            "feature_count": len(features),
            "average_precision": float(ap),
            "roc_auc": float(auc),
        }

        for threshold in [0.87, 0.984]:
            pred = scores >= threshold

            tp = int(((pred) & (y_val.to_numpy() == 1)).sum())
            fp = int(((pred) & (y_val.to_numpy() == 0)).sum())
            fn = int(((~pred) & (y_val.to_numpy() == 1)).sum())
            alerts = tp + fp

            row[f"alerts_at_{threshold}"] = alerts
            row[f"recall_at_{threshold}"] = (
                tp / (tp + fn) if (tp + fn) else 0.0
            )
            row[f"precision_at_{threshold}"] = (
                tp / alerts if alerts else 0.0
            )

        results.append(row)

        print(f"Average Precision: {ap:.6f}")
        print(f"ROC-AUC: {auc:.6f}")

    report = pd.DataFrame(results)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_FILE, index=False)

    print("\n--- ABLATION COMPARISON ---")
    print(report.to_string(index=False))
    print("\nReport saved to:", REPORT_FILE.resolve())
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
