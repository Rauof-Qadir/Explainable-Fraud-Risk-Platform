
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.engineering import engineer_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
VALIDATION_FILE = Path("data/processed/splits/validation.csv")

V1_MODEL_FILE = Path("models/baseline_model.joblib")
V2_MODEL_FILE = Path("models/v2_feature_model.joblib")

CSV_REPORT = Path("reports/v1_v2_validation_comparison.csv")
JSON_REPORT = Path("reports/v1_v2_validation_report.json")

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

MAX_ALERT_RATE = 0.05
MIN_RECALL = 0.80
FP_COST = 5.0
FN_COST = 150.0

THRESHOLDS = np.array(
    [round(float(t), 3) for t in np.arange(0.0, 1.001, 0.001)]
    + [1.001],
    dtype=float,
)


# --------------------------------------------------
# 2. Evaluate score thresholds efficiently
# --------------------------------------------------
def evaluate_thresholds(y, scores):
    # Sort scores from highest to lowest.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]

    # Number of frauds among the top N scores.
    cumulative_tp = np.concatenate(
        ([0], np.cumsum(sorted_y, dtype=np.int64))
    )

    # Convert descending scores into ascending values for search.
    ascending_negative_scores = -sorted_scores

    total = len(y)
    total_fraud = int(y.sum())
    total_nonfraud = total - total_fraud
    rows = []

    for threshold in THRESHOLDS:
        # Count transactions whose score is >= threshold.
        alerts = int(
            np.searchsorted(
                ascending_negative_scores,
                -threshold,
                side="right",
            )
        )

        tp = int(cumulative_tp[alerts])
        fp = alerts - tp
        fn = total_fraud - tp
        tn = total_nonfraud - fp

        recall = tp / total_fraud if total_fraud else 0.0
        precision = tp / alerts if alerts else 0.0
        alert_rate = alerts / total

        total_cost = fp * FP_COST + fn * FN_COST

        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "alerts": alerts,
            "alert_rate": float(alert_rate),
            "precision": float(precision),
            "recall": float(recall),
            "estimated_cost": float(total_cost),
            "meets_alert_limit": bool(
                alert_rate <= MAX_ALERT_RATE
            ),
            "meets_recall_target": bool(
                recall >= MIN_RECALL
            ),
        })

    return pd.DataFrame(rows)


def summarize_model(name, model, X, y):
    print(f"\nScoring {name}...")
    scores = model.predict_proba(X)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError(f"{name} produced non-finite scores.")

    # Standard comparison at threshold 0.5.
    pred_05 = (scores >= 0.5).astype(np.int8)

    tp_05 = int(((pred_05 == 1) & (y == 1)).sum())
    fp_05 = int(((pred_05 == 1) & (y == 0)).sum())
    fn_05 = int(((pred_05 == 0) & (y == 1)).sum())
    tn_05 = int(((pred_05 == 0) & (y == 0)).sum())

    recall_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) else 0.0
    precision_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) else 0.0

    # Threshold-level operational analysis.
    table = evaluate_thresholds(y, scores)

    within_cap = table[
        table["alert_rate"] <= MAX_ALERT_RATE
    ]

    best_recall_under_cap = (
        within_cap
        .sort_values(
            ["recall", "estimated_cost", "threshold"],
            ascending=[False, True, False],
        )
        .iloc[0]
    )

    feasible = table[
        (table["alert_rate"] <= MAX_ALERT_RATE)
        & (table["recall"] >= MIN_RECALL)
    ]

    best_feasible = None

    if not feasible.empty:
        best_feasible = (
            feasible
            .sort_values(
                ["estimated_cost", "threshold"],
                ascending=[True, False],
            )
            .iloc[0]
        )

    target_candidates = table[
        table["recall"] >= MIN_RECALL
    ]

    lowest_alert_at_target = None

    if not target_candidates.empty:
        lowest_alert_at_target = (
            target_candidates
            .sort_values(
                ["alerts", "estimated_cost", "threshold"],
                ascending=[True, True, False],
            )
            .iloc[0]
        )

    summary = {
        "model": name,
        "average_precision_pr_auc": float(
            average_precision_score(y, scores)
        ),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold_0_5": {
            "tp": tp_05,
            "fp": fp_05,
            "fn": fn_05,
            "tn": tn_05,
            "precision": float(precision_05),
            "recall": float(recall_05),
            "alerts": int(tp_05 + fp_05),
            "alert_rate": float((tp_05 + fp_05) / len(y)),
        },
        "best_recall_within_5_percent_alert_limit": {
            "threshold": float(best_recall_under_cap["threshold"]),
            "recall": float(best_recall_under_cap["recall"]),
            "precision": float(best_recall_under_cap["precision"]),
            "alerts": int(best_recall_under_cap["alerts"]),
            "alert_rate": float(best_recall_under_cap["alert_rate"]),
            "estimated_cost": float(
                best_recall_under_cap["estimated_cost"]
            ),
            "meets_80_percent_recall": bool(
                best_recall_under_cap["recall"] >= MIN_RECALL
            ),
        },
        "best_feasible_policy": (
            {
                "threshold": float(best_feasible["threshold"]),
                "recall": float(best_feasible["recall"]),
                "precision": float(best_feasible["precision"]),
                "alerts": int(best_feasible["alerts"]),
                "alert_rate": float(best_feasible["alert_rate"]),
                "estimated_cost": float(
                    best_feasible["estimated_cost"]
                ),
            }
            if best_feasible is not None
            else None
        ),
        "lowest_alert_policy_reaching_80_percent_recall": (
            {
                "threshold": float(lowest_alert_at_target["threshold"]),
                "recall": float(lowest_alert_at_target["recall"]),
                "precision": float(lowest_alert_at_target["precision"]),
                "alerts": int(lowest_alert_at_target["alerts"]),
                "alert_rate": float(
                    lowest_alert_at_target["alert_rate"]
                ),
                "meets_5_percent_alert_limit": bool(
                    lowest_alert_at_target["alert_rate"]
                    <= MAX_ALERT_RATE
                ),
            }
            if lowest_alert_at_target is not None
            else None
        ),
    }

    return summary, table


# --------------------------------------------------
# 3. Load validation data ONLY
# --------------------------------------------------
print("Loading validation data only...")
df = pd.read_csv(VALIDATION_FILE)

if df.empty:
    raise ValueError("Validation data is empty.")

if "isFraud" not in df.columns:
    raise ValueError("Missing target column isFraud.")

y = df["isFraud"].to_numpy(dtype=np.int8)

if set(np.unique(y)) - {0, 1}:
    raise ValueError("Target must contain only 0 and 1.")

print(f"Validation rows: {len(df):,}")
print(f"Validation fraud cases: {int(y.sum()):,}")

# Build V1 and V2 inputs from the SAME validation rows.
X_v1 = df[BASE_FEATURES]
featured_validation = engineer_features(df)
X_v2 = featured_validation[V2_FEATURES]


# --------------------------------------------------
# 4. Load both already-trained models
# --------------------------------------------------
print("\nLoading existing V1 and V2 models...")
v1_model = joblib.load(V1_MODEL_FILE)
v2_model = joblib.load(V2_MODEL_FILE)

v1_summary, v1_table = summarize_model(
    "V1_baseline",
    v1_model,
    X_v1,
    y,
)

v2_summary, v2_table = summarize_model(
    "V2_engineered_features",
    v2_model,
    X_v2,
    y,
)


# --------------------------------------------------
# 5. Save reports
# --------------------------------------------------
comparison = pd.concat(
    [
        v1_table.assign(model="V1_baseline"),
        v2_table.assign(model="V2_engineered_features"),
    ],
    ignore_index=True,
)

CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(CSV_REPORT, index=False)

report = {
    "evaluation_data": "validation only",
    "validation_rows": int(len(y)),
    "validation_fraud_cases": int(y.sum()),
    "operational_requirements": {
        "maximum_alert_rate": MAX_ALERT_RATE,
        "minimum_recall": MIN_RECALL,
    },
    "cost_assumptions": {
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
    },
    "models": {
        "V1_baseline": v1_summary,
        "V2_engineered_features": v2_summary,
    },
    "notes": [
        "Both models were evaluated on the same validation rows.",
        "Validation data was not used to fit either model in this script.",
        "Test data was not loaded.",
        "Costs are assumptions, not verified real-world losses.",
        "Threshold candidates use a 0.001 grid.",
        "PR-AUC here means average precision.",
        "Model scores have not been calibrated.",
    ],
}

JSON_REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# --------------------------------------------------
# 6. Print comparison
# --------------------------------------------------
print("\n" + "=" * 72)
print("V1 vs V2 — VALIDATION COMPARISON")
print("=" * 72)

for summary in [v1_summary, v2_summary]:
    print(f"\n--- {summary['model']} ---")
    print(
        "Average precision / PR-AUC: "
        f"{summary['average_precision_pr_auc']:.6f}"
    )
    print(f"ROC-AUC: {summary['roc_auc']:.6f}")

    fixed = summary["threshold_0_5"]
    print("\nAt threshold 0.5:")
    print(f"  Precision: {fixed['precision']:.4%}")
    print(f"  Recall:    {fixed['recall']:.4%}")
    print(f"  Alerts:    {fixed['alerts']:,}")
    print(f"  Alert rate:{fixed['alert_rate']:.3%}")

    best = summary["best_recall_within_5_percent_alert_limit"]
    print("\nBest recall under 5% alert limit:")
    print(f"  Threshold: {best['threshold']:.3f}")
    print(f"  Recall:    {best['recall']:.2%}")
    print(f"  Precision: {best['precision']:.2%}")
    print(f"  Alerts:    {best['alerts']:,}")
    print(f"  Alert rate:{best['alert_rate']:.3%}")
    print(f"  Cost:      ${best['estimated_cost']:,.2f}")
    print(
        "  Meets 80% recall: "
        f"{best['meets_80_percent_recall']}"
    )

    target = summary["lowest_alert_policy_reaching_80_percent_recall"]
    if target is None:
        print("\n  80% recall was not reached on the tested grid.")
    else:
        print("\nLowest-alert policy reaching 80% recall:")
        print(f"  Threshold: {target['threshold']:.3f}")
        print(f"  Alerts:    {target['alerts']:,}")
        print(f"  Alert rate:{target['alert_rate']:.3%}")
        print(
            "  Meets 5% alert limit: "
            f"{target['meets_5_percent_alert_limit']}"
        )

print("\nSaved CSV:", CSV_REPORT.resolve())
print("Saved JSON:", JSON_REPORT.resolve())
print("\nTest data was not loaded or changed.")
