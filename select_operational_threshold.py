
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------
MODEL_FILE = Path("models/baseline_model.joblib")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")

CSV_REPORT = Path("reports/operational_threshold_comparison.csv")
JSON_REPORT = Path("reports/operational_threshold_report.json")

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

# Business requirements
MAX_ALERT_RATE = 0.05       # Maximum 5% transactions flagged
MIN_RECALL = 0.80           # Catch at least 80% of fraud

# Same assumed costs used in earlier analysis
FP_COST = 5.0               # Cost of a false alert
FN_COST = 150.0             # Cost of a missed fraud

# Test thresholds from 0.000 to 1.000.
# 1.001 is a no-alert benchmark.
THRESHOLDS = sorted(
    set(
        [round(float(t), 3)
         for t in np.arange(0.0, 1.001, 0.001)]
        + [1.001]
    )
)


# --------------------------------------------------
# 2. Load model and VALIDATION data only
# --------------------------------------------------
print("Loading model...")
model = joblib.load(MODEL_FILE)

print("Loading validation data only...")
df = pd.read_csv(VALIDATION_FILE)

missing = set(FEATURES + ["isFraud"]) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

X = df[FEATURES]
y = df["isFraud"].to_numpy(dtype=np.int8)

if len(y) == 0:
    raise ValueError("Validation dataset is empty.")

if not set(np.unique(y)).issubset({0, 1}):
    raise ValueError("Target must contain only binary labels 0 and 1.")

print(f"Validation rows: {len(y):,}")
print(f"Actual fraud cases: {int(y.sum()):,}")
print(f"Maximum alerts allowed: {int(np.floor(len(y) * MAX_ALERT_RATE)):,}")
print("Scoring validation transactions...")

scores = model.predict_proba(X)[:, 1]


# --------------------------------------------------
# 3. Evaluate each threshold
# --------------------------------------------------
rows = []

for threshold in THRESHOLDS:
    predictions = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    alerts = int(tp + fp)
    total = int(len(y))

    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    precision = float(tp / alerts) if alerts else 0.0
    alert_rate = float(alerts / total)

    false_alert_cost = float(fp * FP_COST)
    missed_fraud_cost = float(fn * FN_COST)
    total_cost = false_alert_cost + missed_fraud_cost

    meets_alert_limit = alert_rate <= MAX_ALERT_RATE
    meets_recall_target = recall >= MIN_RECALL

    rows.append({
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "alerts": alerts,
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "false_alert_cost": false_alert_cost,
        "missed_fraud_cost": missed_fraud_cost,
        "total_cost": float(total_cost),
        "meets_alert_limit": bool(meets_alert_limit),
        "meets_recall_target": bool(meets_recall_target),
        "is_feasible": bool(
            meets_alert_limit and meets_recall_target
        ),
    })

results = pd.DataFrame(rows)


# --------------------------------------------------
# 4. Find feasible policy and useful alternatives
# --------------------------------------------------
# A feasible policy must satisfy BOTH requirements.
feasible = results[results["is_feasible"]].copy()

# Best recall possible while respecting the 5% alert cap.
within_alert_limit = results[
    results["alert_rate"] <= MAX_ALERT_RATE
].copy()

best_recall_under_cap = (
    within_alert_limit
    .sort_values(
        ["recall", "total_cost", "threshold"],
        ascending=[False, True, False],
    )
    .iloc[0]
)

# Lowest-alert policy that reaches the recall target,
# whether or not it exceeds the 5% cap.
recall_target_candidates = results[
    results["recall"] >= MIN_RECALL
].copy()

if not recall_target_candidates.empty:
    lowest_alert_policy_at_target = (
        recall_target_candidates
        .sort_values(
            ["alerts", "total_cost", "threshold"],
            ascending=[True, True, False],
        )
        .iloc[0]
    )
else:
    lowest_alert_policy_at_target = None

# Among feasible policies, minimize estimated cost.
if not feasible.empty:
    selected = (
        feasible
        .sort_values(
            ["total_cost", "threshold"],
            ascending=[True, False],
        )
        .iloc[0]
    )

    selected_policy = {
        "status": "FEASIBLE_POLICY_FOUND",
        "threshold": float(selected["threshold"]),
        "estimated_total_cost": float(selected["total_cost"]),
        "alerts": int(selected["alerts"]),
        "alert_rate": float(selected["alert_rate"]),
        "true_positives": int(selected["tp"]),
        "false_positives": int(selected["fp"]),
        "false_negatives": int(selected["fn"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
    }
else:
    selected_policy = {
        "status": "NO_FEASIBLE_POLICY_ON_TESTED_GRID",
        "message": (
            "No tested threshold satisfies both the maximum "
            "alert rate and minimum recall requirements."
        ),
    }


# --------------------------------------------------
# 5. Save reports
# --------------------------------------------------
CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(CSV_REPORT, index=False)

report = {
    "evaluation_data": "validation only",
    "validation_rows": int(len(y)),
    "validation_frauds": int(y.sum()),
    "requirements": {
        "maximum_alert_rate": MAX_ALERT_RATE,
        "minimum_recall": MIN_RECALL,
    },
    "cost_assumptions": {
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
    },
    "threshold_grid": {
        "minimum": float(min(THRESHOLDS)),
        "maximum": float(max(THRESHOLDS)),
        "number_of_candidates": int(len(THRESHOLDS)),
    },
    "selected_policy": selected_policy,
    "best_recall_within_alert_limit": {
        "threshold": float(best_recall_under_cap["threshold"]),
        "alerts": int(best_recall_under_cap["alerts"]),
        "alert_rate": float(best_recall_under_cap["alert_rate"]),
        "recall": float(best_recall_under_cap["recall"]),
        "precision": float(best_recall_under_cap["precision"]),
        "estimated_total_cost": float(
            best_recall_under_cap["total_cost"]
        ),
        "meets_recall_target": bool(
            best_recall_under_cap["meets_recall_target"]
        ),
    },
    "lowest_alert_policy_reaching_recall_target": (
        {
            "threshold": float(
                lowest_alert_policy_at_target["threshold"]
            ),
            "alerts": int(lowest_alert_policy_at_target["alerts"]),
            "alert_rate": float(
                lowest_alert_policy_at_target["alert_rate"]
            ),
            "recall": float(
                lowest_alert_policy_at_target["recall"]
            ),
            "precision": float(
                lowest_alert_policy_at_target["precision"]
            ),
            "estimated_total_cost": float(
                lowest_alert_policy_at_target["total_cost"]
            ),
            "meets_alert_limit": bool(
                lowest_alert_policy_at_target["meets_alert_limit"]
            ),
        }
        if lowest_alert_policy_at_target is not None
        else None
    ),
    "notes": [
        "Only validation data was used.",
        "The test split was not loaded or modified.",
        "Costs are assumptions, not verified real-world losses.",
        "Thresholds were evaluated on a 0.001 grid.",
        "The selected policy minimizes estimated cost only "
        "among tested thresholds satisfying both constraints.",
        "Model scores have not been calibrated.",
    ],
}

JSON_REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# --------------------------------------------------
# 6. Print business-readable results
# --------------------------------------------------
print("\n" + "=" * 60)
print("OPERATIONAL THRESHOLD SELECTION")
print("=" * 60)

print(f"Maximum alert rate: {MAX_ALERT_RATE:.1%}")
print(f"Minimum recall target: {MIN_RECALL:.1%}")
print(f"False-positive cost assumption: ${FP_COST:.2f}")
print(f"False-negative cost assumption: ${FN_COST:.2f}")

print("\n--- BEST RECALL WITHIN 5% ALERT LIMIT ---")
print(f"Threshold: {best_recall_under_cap['threshold']:.3f}")
print(f"Alerts: {int(best_recall_under_cap['alerts']):,}")
print(f"Alert rate: {best_recall_under_cap['alert_rate']:.3%}")
print(f"Recall: {best_recall_under_cap['recall']:.2%}")
print(f"Precision: {best_recall_under_cap['precision']:.2%}")
print(f"Estimated cost: ${best_recall_under_cap['total_cost']:,.2f}")
print(
    "Meets 80% recall target: "
    f"{bool(best_recall_under_cap['meets_recall_target'])}"
)

print("\n--- FINAL POLICY STATUS ---")
print(selected_policy["status"])

if not feasible.empty:
    print(f"Selected threshold: {selected['threshold']:.3f}")
    print(f"Alerts: {int(selected['alerts']):,}")
    print(f"Alert rate: {selected['alert_rate']:.3%}")
    print(f"Recall: {selected['recall']:.2%}")
    print(f"Precision: {selected['precision']:.2%}")
    print(f"Estimated cost: ${selected['total_cost']:,.2f}")
else:
    print("No tested threshold satisfies both requirements.")
    if lowest_alert_policy_at_target is not None:
        print(
            "Lowest-alert threshold reaching target recall: "
            f"{lowest_alert_policy_at_target['threshold']:.3f}"
        )
        print(
            "Its alert rate: "
            f"{lowest_alert_policy_at_target['alert_rate']:.3%}"
        )
        print(
            "It exceeds the alert limit: "
            f"{not lowest_alert_policy_at_target['meets_alert_limit']}"
        )
    else:
        print("The 80% recall target was not reached on this grid.")

print("\nCSV report:", CSV_REPORT.resolve())
print("JSON report:", JSON_REPORT.resolve())
print("\nTest data was not loaded or changed.")
