
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

MODEL_FILE = Path("models/baseline_model.joblib")
VALIDATION_FILE = Path("data/processed/splits/validation.csv")
CSV_REPORT = Path("reports/cost_sensitivity.csv")
JSON_REPORT = Path("reports/cost_sensitivity_report.json")

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

FP_COST = 5.0
FN_COSTS = [150.0, 500.0, 1000.0]

# Coarse grid across the full range plus fine grid near high scores.
THRESHOLDS = sorted(set(
    [round(x, 2) for x in np.arange(0.10, 1.001, 0.01)]
    + [round(x, 3) for x in np.arange(0.90, 1.001, 0.001)]
))

print("Loading model and validation data...")
model = joblib.load(MODEL_FILE)
df = pd.read_csv(VALIDATION_FILE)

X = df[FEATURES]
y = df["isFraud"].to_numpy(dtype=np.int8)

print("Scoring validation data...")
scores = model.predict_proba(X)[:, 1]

# Har threshold par confusion matrix ek baar calculate karo.
counts = []

for threshold in THRESHOLDS:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    counts.append({
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "alerts": int(tp + fp),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
    })

base_df = pd.DataFrame(counts)
all_results = []
scenario_summary = {}

for fn_cost in FN_COSTS:
    scenario = base_df.copy()
    scenario["fp_cost_assumption"] = FP_COST
    scenario["fn_cost_assumption"] = fn_cost
    scenario["false_alert_cost"] = scenario["fp"] * FP_COST
    scenario["missed_fraud_cost"] = scenario["fn"] * fn_cost
    scenario["total_cost"] = (
        scenario["false_alert_cost"]
        + scenario["missed_fraud_cost"]
    )

    best = scenario.sort_values(
        ["total_cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    scenario_summary[str(int(fn_cost))] = {
        "false_positive_cost": FP_COST,
        "false_negative_cost": fn_cost,
        "best_threshold_on_tested_grid": float(best["threshold"]),
        "estimated_total_cost": float(best["total_cost"]),
        "alerts": int(best["alerts"]),
        "true_positives": int(best["tp"]),
        "false_positives": int(best["fp"]),
        "false_negatives": int(best["fn"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
    }

    all_results.append(scenario)

results = pd.concat(all_results, ignore_index=True)

CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(CSV_REPORT, index=False)

report = {
    "evaluation_data": "validation only",
    "validation_rows": int(len(y)),
    "validation_frauds": int(y.sum()),
    "false_positive_cost": FP_COST,
    "false_negative_cost_scenarios": FN_COSTS,
    "best_result_per_scenario": scenario_summary,
    "note": (
        "Thresholds are candidates on the tested grid. "
        "Costs are assumptions, not verified real-world losses. "
        "Model scores have not been calibrated."
    ),
}

JSON_REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- COST SENSITIVITY SUMMARY ---")
print(json.dumps(scenario_summary, indent=2))
print("\nFull comparison:", CSV_REPORT.resolve())
print("Summary report:", JSON_REPORT.resolve())
