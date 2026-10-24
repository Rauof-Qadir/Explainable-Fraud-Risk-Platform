
from pathlib import Path
import json

import pandas as pd
import yaml

CONFIG_FILE = Path("configs/model_config.yaml")
DATA_FILE = Path("data/raw/transactions.csv")
OUTPUT_FILE = Path("reports/temporal_split_report.json")

with CONFIG_FILE.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

evaluation = config["evaluation"]

train_fraction = evaluation["train_fraction"]
validation_fraction = evaluation["validation_fraction"]
test_fraction = evaluation["test_fraction"]

if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
    raise ValueError("Train, validation and test fractions must sum to 1.")

time_column = config["data"]["time_column"]
target = config["data"]["target"]

# Sirf time column aur target read karenge.
# Is se memory usage kam rahegi.
data = pd.read_csv(
    DATA_FILE,
    usecols=[time_column, target],
    dtype={time_column: "int64", target: "int8"},
)

if data.empty:
    raise ValueError("Dataset is empty.")

if data[time_column].isna().any() or data[target].isna().any():
    raise ValueError("Time column or target contains missing values.")

steps = sorted(data[time_column].unique())
n_steps = len(steps)

if n_steps < 3:
    raise ValueError("Not enough unique time steps for three splits.")

train_index = max(1, int(n_steps * train_fraction))
validation_index = max(train_index + 1, int(n_steps * (train_fraction + validation_fraction)))

if validation_index >= n_steps:
    raise ValueError("Split boundaries leave no time steps for the test set.")

train_end = int(steps[train_index - 1])
validation_end = int(steps[validation_index - 1])

conditions = [
    data[time_column] <= train_end,
    (data[time_column] > train_end) & (data[time_column] <= validation_end),
    data[time_column] > validation_end,
]

split_names = ["train", "validation", "test"]

report = {
    "dataset_rows": int(len(data)),
    "unique_time_steps": int(n_steps),
    "split_strategy": "chronological_by_step",
    "boundaries": {
        "train_end_step": train_end,
        "validation_end_step": validation_end,
    },
    "splits": {},
}

for name, mask in zip(split_names, conditions):
    part = data.loc[mask]
    fraud_count = int(part[target].sum())
    row_count = int(len(part))

    report["splits"][name] = {
        "rows": row_count,
        "min_step": int(part[time_column].min()) if row_count else None,
        "max_step": int(part[time_column].max()) if row_count else None,
        "fraud_count": fraud_count,
        "fraud_rate_percent": (
            round(100 * fraud_count / row_count, 6)
            if row_count else None
        ),
    }

report["row_counts_sum"] = sum(
    item["rows"] for item in report["splits"].values()
)
report["row_counts_match_dataset"] = (
    report["row_counts_sum"] == report["dataset_rows"]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- TEMPORAL SPLIT INSPECTION ---")
print(json.dumps(report, indent=2))
print(f"\nReport saved to: {OUTPUT_FILE.resolve()}")
