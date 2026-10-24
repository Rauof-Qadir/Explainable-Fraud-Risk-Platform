
from pathlib import Path
import json

import pandas as pd

DATA_FILE = Path("data/raw/transactions.csv")
OUTPUT_DIR = Path("data/processed/splits")
REPORT_FILE = Path("reports/temporal_split_report.json")

CHUNK_SIZE = 200_000

TRAIN_END_STEP = 322
VALIDATION_END_STEP = 377

COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]

SPLITS = {
    "train": lambda df: df["step"] <= TRAIN_END_STEP,
    "validation": lambda df: (
        (df["step"] > TRAIN_END_STEP)
        & (df["step"] <= VALIDATION_END_STEP)
    ),
    "test": lambda df: df["step"] > VALIDATION_END_STEP,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Purani outputs remove karo taake duplicate rows append na hon.
for name in SPLITS:
    output_file = OUTPUT_DIR / f"{name}.csv"
    if output_file.exists():
        output_file.unlink()

summary = {
    name: {"rows": 0, "fraud_count": 0}
    for name in SPLITS
}

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=COLUMNS,
    chunksize=CHUNK_SIZE,
):
    for name, mask_function in SPLITS.items():
        part = chunk.loc[mask_function(chunk)]

        if part.empty:
            continue

        output_file = OUTPUT_DIR / f"{name}.csv"
        write_header = not output_file.exists()

        part.to_csv(
            output_file,
            index=False,
            mode="a",
            header=write_header,
        )

        summary[name]["rows"] += len(part)
        summary[name]["fraud_count"] += int(
            part["isFraud"].sum()
        )

total_rows = sum(item["rows"] for item in summary.values())

for name, item in summary.items():
    item["fraud_rate_percent"] = round(
        100 * item["fraud_count"] / item["rows"], 6
    )
    item["row_share_percent"] = round(
        100 * item["rows"] / total_rows, 3
    )

report = {
    "dataset_rows_written": total_rows,
    "step_boundaries": {
        "train_end_step": TRAIN_END_STEP,
        "validation_end_step": VALIDATION_END_STEP,
    },
    "columns": COLUMNS,
    "splits": summary,
}

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- TEMPORAL SPLIT RESULTS ---")
print(json.dumps(report, indent=2))
print("\nFiles created:")
for name in SPLITS:
    print((OUTPUT_DIR / f"{name}.csv").resolve())
print("\nReport:", REPORT_FILE.resolve())
