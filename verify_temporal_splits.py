
from pathlib import Path
import json

import pandas as pd

SPLIT_DIR = Path("data/processed/splits")
REPORT_FILE = Path("reports/split_integrity_report.json")
CHUNK_SIZE = 200_000

EXPECTED = {
    "train": {
        "rows": 4_433_703,
        "frauds": 3_633,
        "min_step": 1,
        "max_step": 322,
    },
    "validation": {
        "rows": 973_173,
        "frauds": 570,
        "min_step": 323,
        "max_step": 377,
    },
    "test": {
        "rows": 955_744,
        "frauds": 4_010,
        "min_step": 378,
        "max_step": 743,
    },
}

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]

report = {
    "checks": {},
    "splits": {},
}

all_checks = []

for name, expected in EXPECTED.items():
    file_path = SPLIT_DIR / f"{name}.csv"

    assert file_path.exists(), f"Missing file: {file_path}"

    # Sirf header read karke columns check karo.
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    columns_ok = columns == REQUIRED_COLUMNS

    rows = 0
    frauds = 0
    min_step = None
    max_step = None
    invalid_labels = 0
    invalid_steps = 0

    # Large files ko chunks mein verify karo.
    for chunk in pd.read_csv(
        file_path,
        usecols=["step", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        rows += len(chunk)
        frauds += int(chunk["isFraud"].sum())

        chunk_min = int(chunk["step"].min())
        chunk_max = int(chunk["step"].max())

        min_step = (
            chunk_min if min_step is None
            else min(min_step, chunk_min)
        )
        max_step = (
            chunk_max if max_step is None
            else max(max_step, chunk_max)
        )

        invalid_labels += int(
            (~chunk["isFraud"].isin([0, 1])).sum()
        )

        invalid_steps += int(
            (
                (chunk["step"] < expected["min_step"])
                | (chunk["step"] > expected["max_step"])
            ).sum()
        )

    checks = {
        "file_exists": file_path.exists(),
        "columns_match": columns_ok,
        "row_count_matches": rows == expected["rows"],
        "fraud_count_matches": frauds == expected["frauds"],
        "step_range_matches": (
            min_step == expected["min_step"]
            and max_step == expected["max_step"]
        ),
        "binary_labels_valid": invalid_labels == 0,
        "all_steps_within_boundary": invalid_steps == 0,
    }

    all_checks.extend(checks.values())

    report["splits"][name] = {
        "rows": rows,
        "fraud_count": frauds,
        "min_step": min_step,
        "max_step": max_step,
        "invalid_labels": invalid_labels,
        "invalid_steps": invalid_steps,
        "checks": checks,
    }

total_rows = sum(
    item["rows"] for item in report["splits"].values()
)

report["total_rows"] = total_rows
report["total_rows_match_expected"] = (
    total_rows == sum(item["rows"] for item in EXPECTED.values())
)

all_checks.append(report["total_rows_match_expected"])

report["all_checks_passed"] = all(all_checks)

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n--- SPLIT INTEGRITY REPORT ---")
print(json.dumps(report, indent=2))
print("\nALL CHECKS PASSED:", report["all_checks_passed"])
print("Report saved to:", REPORT_FILE.resolve())

if not report["all_checks_passed"]:
    raise SystemExit("FAILED: Please investigate the failed checks.")

print("\nSUCCESS: All integrity checks passed.")
