
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/transactions.csv")
REPORT_PATH = Path("reports/data_audit.json")
CHUNK_SIZE = 200_000

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH.resolve())

    # Read only the header first.
    header = pd.read_csv(DATA_PATH, nrows=0)
    columns = header.columns.tolist()

    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    extra_columns = sorted(set(columns) - REQUIRED_COLUMNS)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    total_rows = 0
    missing_counts = {col: 0 for col in columns}
    fraud_counts = {}
    type_counts = {}
    numeric_stats = {
        col: {
            "min": None,
            "max": None,
            "negative_count": 0,
            "zero_count": 0,
            "non_finite_count": 0,
        }
        for col in NUMERIC_COLUMNS
    }

    invalid_fraud_labels = 0
    invalid_flag_labels = 0
    duplicate_rows_within_chunks = 0
    chunk_count = 0

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
    ):
        chunk_count += 1
        total_rows += len(chunk)

        for col in columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        for value, count in chunk["isFraud"].value_counts(
            dropna=False
        ).items():
            key = str(value)
            fraud_counts[key] = fraud_counts.get(key, 0) + int(count)

        for value, count in chunk["type"].value_counts(
            dropna=False
        ).items():
            key = str(value)
            type_counts[key] = type_counts.get(key, 0) + int(count)

        invalid_fraud_labels += int(
            (~chunk["isFraud"].isin([0, 1])).sum()
        )
        invalid_flag_labels += int(
            (~chunk["isFlaggedFraud"].isin([0, 1])).sum()
        )

        # This detects duplicates inside each chunk only.
        # It does not detect duplicates across chunk boundaries.
        duplicate_rows_within_chunks += int(
            chunk.duplicated().sum()
        )

        for col in NUMERIC_COLUMNS:
            values = pd.to_numeric(
                chunk[col], errors="coerce"
            ).to_numpy(dtype=np.float64)

            finite = np.isfinite(values)
            finite_values = values[finite]

            numeric_stats[col]["non_finite_count"] += int(
                (~finite).sum()
            )
            numeric_stats[col]["negative_count"] += int(
                (finite_values < 0).sum()
            )
            numeric_stats[col]["zero_count"] += int(
                (finite_values == 0).sum()
            )

            if finite_values.size:
                chunk_min = float(finite_values.min())
                chunk_max = float(finite_values.max())

                current_min = numeric_stats[col]["min"]
                current_max = numeric_stats[col]["max"]

                numeric_stats[col]["min"] = (
                    chunk_min if current_min is None
                    else min(current_min, chunk_min)
                )
                numeric_stats[col]["max"] = (
                    chunk_max if current_max is None
                    else max(current_max, chunk_max)
                )

    report = {
        "file": str(DATA_PATH),
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "missing_required_columns": missing_columns,
        "unexpected_columns": extra_columns,
        "chunks_processed": chunk_count,
        "chunk_size": CHUNK_SIZE,
        "missing_counts": missing_counts,
        "fraud_label_counts": fraud_counts,
        "transaction_type_counts": type_counts,
        "invalid_isFraud_labels": invalid_fraud_labels,
        "invalid_isFlaggedFraud_labels": invalid_flag_labels,
        "duplicate_rows_within_chunks_only":
            duplicate_rows_within_chunks,
        "numeric_statistics": numeric_stats,
        "limitations": [
            "Duplicate check only covers rows within each chunk.",
            "Missing values are counted, but invalid numeric "
            "parsing and infinities require separate review.",
            "Negative values are flagged for investigation, "
            "not automatically deleted.",
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n--- FULL DATA AUDIT ---")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(columns)}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Missing values: {sum(missing_counts.values()):,}")
    print(f"Fraud labels: {fraud_counts}")
    print(f"Transaction types: {type_counts}")
    print(f"Invalid fraud labels: {invalid_fraud_labels:,}")
    print(f"Invalid flag labels: {invalid_flag_labels:,}")
    print(
        "Duplicates within chunks: "
        f"{duplicate_rows_within_chunks:,}"
    )
    print(f"\nReport saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
