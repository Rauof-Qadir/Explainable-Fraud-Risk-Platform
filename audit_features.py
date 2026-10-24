
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/transactions.csv")
REPORT_PATH = Path("reports/feature_leakage_audit.json")
CHUNK_SIZE = 200_000

# These columns are not model inputs.
TARGET_COLUMN = "isFraud"
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]

# Their availability depends on when the risk decision is made.
POST_TRANSACTION_COLUMNS = [
    "newbalanceOrig",
    "newbalanceDest",
]

# Must be investigated before use.
EXISTING_DECISION_COLUMNS = ["isFlaggedFraud"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH.resolve())

    rows = 0
    fraud_by_type = {}
    rows_by_type = {}
    flagged_counts = {}
    flagged_fraud_counts = {}
    fraud_counts = {"0": 0, "1": 0}
    balance_checks = {
        "sender_balance_mismatch": 0,
        "receiver_balance_mismatch": 0,
        "sender_rows_checked": 0,
        "receiver_rows_checked": 0,
    }

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
    ):
        rows += len(chunk)

        # Fraud prevalence by transaction type.
        grouped = chunk.groupby("type")["isFraud"].agg(
            ["count", "sum"]
        )

        for transaction_type, record in grouped.iterrows():
            key = str(transaction_type)
            rows_by_type[key] = (
                rows_by_type.get(key, 0)
                + int(record["count"])
            )
            fraud_by_type[key] = (
                fraud_by_type.get(key, 0)
                + int(record["sum"])
            )

        # Relationship between the existing flag and actual label.
        cross = pd.crosstab(
            chunk["isFlaggedFraud"],
            chunk["isFraud"],
        )

        for flag_value, record in cross.iterrows():
            flag_key = str(flag_value)
            flagged_counts[flag_key] = (
                flagged_counts.get(flag_key, 0)
                + int(record.sum())
            )

            fraud_value = int(record.get(1, 0))
            flagged_fraud_counts[flag_key] = (
                flagged_fraud_counts.get(flag_key, 0)
                + fraud_value
            )

        labels = chunk["isFraud"].value_counts()
        for label, count in labels.items():
            fraud_counts[str(label)] += int(count)

        # Investigative balance checks, not automatic data errors.
        # Tolerance avoids floating-point noise.
        sender_expected = (
            chunk["oldbalanceOrg"] - chunk["amount"]
        )
        sender_comparable = (
            chunk["type"].isin(["TRANSFER", "CASH_OUT"])
        )
        sender_mismatch = (
            (sender_expected - chunk["newbalanceOrig"]).abs()
            > 0.01
        ) & sender_comparable

        receiver_expected = (
            chunk["oldbalanceDest"] + chunk["amount"]
        )
        receiver_comparable = (
            chunk["type"].isin(["TRANSFER", "CASH_IN"])
        )
        receiver_mismatch = (
            (receiver_expected - chunk["newbalanceDest"]).abs()
            > 0.01
        ) & receiver_comparable

        balance_checks["sender_rows_checked"] += int(
            sender_comparable.sum()
        )
        balance_checks["receiver_rows_checked"] += int(
            receiver_comparable.sum()
        )
        balance_checks["sender_balance_mismatch"] += int(
            sender_mismatch.sum()
        )
        balance_checks["receiver_balance_mismatch"] += int(
            receiver_mismatch.sum()
        )

    type_summary = {}

    for transaction_type, count in rows_by_type.items():
        fraud_count = fraud_by_type[transaction_type]
        type_summary[transaction_type] = {
            "transactions": count,
            "fraud_transactions": fraud_count,
            "fraud_rate_percent": (
                100 * fraud_count / count if count else 0
            ),
        }

    flag_summary = {}

    for flag_value, count in flagged_counts.items():
        fraud_count = flagged_fraud_counts.get(flag_value, 0)
        flag_summary[flag_value] = {
            "transactions": count,
            "fraud_transactions": fraud_count,
            "fraud_rate_percent": (
                100 * fraud_count / count if count else 0
            ),
        }

    report = {
        "total_rows": rows,
        "target_column": TARGET_COLUMN,
        "fraud_counts": fraud_counts,
        "fraud_by_transaction_type": type_summary,
        "actual_fraud_by_existing_flag": flag_summary,
        "balance_consistency_checks": balance_checks,
        "feature_availability_review": {
            "isFraud": (
                "Target only. Never include as a model input."
            ),
            "nameOrig": (
                "Identifier. Exclude initially; assess privacy "
                "and memorization risks before any alternative use."
            ),
            "nameDest": (
                "Identifier. Exclude initially; assess privacy "
                "and memorization risks before any alternative use."
            ),
            "isFlaggedFraud": (
                "Exclude from the first baseline. Investigate "
                "how and when the flag is generated."
            ),
            "newbalanceOrig": (
                "Post-transaction value. Exclude from a "
                "pre-transaction model unless availability "
                "at decision time is demonstrated."
            ),
            "newbalanceDest": (
                "Post-transaction value. Exclude from a "
                "pre-transaction model unless availability "
                "at decision time is demonstrated."
            ),
        },
        "limitations": [
            "Balance checks are investigative, not proof of corruption.",
            "Balance equations may not hold for every transaction type.",
            "Observed association does not establish causation.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n--- FEATURE & LEAKAGE AUDIT ---")
    print(f"Rows inspected: {rows:,}")

    print("\nFraud by transaction type:")
    for name, values in type_summary.items():
        print(
            f"{name}: {values['fraud_transactions']:,} / "
            f"{values['transactions']:,} "
            f"({values['fraud_rate_percent']:.4f}%)"
        )

    print("\nActual fraud by existing flag:")
    for name, values in flag_summary.items():
        print(f"Flag={name}: {values}")

    print("\nBalance consistency checks:")
    print(json.dumps(balance_checks, indent=2))

    print(f"\nReport saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
