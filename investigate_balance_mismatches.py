
from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/raw/transactions.csv")
CHUNK_SIZE = 200_000
TOLERANCE = 0.01

# Har transaction type aur fraud label ka result store hoga.
stats = {}

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
    sender_types = chunk["type"].isin(["TRANSFER", "CASH_OUT"])
    receiver_types = chunk["type"].isin(["TRANSFER", "CASH_IN"])

    sender_expected = chunk["oldbalanceOrg"] - chunk["amount"]
    receiver_expected = chunk["oldbalanceDest"] + chunk["amount"]

    sender_mismatch = (
        (chunk["newbalanceOrig"] - sender_expected).abs() > TOLERANCE
    ) & sender_types

    receiver_mismatch = (
        (chunk["newbalanceDest"] - receiver_expected).abs() > TOLERANCE
    ) & receiver_types

    for tx_type in chunk["type"].unique():
        for fraud_label in (0, 1):
            mask = (
                (chunk["type"] == tx_type)
                & (chunk["isFraud"] == fraud_label)
            )

            if not mask.any():
                continue

            key = (tx_type, fraud_label)

            if key not in stats:
                stats[key] = {
                    "transactions": 0,
                    "sender_checked": 0,
                    "sender_mismatches": 0,
                    "receiver_checked": 0,
                    "receiver_mismatches": 0,
                }

            item = stats[key]
            item["transactions"] += int(mask.sum())

            sender_mask = mask & sender_types
            receiver_mask = mask & receiver_types

            item["sender_checked"] += int(sender_mask.sum())
            item["sender_mismatches"] += int(
                (sender_mismatch & mask).sum()
            )

            item["receiver_checked"] += int(receiver_mask.sum())
            item["receiver_mismatches"] += int(
                (receiver_mismatch & mask).sum()
            )

rows = []

for (tx_type, fraud_label), item in sorted(stats.items()):
    sender_checked = item["sender_checked"]
    receiver_checked = item["receiver_checked"]

    rows.append({
        "type": tx_type,
        "isFraud": fraud_label,
        "transactions": item["transactions"],
        "sender_checked": sender_checked,
        "sender_mismatches": item["sender_mismatches"],
        "sender_mismatch_pct": (
            100 * item["sender_mismatches"] / sender_checked
            if sender_checked else None
        ),
        "receiver_checked": receiver_checked,
        "receiver_mismatches": item["receiver_mismatches"],
        "receiver_mismatch_pct": (
            100 * item["receiver_mismatches"] / receiver_checked
            if receiver_checked else None
        ),
    })

report = pd.DataFrame(rows)

output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "balance_mismatch_breakdown.csv"
report.to_csv(output_file, index=False)

print("\n--- BALANCE MISMATCH BREAKDOWN ---")
print(report.to_string(index=False))
print(f"\nReport saved to: {output_file.resolve()}")
