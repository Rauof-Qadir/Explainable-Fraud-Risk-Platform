
from pathlib import Path
import pandas as pd

SPLITS = {
    "train": Path("data/processed/splits/train.csv"),
    "validation": Path("data/processed/splits/validation.csv"),
}

OUTPUT = Path("reports/v2_rule_pattern_audit.csv")


def audit_split(name, path):
    print(f"\nLoading {name} split...")
    df = pd.read_csv(path)

    # Fraud is only compared within TRANSFER and CASH_OUT.
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    df["sender_balance_zero"] = df["oldbalanceOrg"].eq(0)
    df["receiver_balance_zero"] = df["oldbalanceDest"].eq(0)
    df["amount_exceeds_sender_balance"] = (
        df["amount"] > df["oldbalanceOrg"]
    )
    df["amount_exceeds_receiver_balance"] = (
        df["amount"] > df["oldbalanceDest"]
    )

    indicators = [
        "sender_balance_zero",
        "receiver_balance_zero",
        "amount_exceeds_sender_balance",
        "amount_exceeds_receiver_balance",
    ]

    rows = []

    for transaction_type, group in df.groupby("type", observed=True):
        for label, subset in group.groupby("isFraud"):
            for indicator in indicators:
                rows.append({
                    "split": name,
                    "type": transaction_type,
                    "label": int(label),
                    "label_name": (
                        "fraud" if int(label) == 1 else "nonfraud"
                    ),
                    "transactions": int(len(subset)),
                    "pattern": indicator,
                    "pattern_count": int(subset[indicator].sum()),
                    "pattern_rate": float(subset[indicator].mean()),
                })

    print(f"\n{name.upper()} — balance pattern rates")
    result = pd.DataFrame(rows)
    print(
        result[
            [
                "type", "label_name", "transactions",
                "pattern", "pattern_count", "pattern_rate",
            ]
        ].to_string(index=False)
    )

    return rows


def main():
    all_rows = []

    for name, path in SPLITS.items():
        all_rows.extend(audit_split(name, path))

    report = pd.DataFrame(all_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("\nReport saved to:", OUTPUT.resolve())
    print("Only training and validation splits were used.")
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
