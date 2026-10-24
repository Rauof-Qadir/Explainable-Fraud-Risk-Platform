
from pathlib import Path
import json

import pandas as pd

DATA_FILE = Path("data/raw/transactions.csv")
OUTPUT_FILE = Path("reports/step_distribution_report.json")
CHUNK_SIZE = 200_000

# Har step ke transaction aur fraud counts collect karo.
counts = {}

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=["step", "isFraud"],
    chunksize=CHUNK_SIZE,
):
    grouped = chunk.groupby("step")["isFraud"].agg(
        transactions="count",
        fraud_count="sum",
    )

    for step, row in grouped.iterrows():
        step = int(step)

        if step not in counts:
            counts[step] = {
                "transactions": 0,
                "fraud_count": 0,
            }

        counts[step]["transactions"] += int(row["transactions"])
        counts[step]["fraud_count"] += int(row["fraud_count"])

rows = [
    {
        "step": step,
        **values,
        "fraud_rate_percent": (
            100 * values["fraud_count"] / values["transactions"]
        ),
    }
    for step, values in sorted(counts.items())
]

df = pd.DataFrame(rows)

total_rows = int(df["transactions"].sum())
total_steps = len(df)

# Boundaries ko cumulative transaction count se estimate karo.
# Har step ko poora rakha jata hai.
cumulative = df["transactions"].cumsum()

train_target = total_rows * 0.70
validation_target = total_rows * 0.85

train_idx = int((cumulative < train_target).sum())
train_idx = min(max(train_idx, 1), total_steps - 2)

validation_idx = int((cumulative < validation_target).sum())
validation_idx = min(
    max(validation_idx, train_idx + 1),
    total_steps - 1,
)

candidate_boundaries = {
    "train_end_step": int(df.iloc[train_idx - 1]["step"]),
    "validation_end_step": int(df.iloc[validation_idx - 1]["step"]),
}

report = {
    "total_rows": total_rows,
    "unique_steps": total_steps,
    "transactions_per_step": {
        "minimum": int(df["transactions"].min()),
        "median": float(df["transactions"].median()),
        "maximum": int(df["transactions"].max()),
        "first_10_steps": df.head(10).to_dict(orient="records"),
        "last_10_steps": df.tail(10).to_dict(orient="records"),
    },
    "candidate_row_based_boundaries": candidate_boundaries,
    "candidate_split_summary": {},
}

train_end = candidate_boundaries["train_end_step"]
validation_end = candidate_boundaries["validation_end_step"]

masks = {
    "train": df["step"] <= train_end,
    "validation": (
        (df["step"] > train_end)
        & (df["step"] <= validation_end)
    ),
    "test": df["step"] > validation_end,
}

for name, mask in masks.items():
    part = df.loc[mask]
    n = int(part["transactions"].sum())
    fraud = int(part["fraud_count"].sum())

    report["candidate_split_summary"][name] = {
        "min_step": int(part["step"].min()),
        "max_step": int(part["step"].max()),
        "rows": n,
        "row_share_percent": round(100 * n / total_rows, 3),
        "fraud_count": fraud,
        "fraud_rate_percent": round(100 * fraud / n, 6),
    }

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n--- STEP DISTRIBUTION ---")
print("Total rows:", total_rows)
print("Unique steps:", total_steps)
print(
    "Transactions per step — min / median / max:",
    df["transactions"].min(),
    df["transactions"].median(),
    df["transactions"].max(),
)
print("\nFirst 10 steps:")
print(df.head(10).to_string(index=False))
print("\nLast 10 steps:")
print(df.tail(10).to_string(index=False))
print("\nCandidate row-based temporal boundaries:")
print(json.dumps(candidate_boundaries, indent=2))
print("\nCandidate split summary:")
print(json.dumps(report["candidate_split_summary"], indent=2))
print(f"\nReport saved to: {OUTPUT_FILE.resolve()}")
