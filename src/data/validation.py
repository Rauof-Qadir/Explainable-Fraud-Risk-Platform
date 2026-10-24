
import numpy as np
import pandas as pd

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

NON_NEGATIVE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

BINARY_COLUMNS = ["isFraud", "isFlaggedFraud"]


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return validation errors without modifying the input."""
    errors = []

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        return [f"Missing required columns: {sorted(missing)}"]

    if df.empty:
        errors.append("Dataset is empty")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            errors.append(f"{col} contains missing values")

    for col in NON_NEGATIVE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")

        if values.isna().any():
            errors.append(f"{col} contains invalid numeric values")
            continue

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            errors.append(f"{col} contains non-finite values")
            continue

        if (values < 0).any():
            errors.append(f"{col} contains negative values")

    for col in BINARY_COLUMNS:
        if not df[col].isin([0, 1]).all():
            errors.append(f"{col} must contain only 0 and 1")

    if not df["type"].isin(
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
    ).all():
        errors.append("Unknown transaction type found")

    return errors
