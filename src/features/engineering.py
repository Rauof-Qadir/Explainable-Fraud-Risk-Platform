
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

NEW_FEATURES = [
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create row-level, pre-transaction features.

    Does not modify the original DataFrame.
    Does not use target labels, post-transaction balances,
    transaction identifiers, or isFlaggedFraud.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    for column in REQUIRED_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise")

        if values.isna().any():
            raise ValueError(f"{column} contains missing values.")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values.")

        if (values < 0).any():
            raise ValueError(f"{column} contains negative values.")

        result[column] = values.astype(float)

    amount = result["amount"]
    sender_balance = result["oldbalanceOrg"]
    receiver_balance = result["oldbalanceDest"]

    # Reduce the skew caused by very large transaction amounts.
    result["log_amount"] = np.log1p(amount)

    # Bounded ratio between 0 and 1.
    result["sender_balance_utilization"] = (
        amount / (sender_balance + amount + 1.0)
    )

    # How large the transaction is relative to the receiver's
    # existing balance plus this transaction.
    result["receiver_balance_relative_amount"] = (
        amount / (receiver_balance + amount + 1.0)
    )

    # A simple indicator that the amount exceeds the sender's
    # recorded starting balance.
    result["sender_balance_insufficient"] = (
        amount > sender_balance
    ).astype("int8")

    return result
