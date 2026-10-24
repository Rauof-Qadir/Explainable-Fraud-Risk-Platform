
import pandas as pd

from src.data.validation import validate_dataframe


def valid_transaction():
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500.0,
        "nameOrig": "C100",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 500.0,
        "nameDest": "C200",
        "oldbalanceDest": 200.0,
        "newbalanceDest": 700.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def test_valid_transaction_passes():
    df = pd.DataFrame([valid_transaction()])
    assert validate_dataframe(df) == []


def test_missing_column_is_detected():
    df = pd.DataFrame([valid_transaction()])
    df = df.drop(columns=["amount"])

    errors = validate_dataframe(df)

    assert any("Missing required columns" in e for e in errors)


def test_missing_value_is_detected():
    row = valid_transaction()
    row["amount"] = float("nan")
    df = pd.DataFrame([row])

    errors = validate_dataframe(df)

    assert any("amount contains missing values" in e for e in errors)


def test_negative_amount_is_detected():
    row = valid_transaction()
    row["amount"] = -100
    df = pd.DataFrame([row])

    errors = validate_dataframe(df)

    assert any("amount contains negative values" in e for e in errors)


def test_invalid_fraud_label_is_detected():
    row = valid_transaction()
    row["isFraud"] = 2
    df = pd.DataFrame([row])

    errors = validate_dataframe(df)

    assert any("isFraud must contain only 0 and 1" in e for e in errors)


def test_unknown_transaction_type_is_detected():
    row = valid_transaction()
    row["type"] = "UNKNOWN"
    df = pd.DataFrame([row])

    errors = validate_dataframe(df)

    assert any("Unknown transaction type" in e for e in errors)


def test_empty_dataframe_is_detected():
    df = pd.DataFrame(columns=sorted([
        "step", "type", "amount", "nameOrig",
        "oldbalanceOrg", "newbalanceOrig", "nameDest",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    ]))

    errors = validate_dataframe(df)

    assert "Dataset is empty" in errors

