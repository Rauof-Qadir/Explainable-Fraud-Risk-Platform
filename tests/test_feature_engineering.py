
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import engineer_features


def sample_data():
    return pd.DataFrame({
        "amount": [50.0, 200.0, 0.0],
        "oldbalanceOrg": [100.0, 100.0, 0.0],
        "oldbalanceDest": [25.0, 0.0, 0.0],
    })


def test_expected_features_are_created():
    result = engineer_features(sample_data())

    expected = {
        "log_amount",
        "sender_balance_utilization",
        "receiver_balance_relative_amount",
        "sender_balance_insufficient",
    }

    assert expected.issubset(result.columns)


def test_log_amount_is_correct():
    result = engineer_features(sample_data())

    assert result.loc[0, "log_amount"] == pytest.approx(
        np.log1p(50.0)
    )


def test_ratios_stay_between_zero_and_one():
    result = engineer_features(sample_data())

    for column in [
        "sender_balance_utilization",
        "receiver_balance_relative_amount",
    ]:
        assert result[column].between(0, 1).all()


def test_insufficient_balance_indicator():
    result = engineer_features(sample_data())

    assert result["sender_balance_insufficient"].tolist() == [
        0, 1, 0
    ]


def test_original_dataframe_is_not_modified():
    original = sample_data()
    before = original.copy(deep=True)

    engineer_features(original)

    pd.testing.assert_frame_equal(original, before)


def test_missing_column_is_rejected():
    df = sample_data().drop(columns=["amount"])

    with pytest.raises(ValueError, match="Missing required columns"):
        engineer_features(df)


def test_negative_amount_is_rejected():
    df = sample_data()
    df.loc[0, "amount"] = -10.0

    with pytest.raises(ValueError, match="negative values"):
        engineer_features(df)
