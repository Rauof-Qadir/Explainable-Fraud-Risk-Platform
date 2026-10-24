
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.features.engineering import engineer_features


MODEL_PATH = Path("models/v2_feature_model.joblib")
VALIDATION_PATH = Path("data/processed/splits/validation.csv")
REPORT_DIR = Path("reports")

GLOBAL_REPORT = REPORT_DIR / "shap_global_importance.csv"
LOCAL_REPORT = REPORT_DIR / "shap_sample_explanation.json"

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "sender_balance_utilization",
    "receiver_balance_relative_amount",
    "sender_balance_insufficient",
]

SAMPLE_SIZE = 500
BACKGROUND_SIZE = 100
RANDOM_SEED = 42


def get_transformed_feature_names(preprocessor):
    """Get feature names after one-hot encoding and scaling."""
    return preprocessor.get_feature_names_out()


def main():
    print("Loading existing V2 model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation data only...")
    df = pd.read_csv(VALIDATION_PATH)
    df = engineer_features(df)

    X = df[FEATURES]

    # Use a reproducible sample to keep computation manageable.
    sample = X.sample(
        n=min(SAMPLE_SIZE, len(X)),
        random_state=RANDOM_SEED,
    )

    background = X.sample(
        n=min(BACKGROUND_SIZE, len(X)),
        random_state=RANDOM_SEED + 1,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    print("Transforming sample and background...")
    X_transformed = preprocessor.transform(sample)
    background_transformed = preprocessor.transform(background)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    feature_names = get_transformed_feature_names(preprocessor)

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=sample.index,
    )

    background_transformed = pd.DataFrame(
        background_transformed,
        columns=feature_names,
        index=background.index,
    )

    print("Calculating SHAP values...")
    # Explain the classifier after preprocessing.
    # The output explains the model's log-odds, not probability directly.
    explainer = shap.LinearExplainer(
        classifier,
        background_transformed,
    )

    shap_values = explainer.shap_values(X_transformed)

    values = np.asarray(shap_values)

    # Some SHAP versions may return an extra output dimension.
    if values.ndim == 3:
        values = values[:, :, -1]

    if values.shape != X_transformed.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"features shape is {X_transformed.shape}"
        )

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(GLOBAL_REPORT, index=False)

    # Explain the highest-scoring transaction in this sample.
    sample_scores = model.predict_proba(sample)[:, 1]
    local_position = int(np.argmax(sample_scores))

    local_values = values[local_position]
    local_row = X_transformed.iloc[local_position]

    local_explanation = {
        "explanation_type": "SHAP values on transformed model inputs",
        "output_space": "classifier log-odds",
        "sample_row_index": str(sample.index[local_position]),
        "predicted_fraud_score": float(sample_scores[local_position]),
        "transformed_feature_values": {
            str(name): float(value)
            for name, value in local_row.items()
        },
        "shap_values": {
            str(name): float(value)
            for name, value in zip(feature_names, local_values)
        },
        "top_positive_contributors": [
            {
                "feature": str(feature_names[i]),
                "shap_value": float(local_values[i]),
            }
            for i in np.argsort(local_values)[::-1]
            if local_values[i] > 0
        ][:10],
        "top_negative_contributors": [
            {
                "feature": str(feature_names[i]),
                "shap_value": float(local_values[i]),
            }
            for i in np.argsort(local_values)
            if local_values[i] < 0
        ][:10],
        "caution": (
            "SHAP describes model behavior, not proof of fraud or causation. "
            "This is one sampled validation transaction, not a test example."
        ),
    }

    LOCAL_REPORT.write_text(
        json.dumps(local_explanation, indent=2),
        encoding="utf-8",
    )

    print("\n--- GLOBAL SHAP IMPORTANCE ---")
    print(importance.head(15).to_string(index=False))

    print("\n--- SAMPLE TRANSACTION EXPLANATION ---")
    print("Validation row index:", sample.index[local_position])
    print("Predicted fraud score:", f"{sample_scores[local_position]:.6f}")

    print("\nTop positive contributors:")
    for item in local_explanation["top_positive_contributors"][:5]:
        print(f"  {item['feature']}: {item['shap_value']:.6f}")

    print("\nTop negative contributors:")
    for item in local_explanation["top_negative_contributors"][:5]:
        print(f"  {item['feature']}: {item['shap_value']:.6f}")

    print("\nSaved global report:", GLOBAL_REPORT.resolve())
    print("Saved local explanation:", LOCAL_REPORT.resolve())
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
