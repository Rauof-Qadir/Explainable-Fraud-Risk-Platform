from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/transactions.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH, nrows=1000)

    print("\n--- DATASET PREVIEW ---")
    print(df.head())

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES IN FIRST 1000 ROWS ---")
    print(df.isna().sum())

    print("\n--- DATASET SHAPE ---")
    print(f"Rows inspected: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()