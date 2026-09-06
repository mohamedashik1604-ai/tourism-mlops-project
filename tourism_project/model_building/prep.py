"""
Data preparation script for the Tourism Wellness Package project.
Loads the raw dataset, cleans it, and splits it into train/test sets
saved locally as CSV files for the next pipeline stage.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# Columns that carry no predictive signal and should be dropped
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unnecessary columns (ignore errors in case one is already absent)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Fix known data-entry inconsistencies
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns remaining: {list(df.columns)}")

    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve the ~80/20 ProdTaken imbalance in both splits
    )

    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")
    print("Train target distribution:")
    print(y_train.value_counts(normalize=True))
    print("Test target distribution:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    df = load_and_clean()
    split_and_save(df)
