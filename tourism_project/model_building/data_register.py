"""
Data registration script for the Tourism Wellness Package project.
Loads the dataset, validates expected columns, prints a summary.
"""

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Path:                  {path}")
    print(f"Rows:                  {df.shape[0]}")
    print(f"Columns:               {df.shape[1]}")
    print(f"Expected columns present: {len(EXPECTED_COLUMNS)}/{len(EXPECTED_COLUMNS)}")
    if extra_cols:
        print(f"Unexpected/extra columns found: {extra_cols}")
    print(f"Missing values (total): {int(df.isnull().sum().sum())}")
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print("=" * 60)

    return df


if __name__ == "__main__":
    register_dataset()
