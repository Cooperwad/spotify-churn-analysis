"""
clean_data.py

Before running this script, make sure the raw Kaggle dataset is saved here:

    data/raw/spotify_churn_dataset.csv

Run this script from the project root:

    python src/clean_data.py

This script follows the same cleaning and preprocessing workflow used in
notebooks/01_data_cleaning.ipynb:

1. Load the raw Spotify churn dataset.
2. Run a basic audit and validation checks.
3. Standardize categorical text formatting.
4. Save a readable cleaned dataset.
5. Create a model-ready clustering matrix by scaling numeric features and
   one-hot encoding categorical features.
6. Add user_id and is_churned back to the processed file for later analysis,
   but they should still be dropped before fitting clustering models.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spotify_churn_dataset.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_churn_cleaned.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_churn_model_matrix.csv"


# ---------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------

ID_COL = "user_id"
TARGET_COL = "is_churned"

CATEGORICAL_COLS = [
    "gender",
    "country",
    "subscription_type",
    "device_type",
]

NUMERIC_COLS = [
    "age",
    "listening_time",
    "songs_played_per_day",
    "skip_rate",
    "ads_listened_per_week",
]

BINARY_COLS = [
    "offline_listening",
]

CLUSTER_FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def make_one_hot_encoder():
    """
    Create a OneHotEncoder that works across different scikit-learn versions.

    Newer versions use sparse_output=False.
    Older versions use sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_raw_data() -> pd.DataFrame:
    """Load the raw Spotify churn dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_PATH}\n"
            "Place the Kaggle CSV in data/raw/ and name it "
            "spotify_churn_dataset.csv."
        )

    return pd.read_csv(RAW_DATA_PATH)


def run_basic_audit(df_raw: pd.DataFrame) -> None:
    """Print the same basic audit information used in the notebook."""
    print("\n--- Basic Dataset Audit ---")
    print(f"Shape: {df_raw.shape}")
    print("\nData types:")
    print(df_raw.dtypes)

    missing_values = df_raw.isna().sum()
    duplicate_rows = df_raw.duplicated().sum()
    duplicate_user_ids = df_raw[ID_COL].duplicated().sum()

    print("\nMissing values by column:")
    print(missing_values)

    print("\nDuplicate rows:", duplicate_rows)
    print("Duplicate user IDs:", duplicate_user_ids)

    print("\nUnique values by column:")
    print(df_raw.nunique().sort_values())


def run_validation_checks(df: pd.DataFrame) -> pd.Series:
    """Run validation checks for missing, duplicate, invalid, or impossible values."""
    validation_checks = {
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_user_ids": df[ID_COL].duplicated().sum(),
        "age_below_0": (df["age"] < 0).sum(),
        "listening_time_below_0": (df["listening_time"] < 0).sum(),
        "songs_played_per_day_below_0": (df["songs_played_per_day"] < 0).sum(),
        "skip_rate_below_0": (df["skip_rate"] < 0).sum(),
        "skip_rate_above_1": (df["skip_rate"] > 1).sum(),
        "ads_listened_per_week_below_0": (df["ads_listened_per_week"] < 0).sum(),
        "offline_listening_invalid": (~df["offline_listening"].isin([0, 1])).sum(),
        "is_churned_invalid": (~df["is_churned"].isin([0, 1])).sum(),
    }

    validation_results = pd.Series(validation_checks)

    print("\n--- Validation Results ---")
    print(validation_results)

    return validation_results


def standardize_categorical_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightly clean categorical columns.

    The dataset is already mostly clean, so this only trims whitespace,
    preserves missing values, and standardizes capitalization.
    """
    df_clean = df.copy()

    for col in CATEGORICAL_COLS:
        df_clean[col] = df_clean[col].str.strip()

    df_clean["gender"] = df_clean["gender"].str.title()
    df_clean["country"] = df_clean["country"].str.upper()
    df_clean["subscription_type"] = df_clean["subscription_type"].str.title()
    df_clean["device_type"] = df_clean["device_type"].str.title()

    return df_clean


def create_model_matrix(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Create the model-ready matrix for clustering.

    user_id and is_churned are excluded from CLUSTER_FEATURE_COLS. They are added
    back after preprocessing so later notebooks can connect cluster labels to
    users and calculate churn rates by cluster.
    """
    X = df_clean[CLUSTER_FEATURE_COLS].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            ("cat", make_one_hot_encoder(), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    X_processed = preprocessor.fit_transform(X)

    cat_feature_names = (
        preprocessor
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )

    processed_feature_names = NUMERIC_COLS + BINARY_COLS + cat_feature_names
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

    model_df = X_processed_df.copy()
    model_df[ID_COL] = df_clean[ID_COL].values
    model_df[TARGET_COL] = df_clean[TARGET_COL].values

    return model_df


def main() -> None:
    """Run the full cleaning and preprocessing workflow."""
    print("Project root:", PROJECT_ROOT)
    print("Raw data path:", RAW_DATA_PATH)

    df_raw = load_raw_data()
    run_basic_audit(df_raw)

    df = df_raw.copy()
    validation_results = run_validation_checks(df)

    if (validation_results != 0).any():
        print(
            "\nWarning: At least one validation check returned a nonzero value. "
            "Review the validation output before modeling."
        )
    else:
        print(
            "\nAll validation checks returned 0. No major row removal or "
            "imputation is needed."
        )

    df_clean = standardize_categorical_text(df)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nSaved cleaned data to: {CLEAN_DATA_PATH}")

    model_df = create_model_matrix(df_clean)
    model_df.to_csv(MODEL_DATA_PATH, index=False)
    print(f"Saved model-ready data to: {MODEL_DATA_PATH}")

    print("\nFinal outputs:")
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Model matrix shape: {model_df.shape}")


if __name__ == "__main__":
    main()
