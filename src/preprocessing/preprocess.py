import argparse
import sys
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

def inspect_dataset(df):
    print("--- Dataset Inspection ---")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicates:")
    print(df.duplicated().sum())

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        print(f"\nWARNING: Missing expected columns: {missing_cols}")
    else:
        print("\nAll expected feature and target columns are present.")

def preprocess_data(df):
    print("--- Preprocessing ---")
    df_clean = df.drop_duplicates()
    
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df_clean.columns:
            raise ValueError(f"Missing required column: {col}")

    df_clean = df_clean.dropna(subset=required_cols)
    for col in required_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=required_cols)
    print(f"Shape after preprocessing: {df_clean.shape}")
    return df_clean

def main():
    parser = argparse.ArgumentParser(description="Preprocess nutrition dataset.")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect the dataset.")
    args = parser.parse_args()

    if not RAW_DATA_PATH.exists():
        print(f"Error: Dataset not found at {RAW_DATA_PATH}")
        sys.exit(1)

    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    if args.inspect_only:
        inspect_dataset(df)
        return

    df_clean = preprocess_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
