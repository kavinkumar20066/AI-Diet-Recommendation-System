import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config import PROCESSED_DATA_PATH, MODEL_PATH, METRICS_PATH, FEATURE_COLUMNS, TARGET_COLUMN

def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    valid_idx = X.notnull().all(axis=1) & y.notnull()
    X = X[valid_idx]
    y = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_test, y_pred))

    coefficients = {k: float(v) for k, v in zip(FEATURE_COLUMNS, model.coef_)}
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'coefficients': coefficients,
        'intercept': float(model.intercept_)
    }

    return model, metrics

def main():
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: Processed dataset not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
        sys.exit(1)

    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Training Linear Regression model...")
    model, metrics = train_and_evaluate(df)

    print("--- Model Metrics ---")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")

    print("\n--- Model Coefficients ---")
    print(f"Intercept: {metrics['intercept']:.4f}")
    for feature, coef in metrics['coefficients'].items():
        print(f"{feature}: {coef:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
