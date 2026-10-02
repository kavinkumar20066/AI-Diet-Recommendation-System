import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "nutrition_dataset_v1.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_nutrition.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "linear_regression_model.joblib"
METRICS_PATH = MODEL_DIR / "linear_regression_metrics.json"

# Features and Target
FEATURE_COLUMNS = ['Fat', 'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber']
TARGET_COLUMN = 'Caloric Value'
