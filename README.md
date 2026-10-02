# AI Diet Recommendation System

## Overview
An educational Machine Learning project for analyzing nutrition data and building an AI-assisted diet recommendation system (note: this is an educational project and does not constitute medical advice).

## Problem Statement
Manual calorie and nutrition calculation is tedious. This project explores building an ML-assisted system to analyze nutritional content and predict caloric values based on macronutrients, with a long-term goal of personalized diet recommendations.

## Objectives
- **Nutrition Data Analysis:** Clean and process raw nutrition data.
- **Calorie Prediction:** Train a Linear Regression model to predict caloric value based on macronutrients.
- **ML Experimentation:** Build an end-to-end ML pipeline.
- **Personalized Nutrition Calculations:** (Planned) Calculate BMI/BMR/TDEE.
- **Future Diet Recommendations:** (Planned) Recommend optimal foods based on user profiles.

## Dataset
- **Status:** Available
- **Size:** 2395 rows × 35 columns
- **Important Columns:** `Fat`, `Carbohydrates`, `Sugars`, `Protein`, `Dietary Fiber`, and `Caloric Value` (Target).
- **Source:** *Unknown* (The dataset file `nutrition_dataset_v1.csv` should be placed in `data/raw/`).

## Data Preprocessing
- Duplicates removed.
- Required feature columns and target verified.
- Non-numeric values coerced to numeric, and missing rows dropped.

## Machine Learning
- **Features (X):** `Fat`, `Carbohydrates`, `Sugars`, `Protein`, `Dietary Fiber`
- **Target (y):** `Caloric Value`
- **Model:** Linear Regression

### Model Evaluation
- **MAE:** 18.8481
- **MSE:** 23599.6974
- **RMSE:** 153.6219
- **R2 Score:** 0.8995

**Coefficients & Interpretation:**
- **Intercept:** 4.1975
- **Fat:** 8.9788
- **Carbohydrates:** 3.7403
- **Sugars:** 0.1595
- **Protein:** 4.1501
- **Dietary Fiber:** 0.2962

*Interpretation:* Each coefficient describes the model's learned relationship between that specific macronutrient and the predicted Caloric Value, holding all other features constant. A higher value for Fat, Carbohydrates, and Protein is strongly associated with a higher predicted Caloric Value. This roughly mirrors the known energy density (Fat ~9 kcal/g, Carbs and Protein ~4 kcal/g), confirming the model has learned physically plausible relationships. Note that association in the model does not imply causation.

## Project Structure
```
AI-Diet-Recommendation-System/
├── data/
│   ├── raw/                 # Place nutrition_dataset_v1.csv here
│   └── processed/           # Processed datasets
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_linear_regression.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/
│   ├── models/
│   ├── nutrition/           # (Planned)
│   └── recommendation/      # (Planned)
├── models/                  # Saved models and metrics
├── tests/                   # Pytest tests
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Technologies
- Python 3
- Pandas, NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- Pytest

## Current Status
- **Completed:** Data Preprocessing, Linear Regression Model Training & Evaluation.
- **In Progress:** Testing and documentation.
- **Planned:** 
  - Calorie prediction -> BMI/BMR/TDEE calculation
  - Recommendation logic based on TDEE
  - FastAPI backend integration
  - Frontend development
  - Model deployment

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `nutrition_dataset_v1.csv` inside `data/raw/`.
3. Preprocess data:
   ```bash
   python -m src.preprocessing.preprocess
   ```
4. Train the model:
   ```bash
   python -m src.models.linear_regression
   ```

## Limitations
- **Educational project:** Not medical advice.
- Only one model (Linear Regression) is currently tested.
- Results and metrics depend solely on the provided `nutrition_dataset_v1.csv` dataset.

## Author
KavinKumar S