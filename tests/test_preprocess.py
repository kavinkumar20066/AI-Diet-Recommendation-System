import pandas as pd
import pytest
from src.preprocessing.preprocess import preprocess_data

def test_preprocess_valid_data():
    data = {
        'food': ['apple', 'apple', 'banana'],
        'Fat': [0.1, 0.1, 0.3],
        'Carbohydrates': [14, 14, 22],
        'Sugars': [10, 10, 12],
        'Protein': [0.3, 0.3, 1.1],
        'Dietary Fiber': [2.4, 2.4, 2.6],
        'Caloric Value': [52, 52, 89]
    }
    df = pd.DataFrame(data)
    df_clean = preprocess_data(df)
    
    assert len(df_clean) == 2, "Duplicates should be removed"
    assert df_clean['Fat'].dtype == float
    assert not df_clean.isnull().any().any(), "No nulls should exist"

def test_preprocess_missing_columns():
    data = {'food': ['apple'], 'Fat': [0.1]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Missing required column"):
        preprocess_data(df)
