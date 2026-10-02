import pandas as pd
from src.models.linear_regression import train_and_evaluate

def test_train_and_evaluate():
    data = {
        'Fat': [0.1, 0.3, 10.0, 5.0, 2.0, 3.0, 4.0, 1.0, 0.5, 0.2],
        'Carbohydrates': [14, 22, 0, 5, 10, 15, 20, 25, 30, 35],
        'Sugars': [10, 12, 0, 2, 5, 7, 10, 12, 15, 18],
        'Protein': [0.3, 1.1, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2],
        'Dietary Fiber': [2.4, 2.6, 0, 1, 2, 3, 4, 5, 6, 7],
        'Caloric Value': [52, 89, 250, 120, 80, 90, 110, 130, 140, 150]
    }
    df = pd.DataFrame(data)
    model, metrics = train_and_evaluate(df)
    
    assert model is not None
    assert 'MAE' in metrics
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    assert 'coefficients' in metrics
    assert len(metrics['coefficients']) == 5
