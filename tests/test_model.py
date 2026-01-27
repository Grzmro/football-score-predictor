import pytest
import pandas as pd
import numpy as np
from src.model.match_predictor import MatchPredictor

@pytest.fixture
def sample_features_df():
    dates = pd.date_range(start='2023-01-01', periods=10)
    dates_col = np.repeat(dates, 2)
    
    data = {
        'date': dates_col,
        'league': ['L1']*20,
        'season': ['2023']*20,
        'team': ['A', 'B']*10,
        'opponent': ['B', 'A']*10,
        'result': ['Home Win', 'Away Win']*10,
        'goals_for': np.random.randint(0, 3, 20),
        'goals_against': np.random.randint(0, 3, 20),
        'xg': np.random.rand(20),
        'ga': np.random.randint(0, 3, 20),
        'gf': np.random.randint(0, 3, 20)
    }
    df = pd.DataFrame(data)
    
    return df

def test_prepare_training_data(sample_features_df):
    predictor = MatchPredictor()
    
    df = sample_features_df.copy()
    df['result'] = ['W', 'L', 'D', 'W'] * 5
    
    X, y = predictor.prepare_training_data(df)
    
    assert not X.empty
    assert len(y) == len(X)
    assert 'rolling_avg_goals_for_5' in X.columns
    
    assert not y.isna().any()

def test_train_model(sample_features_df):
    predictor = MatchPredictor()
    df = sample_features_df.copy()
    df['result'] = ['W', 'L', 'D', 'W'] * 5
    
    metrics = predictor.train(df, test_size=0.2)
    
    assert predictor.is_trained
    assert 'accuracy' in metrics
    assert 'cv_mean' in metrics

def test_predict_not_trained():
    predictor = MatchPredictor()
    
    pred, proba = predictor.predict(pd.DataFrame())
    assert pred == 0
    assert not proba

def test_predict(sample_features_df):
    predictor = MatchPredictor()
    df = sample_features_df.copy()
    df['result'] = ['W', 'L', 'D', 'W'] * 5
    predictor.train(df)
    
    X, _ = predictor.prepare_training_data(df)
    features_row = X.iloc[[0]]
    
    pred, proba = predictor.predict(features_row)
    
    assert pred in [0, 1, 2]
    assert isinstance(proba, dict)
    assert 'Home Win' in proba
