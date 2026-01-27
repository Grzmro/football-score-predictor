import pytest
import pandas as pd
import numpy as np
from src.preprocessing.features import (
    calculate_rolling_averages,
    add_advanced_features,
    add_home_away_splits,
    add_form_features
)
from src.utils.exceptions import InvalidDataError

@pytest.fixture
def sample_df():
    data = {
        'team': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'date': pd.date_range(start='2023-01-01', periods=10),
        'goals_for': [1, 2, 1, 3, 0, 2, 1, 0, 1, 2],
        'goals_against': [0, 1, 1, 0, 2, 1, 2, 3, 0, 1],
        'xg': [1.2, 1.5, 0.8, 2.5, 0.5, 1.8, 0.9, 0.4, 1.1, 1.6],
        'xga': [0.5, 1.0, 1.2, 0.3, 1.8, 1.2, 2.0, 2.5, 0.6, 0.9],
        'venue': ['Home', 'Away', 'Home', 'Away', 'Home', 'Away', 'Home', 'Away', 'Home', 'Away']
    }
    return pd.DataFrame(data)

def test_calculate_rolling_averages(sample_df):
    df = calculate_rolling_averages(sample_df, window=3, columns=['goals_for'])
    
    assert 'rolling_avg_goals_for_3' in df.columns
    team_a = df[df['team'] == 'A']
    assert pd.isna(team_a['rolling_avg_goals_for_3'].iloc[0])
    
    assert team_a['rolling_avg_goals_for_3'].iloc[1] == 1.0
    assert team_a['rolling_avg_goals_for_3'].iloc[2] == 1.5

def test_calculate_rolling_averages_insufficient_data():
    short_df = pd.DataFrame({'val': [1, 2]})
    with pytest.raises(InvalidDataError):
        calculate_rolling_averages(short_df, window=5, columns=['val'])

def test_add_home_away_splits(sample_df):
    df = add_home_away_splits(sample_df)
    assert 'is_home' in df.columns
    assert 'is_away' in df.columns
    assert df['is_home'].iloc[0] == 1
    assert df['is_away'].iloc[0] == 0
    assert df['is_home'].iloc[1] == 0

def test_add_form_features(sample_df):
    df = add_form_features(sample_df)
    
    assert 'match_points' in df.columns
    assert 'is_win' in df.columns
    
    assert df['match_points'].iloc[0] == 3
    assert df['is_win'].iloc[0] == 1
    
    assert 'win_rate_5' in df.columns

def test_add_advanced_features(sample_df):
    df = add_advanced_features(sample_df)
    
    assert 'goal_diff' in df.columns
    assert 'clean_sheet' in df.columns
    assert 'xg_overperformance' in df.columns
