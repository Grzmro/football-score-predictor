import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from src.loader.loader import DataLoader, DataSourceError


@pytest.fixture
def mock_data_dir(tmp_path):
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    return str(d)


@pytest.fixture
def sample_match_data():
    return pd.DataFrame({
        'league': ['L1', 'L2'],
        'season': ['2223', '2223'],
        'date': ['2023-01-01', '2023-01-08'],
        'team': ['TeamA', 'TeamB'],
        'opponent': ['TeamB', 'TeamA'],
        'goals': [1, 2]
    })


def test_dataloader_init(mock_data_dir):
    loader = DataLoader(data_dir=mock_data_dir, seasons="2223", leagues="Premier League")
    assert loader.seasons == ["2223"]
    assert loader.leagues == ["Premier League"]
    assert loader.data_dir == mock_data_dir


def test_dataloader_context_manager(mock_data_dir):
    with DataLoader(data_dir=mock_data_dir) as loader:
        assert isinstance(loader, DataLoader)


@patch("os.listdir")
@patch("pandas.read_csv")
def test_load_match_data(mock_read_csv, mock_listdir, mock_data_dir, sample_match_data):
    mock_listdir.return_value = ["shooting_L1_2223.csv", "shooting_L2_2223.csv"]
    mock_read_csv.return_value = sample_match_data

    loader = DataLoader(data_dir=mock_data_dir, seasons=["2223"], leagues=["L1"], stat_types=["shooting"])

    df = loader.load_match_data()

    assert not df.empty
    assert len(df) == 2
    assert 'goals' in df.columns
    mock_read_csv.assert_called_once()


@patch("os.listdir")
def test_load_match_data_no_files(mock_listdir, mock_data_dir):
    mock_listdir.return_value = ["random_file.txt"]

    loader = DataLoader(data_dir=mock_data_dir)
    df = loader.load_match_data()

    assert df.empty


def test_check_season():
    loader = DataLoader(seasons=["2324"])
    assert loader._check_season("file_2324.csv") == True
    assert loader._check_season("file_2223.csv") == False


def test_check_league():
    loader = DataLoader(leagues=["Premier League"])
    assert loader._check_league("file_Premier_League_2223.csv") == True
    assert loader._check_league("file_La_Liga_2223.csv") == False


def test_dataloader_add_with_common_keys():
    """Test merging data with common keys (should use groupby logic)."""
    df1 = pd.DataFrame({
        'league': ['L1'], 
        'season': ['2223'], 
        'date': ['2023-01-01'], 
        'team': ['A'], 
        'opponent': ['B'], 
        'val1': [10]
    })
    df2 = pd.DataFrame({
        'league': ['L1'], 
        'season': ['2223'], 
        'date': ['2023-01-01'], 
        'team': ['A'], 
        'opponent': ['B'], 
        'val2': [20]
    })
    
    loader1 = DataLoader(data_dir="raw")
    loader1.data = df1
    
    loader2 = DataLoader(data_dir="raw")
    loader2.data = df2
    
    combined = loader1 + loader2
    
    assert 'val1' in combined.data.columns
    assert 'val2' in combined.data.columns
    assert len(combined.data) == 1
    assert combined.data.iloc[0]['val1'] == 10
    assert combined.data.iloc[0]['val2'] == 20


def test_dataloader_add_one_empty_data():
    """Test adding one loader with data and one empty."""
    df = pd.DataFrame({'league': ['L1'], 'season': ['2223'], 'date': ['2023-01-01'], 'team': ['A'], 'opponent': ['B'], 'val': [1]})
    loader1 = DataLoader(data_dir="raw")
    loader1.data = df
    
    loader2 = DataLoader(data_dir="raw")
    
    combined = loader1 + loader2
    pd.testing.assert_frame_equal(combined.data, df)