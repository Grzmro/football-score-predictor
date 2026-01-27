import pytest
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src.analysis.team_analyzer import TeamAnalyzer
from src.analysis.player_analyzer import PlayerAnalyzer

@pytest.fixture
def sample_analysis_df():
    data = {
        'team': ['A', 'A', 'B', 'B'],
        'player': ['P1', 'P2', 'P3', 'P4'],
        'season': ['2223', '2223', '2223', '2223'],
        'xg_per90': [0.5, 0.4, 0.6, 0.2],
        'goals_per90': [0.6, 0.3, 0.5, 0.1],
        'goals': [10, 5, 8, 2],
        'possession_pct': [50, 50, 45, 45],
        'progressive_passes': [100, 80, 70, 40],
        'progressive_carries': [50, 40, 30, 20],
        'npxg_per90': [0.4, 0.3, 0.5, 0.1],
        'goals_assists_non_penalty_per90': [0.5, 0.2, 0.4, 0.1],
        'minutes_90s': [10, 5, 8, 2],
        'age': [25, 22, 28, 30],
        'matches_played': [10, 5, 8, 2],
        'penalty_goals': [1, 0, 1, 0],
        'xag_per90': [0.1, 0.2, 0.1, 0.05],
        'assists_per90': [0.1, 0.1, 0.1, 0.0]
    }
    return pd.DataFrame(data)

def test_team_analyzer_init(sample_analysis_df):
    analyzer = TeamAnalyzer(sample_analysis_df)
    assert analyzer.df.equals(sample_analysis_df)

def test_analyze_efficiency_plotly(sample_analysis_df):
    analyzer = TeamAnalyzer(sample_analysis_df)
    fig = analyzer.analyze_efficiency_plotly(season='2223')
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Analiza Efektywności - Sezon 2223"

def test_analyze_playstyle_plotly(sample_analysis_df):
    analyzer = TeamAnalyzer(sample_analysis_df)
    fig = analyzer.analyze_playstyle_plotly(season='2223')
    assert isinstance(fig, go.Figure)
    assert "Styl Gry" in fig.layout.title.text

def test_player_analyzer_init(sample_analysis_df):
    analyzer = PlayerAnalyzer(sample_analysis_df)
    assert analyzer.df.equals(sample_analysis_df)

def test_plot_player_comparison_plotly(sample_analysis_df):
    analyzer = PlayerAnalyzer(sample_analysis_df)
    fig = analyzer.plot_player_comparison_plotly(player_name='P1', season='2223')
    assert isinstance(fig, go.Figure)
    assert "Porównanie: P1" in fig.layout.title.text

def test_player_pizza_chart_figure(sample_analysis_df):
    analyzer = PlayerAnalyzer(sample_analysis_df)
    fig = analyzer.player_pizza_chart_figure(player_name='P1', season='2223')
    assert isinstance(fig, plt.Figure)
    
def test_plot_aging_curves_figure(sample_analysis_df):
    analyzer = PlayerAnalyzer(sample_analysis_df)
    fig = analyzer.plot_aging_curves_figure()
    assert isinstance(fig, plt.Figure)
