"""
Modul z klasa teamanalyzer do analizy druzyn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.models import AnalysisPredictor
from src.utils.logger import get_logger

logger = get_logger()

class TeamAnalyzer:
    """Klasa do analizy statystyk druzyn"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicjalizacja analizatora druzyn
        
        Args:
            df: DataFrame z danymi drużyn (już przygotowany)
        """
        self.df = df
        self.predictor = AnalysisPredictor()
    


    def analyze_efficiency_plotly(self, season=None):
        """
        Wykres Plotly efektywnosc gole vs xg dla sezonu
        """
        if season is None:
            season = self.df['season'].max()
            
        season_df = self.df[self.df['season'] == season].copy()
        
        if season_df.empty:
            return None

        max_val = max(season_df['xg_per90'].max(), season_df['goals_per90'].max()) * 1.1

        fig = px.scatter(
            season_df,
            x='xg_per90',
            y='goals_per90',
            color='team',
            size='goals',
            hover_name='team',
            hover_data=['goals', 'xg_per90', 'goals_per90'],
            size_max=20,
            title=f"Analiza Efektywności - Sezon {season}",
            labels={
                'xg_per90': 'Expected Goals (xG) / 90',
                'goals_per90': 'Gole / 90',
                'team': 'Drużyna'
            }
        )
        
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color='Red', dash='dash', width=1)
        )
        
        fig.update_layout(
            height=600,
            xaxis=dict(range=[0, max_val]),
            yaxis=dict(range=[0, max_val])
        )
        
        return fig

    def analyze_playstyle_plotly(self, teams=None, season=None):
        """
        Wykres Plotly styl gry posiadanie vs progresja dla sezonu
        """
        if season is None:
            season = self.df['season'].max()
            
        self.df['total_progression'] = self.df['progressive_passes'] + self.df['progressive_carries']
        season_df = self.df[self.df['season'] == season].copy()
        
        if season_df.empty:
            return None

        if teams:
            season_df['Highlight'] = season_df['team'].apply(lambda x: x if x in teams else 'Inne')
            color_col = 'Highlight'
            season_df = season_df.sort_values('Highlight', ascending=False)
        else:
            color_col = 'team'

        fig = px.scatter(
            season_df,
            x='possession_pct',
            y='total_progression',
            color=color_col,
            size='goals_per90',
            hover_name='team',
            size_max=20,
            title=f"Styl Gry: Posiadanie vs Progresja - Sezon {season}",
            labels={
                'possession_pct': 'Posiadanie piłki (%)',
                'total_progression': 'Suma progresywnych podań i rajdów',
                'team': 'Drużyna',
                'goals_per90': 'Gole/90'
            }
        )
        
        avg_prog = season_df['total_progression'].mean()
        avg_poss = season_df['possession_pct'].mean()
        
        fig.add_hline(y=avg_prog, line_dash="dash", line_color="gray", annotation_text="Średnia Progresja")
        fig.add_vline(x=avg_poss, line_dash="dash", line_color="gray", annotation_text="Średnie Posiadanie")
        
        fig.update_layout(height=600)
        
        return fig

    def plot_cluster_scatter_plotly(self, season=None):
        """
        Wykres Plotly klasteryzacja druzyn dla sezonu
        Wymaga wcześniejszego uruchomienia cluster_teams().
        """
        clustered_data = self.df[self.df['cluster'].notna()].copy()
        
        if clustered_data.empty:
            return None
            
        if season is None:
            season = clustered_data['season'].max()
            
        season_df = clustered_data[clustered_data['season'] == season]
        
        if season_df.empty:
            return None
            
        season_df['cluster'] = season_df['cluster'].astype(int).astype(str)
        
        fig = px.scatter(
            season_df,
            x='progressive_carries',
            y='xag_per90',
            color='cluster',
            hover_name='team',
            size='possession_pct',
            size_max=15,
            title=f"Klasteryzacja Drużyn - Sezon {season}",
            labels={
                'progressive_carries': 'Progressive Carries',
                'xag_per90': 'Expected Assisted Goals (xAG) / 90',
                'cluster': 'Grupa (Klaster)'
            }
        )
        
        fig.update_layout(height=600)
        return fig
    
    def analyze_team_evolution_plotly(self, team, metrics=None):
        """
        Wykres Plotly ewolucja statystyk druzyny przez sezony
        """
        if metrics is None:
            metrics = ['progressive_carries', 'xg_per90', 'yellow_cards']
            
        team_data = self.df[self.df['team'] == team].copy()
        
        if team_data.empty:
            return None
            
        df_evolution = team_data.groupby('season')[metrics].mean().reset_index()
        
        df_evolution['season'] = df_evolution['season'].astype(str)
        
        df_norm = df_evolution.copy()
        for m in metrics:
            max_val = df_norm[m].max()
            if max_val > 0:
                df_norm[m] = df_norm[m] / max_val
                
        df_melted = df_norm.melt(id_vars=['season'], value_vars=metrics, 
                                var_name='Metric', value_name='Normalized Value')
        
        fig = px.line(
            df_melted,
            x='season',
            y='Normalized Value',
            color='Metric',
            markers=True,
            title=f"Ewolucja Stylu: {team}",
            labels={
                'season': 'Sezon',
                'Normalized Value': 'Wartość znormalizowana (0-1)',
                'Metric': 'Metryka'
            }
        )
        
        fig.update_xaxes(type='category', categoryorder='category ascending')
        fig.update_layout(height=500)
        return fig


    

    
    def cluster_teams(self, n_clusters=3, features=None, season=None):
        """
        Klasteryzacja druzyn na podstawie wybranych cech
        
        Args:
            n_clusters: Liczba klastrów
            features: Lista cech do klasteryzacji (None = domyślne)
            season: Konkretny sezon do analizy (None = najnowszy sezon)
            
        Returns:
            DataFrame z dodaną kolumną 'cluster' (tylko dla wybranego sezonu)
        """
        if features is None:
            features = ['possession_pct', 'progressive_carries', 'xag_per90', 'yellow_cards']
        
        if season is None:
            available_seasons = sorted(self.df['season'].unique(), reverse=True)
            season = available_seasons[0]
            print(f"Wybrano najnowszy sezon: {season}")
        
        season_df = self.df[self.df['season'] == season].copy()
        
        if season_df.empty:
            print(f"Brak danych dla sezonu {season}")
            return self.df
        
        season_df, cluster_centers = self.predictor.cluster_teams(
            season_df, 
            features=features, 
            n_clusters=n_clusters
        )
        


        print()
        
        self.df.loc[self.df['season'] == season, 'cluster'] = season_df['cluster'].values
        
        return season_df
    


    def analyze_penalties_figure(self, season):
        """Zwraca figure z analiza karnych"""
        with plt.style.context("dark_background"):
            self.df['penalty_share'] = (self.df['penalty_goals'] / self.df['goals'] * 100).fillna(0)
            season_df = self.df[self.df['season'] == season].copy()
            df_sorted = season_df.sort_values('penalty_share', ascending=False)
            
            chart_height = max(len(df_sorted) * 0.35, 8)
            fig, ax = plt.subplots(figsize=(12, chart_height))
            
            bars = ax.barh(df_sorted['team'], df_sorted['penalty_share'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))),
                        edgecolor='white', linewidth=0.5)
            
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                if row['penalty_share'] > 0:
                    ax.text(row['penalty_share'] + 0.5, i, f'{row["penalty_share"]:.1f}%',
                        va='center', fontsize=10, fontweight='bold', color='white')
            
            ax.set_title(f'Udział Rzutów Karnych - Sezon {season}', fontsize=16, fontweight='bold', pad=20, color='white')
            ax.set_xlabel('Procent całkowitej liczby goli (%)', fontsize=13, color='white')
            ax.set_ylabel('Drużyna', fontsize=13, color='white')
            ax.tick_params(colors='white')
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
            
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            fig.tight_layout()
            
            return fig
