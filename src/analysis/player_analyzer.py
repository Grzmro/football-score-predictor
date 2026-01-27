"""
Modul z klasa playeranalyzer do analizy graczy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.utils.logger import get_logger

logger = get_logger()

class PlayerAnalyzer:
    """Klasa do analizy statystyk graczy"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicjalizacja analizatora graczy
        
        Args:
            df: DataFrame z danymi graczy (już przygotowany)
        """
        self.df = df
    


    def plot_player_comparison_plotly(self, player_name, season, min_minutes=5):
        """
        Wykres Plotly porownanie gracza z liga w sezonie
        """
        df_plot = self.df[(self.df['minutes_90s'] >= min_minutes) & (self.df['season'] == season)].copy()
        
        if df_plot.empty:
            logger.warning(f"No player data found for comparison: {player_name} in season {season}")
            return None
        df_plot['Color'] = df_plot['player'].apply(lambda x: player_name if x == player_name else 'Inni')
        
        player_row = df_plot[df_plot['player'] == player_name]
        others = df_plot[df_plot['player'] != player_name]
        
        if not player_row.empty:
            df_final = pd.concat([others, player_row])
        else:
            df_final = others

        fig = px.scatter(
            df_final,
            x='npxg_per90',
            y='goals_assists_non_penalty_per90',
            color='Color',
            color_discrete_map={player_name: '#FF0000', 'Inni': '#3498db'},
            hover_name='player',
            hover_data=['team', 'npxg_per90', 'goals_assists_non_penalty_per90'],
            size='minutes_90s',
            size_max=15,
            title=f"Porównanie: {player_name} vs Liga - Sezon {season}",
            labels={
                'npxg_per90': 'Non-Penalty xG / 90',
                'goals_assists_non_penalty_per90': 'G+A (bez karnych) / 90',
                'Color': 'Legenda'
            },
            opacity=0.7
        )
        
        max_val = max(df_final['npxg_per90'].max(), df_final['goals_assists_non_penalty_per90'].max()) * 1.1
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color='Gray', dash='dash'),
            layer='below'
        )
        
        fig.update_layout(height=600)
        return fig


    


    def player_pizza_chart_figure(self, player_name, season):
        """Zwraca figure z pizza chart dla gracza"""
        features = ['goals_per90', 'assists_per90', 'progressive_carries', 
                   'progressive_passes', 'xag_per90']
        
        with plt.style.context("dark_background"):
            season_df = self.df[self.df['season'] == season].copy()
            for feat in features:
                season_df[f'{feat}_pct'] = season_df[feat].rank(pct=True) * 100
            
            player_data = season_df[season_df['player'] == player_name]
            
            fig = plt.figure(figsize=(9, 9))
            fig.patch.set_facecolor('black')
            
            if player_data.empty:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f'Nie znaleziono: {player_name}', 
                    ha='center', va='center', fontsize=14, color='white')
                ax.set_facecolor('black')
                ax.axis('off')
                return fig
            
            values = player_data[[f'{f}_pct' for f in features]].values.flatten().tolist()
            values += values[:1]
            angles = [n / float(len(features)) * 2 * np.pi for n in range(len(features))]
            angles += angles[:1]
            
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor('black')
            
            ax.plot(angles, values, linewidth=3, linestyle='solid', color='#e74c3c')
            ax.fill(angles, values, color='#3498db', alpha=0.4)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, size=11, color='white')
            ax.set_ylim(0, 100)
            ax.set_title(f"Profil: {player_name}\\nSezon {season}", 
                        y=1.08, fontsize=15, fontweight='bold', color='white')
            
            ax.grid(True, alpha=0.3, color='gray')
            ax.spines['polar'].set_color('white')
            ax.tick_params(colors='white')
            
            fig.tight_layout()
            return fig



    def plot_aging_curves_figure(self):
        """Zwraca figure z krzywymi starzenia sie"""
        
        with plt.style.context("dark_background"):
            df_filtered = self.df[(self.df['age'] > 0) & (self.df['age'] < 40)]
            aging_df = df_filtered.groupby('age')[
                ['progressive_passes', 'progressive_carries', 'matches_played']
            ].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(11, 7))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            ax.plot(aging_df['age'], aging_df['progressive_passes'], 
                marker='o', label='Progressive Passes', linewidth=3, markersize=8, color='#e74c3c')
            ax.plot(aging_df['age'], aging_df['progressive_carries'], 
                marker='s', label='Progressive Carries', linewidth=3, markersize=8, color='#3498db')
            ax.plot(aging_df['age'], aging_df['matches_played'], 
                marker='d', label='Matches Played', linewidth=3, markersize=8, color='#2ecc71')
            
            ax.set_title("Krzywe Starzenia Zawodników", fontsize=16, fontweight='bold', pad=20, color='white')
            ax.set_xlabel("Wiek", fontsize=13, color='white')
            ax.set_ylabel("Średnia wartość", fontsize=13, color='white')
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            
            legend = ax.legend(fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
            
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
            fig.tight_layout()
            
            return fig


