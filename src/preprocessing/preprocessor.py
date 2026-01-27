"""
Moduł zawierający klasę DataPreprocessor do przygotowania danych.
"""

import pandas as pd
import numpy as np
from src.utils.decorators import measure_time
from src.utils.logger import get_logger

logger = get_logger()

class DataPreprocessor:
    """Klasa do przygotowania danych przed analiza"""
    
    @staticmethod
    def rename_columns(df: pd.DataFrame, is_player_data: bool = False) -> pd.DataFrame:
        """
        Zmienia nazwy kolumn z fbref na nasze czytelne
        
        Args:
            df: DataFrame z dziwnymi nazwami
            is_player_data: Czy to dane graczy
        """
        from src.utils.config import FBREF_COLUMN_MAPPING
        rename_dict = FBREF_COLUMN_MAPPING.copy()
        df_cleaned = df.copy()
        
        if is_player_data:
            rename_dict['Age'] = 'age'
            rename_dict['Born'] = 'born'
            rename_dict['Nation'] = 'nation'
            rename_dict['Pos'] = 'pos'
        else:
            rename_dict['Age'] = 'avg_age'
        
        df_cleaned.rename(columns=rename_dict, inplace=True)
        
        cols_to_drop = ['url', 'match_report']
        df_cleaned.drop(columns=[c for c in cols_to_drop if c in df_cleaned.columns], inplace=True)
        
        return df_cleaned
    
    @staticmethod
    @measure_time
    def prepare_team_data(df: pd.DataFrame, rename_cols: bool = False) -> pd.DataFrame:
        """
        Szykuje dane druzyn konwertuje typy i czysci
        """
        try:
            df = df.copy()
            
            if rename_cols:
                df = DataPreprocessor.rename_columns(df, is_player_data=False)
            
            cols_to_convert = df.columns[3:]
            for col in cols_to_convert:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            if 'season' in df.columns:
                df['season'] = df['season'].astype(str).apply(lambda x: x[:-2]).astype('category')
            
            df['team'] = df['team'].astype('category')
            
            return df
        except Exception as e:
            logger.error(f"Błąd podczas przygotowywania danych drużynowych: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @measure_time
    def prepare_player_data(df: pd.DataFrame, rename_cols: bool = False) -> pd.DataFrame:
        """
        Szykuje dane graczy wypelnia braki i konwertuje
        """
        try:
            df = df.copy()
            
            if rename_cols:
                df = DataPreprocessor.rename_columns(df, is_player_data=True)
            
            df[['team', 'nation', 'pos']] = df[['team', 'nation', 'pos']].astype('category')
            
            if 'age' in df.columns:
                df['age'] = df['age'].astype(str).str.split('-').str[0]

            cols_to_exclude = ['player', 'team', 'nation', 'pos', 'season']
            for col in df.columns:
                if col not in cols_to_exclude:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
            
            if 'season' in df.columns:
                df['season'] = df['season'].astype(str).apply(lambda x: x[:-2]).astype('category')
            
            return df
        except Exception as e:
            logger.error(f"Błąd podczas przygotowywania danych zawodników: {e}")
            return pd.DataFrame()

    @staticmethod
    @measure_time
    def prepare_match_data(df: pd.DataFrame, rename_cols: bool = False) -> pd.DataFrame:
        """
        Szykuje dane meczowe
        """
        try:
            df = df.copy()
            
            if rename_cols:
                df = DataPreprocessor.rename_columns(df, is_player_data=False)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            cat_cols = ['home_team', 'away_team', 'team', 'opponent', 'result', 'league', 'comp', 'round', 'venue', 'day', 'time']
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    
            if 'season' in df.columns:
                df['season'] = df['season'].astype(str).apply(lambda x: x[:-2]).astype('category')

            exclude_cols = cat_cols + ['date', 'season']
            for col in df.columns:
                if col not in exclude_cols:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                
            return df
        except Exception as e:
            logger.error(f"Błąd podczas przygotowywania danych: {e}")
            return pd.DataFrame()
