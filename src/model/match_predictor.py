"""
Moduł z modelem do przewidywania wynikow meczy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Optional, Tuple, Dict
from src.utils.logger import get_logger
from src.utils.decorators import measure_time
from src.utils.exceptions import InvalidDataError

logger = get_logger()

from src.preprocessing.features import add_advanced_features


class MatchPredictor:
    """
    Model do przewidywania wynikow meczy pilkarskich
    
    Przewiduje wynik meczu (Home Win / Draw / Away Win) na podstawie
    rolling features opisujacych forme druzyn
    """
    
    RESULT_MAPPING = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    def __init__(self, random_state: int = 42, **kwargs):
        self.scaler = StandardScaler()
        
        rf_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        rf_params.update(kwargs)
        
        self.model = RandomForestClassifier(**rf_params)
        self.feature_columns: List[str] = []
        self.is_trained = False
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Przygotowuje dane do treningu z dataframe meczowego
        Oblicza roznice cech (Team - Opponent) zeby pasowalo do logiki predykcji
        
        Args:
            df: DataFrame z danymi meczowymi (musi miec rolling features)
            
        Returns:
            Tuple (X, y) - cechy i target
        """
        df_clean = add_advanced_features(df)
        df_clean = df_clean.dropna()
        
        rolling_cols = [c for c in df_clean.columns if 'rolling_avg' in c]
        
        if not rolling_cols:
            raise ValueError("Brak kolumn rolling_avg w danych. Użyj prepare_for_model=True w preprocessorze.")
        
        exclude_cols = ['match_points', 'goal_diff', 'clean_sheet', 'gf', 'ga', 'GF', 'GA']
        feature_cols_base = [c for c in rolling_cols if c not in exclude_cols]
        self.feature_columns = feature_cols_base
        
        
        df_opponent = df_clean.copy()
        
        opp_cols_map = {col: f"{col}_opp" for col in feature_cols_base}
        
        
        cols_to_keep = ['date', 'league', 'season', 'team'] + feature_cols_base
        df_opponent = df_opponent[cols_to_keep].rename(columns={'team': 'opponent_lookup'})
        
        df_opponent = df_opponent.rename(columns=opp_cols_map)
        
        
        df_merged = pd.merge(
            df_clean,
            df_opponent,
            left_on=['date', 'league', 'season', 'opponent'],
            right_on=['date', 'league', 'season', 'opponent_lookup'],
            how='inner',
            suffixes=('', '_remove')
        )
        
        X = pd.DataFrame()
        for col in feature_cols_base:
            col_opp = f"{col}_opp"
            if col_opp in df_merged.columns:
                X[col] = df_merged[col] - df_merged[col_opp]
            else:
                X[col] = df_merged[col]
        
        
        if 'result' in df_merged.columns:
            result_map = {'W': 2, 'D': 1, 'L': 0}
            y = df_merged['result'].map(result_map)
        elif 'match_points' in df_merged.columns:
            y = df_merged['match_points'].map({3: 2, 1: 1, 0: 0})
        else:
            logger.error("Brak kolumny 'result' lub 'match_points' do określenia wyniku.")
            # raise ValueError("Brak kolumny 'result' lub 'match_points' do określenia wyniku.")
            return pd.DataFrame(), pd.Series()
        
        if y.isna().any():
            nan_count = y.isna().sum()
            logger.warning(f"Dropping {nan_count} rows with unknown target.")
            y = y.dropna()
            X = X.loc[y.index]
        
        return X, y
    
    @measure_time
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Trenuje model na danych z meczy
        Uzywa podzialu chronologicznego TimeSeriesSplit
        
        Args:
            df: DataFrame z danymi meczowymi
            test_size: Proporcja danych testowych (ostatnie N% meczow)
            
        Returns:
            Dict z metrykami (accuracy, cv_score)
        """
        if 'date' in df.columns:
            df = df.sort_values('date')
            
        X, y = self.prepare_training_data(df)
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Trenowanie modelu na {len(X_train)} przykładach...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv)
        
        metrics = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"Model wytrenowany. Test Accuracy: {accuracy:.3f}, CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return metrics
    
    @measure_time
    def predict(self, features: pd.DataFrame) -> Tuple[int, Dict[str, float]]:
        """
        Przewiduje wynik spotkania
        
        Args:
            features: DataFrame z cechami (rolling features)
            
        Returns:
            Tuple (prediction, probabilities)
        """
        if not self.is_trained:
            logger.error("Model nie został wytrenowany! Wywołaj najpierw train().")
            return 0, {} # Return dummy values or None equivalent
        
        X = features[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        prob_dict = {
            'Away Win': probabilities[0],
            'Draw': probabilities[1],
            'Home Win': probabilities[2]
        }
        
        return prediction, prob_dict
    
    @measure_time
    def predict_match(self, home_team_stats: Dict, away_team_stats: Dict) -> Tuple[str, Dict[str, float]]:
        """
        Przewiduje wynik meczu miedzy dwoma druzynami
        
        Args:
            home_team_stats: Slownik z rolling features druzyny gospodarzy
            away_team_stats: Slownik z rolling features druzyny gosci
            
        Returns:
            Tuple (wynik jako string, prawdopodobienstwa)
        """
        features = {}
        for col in self.feature_columns:
            home_val = home_team_stats.get(col, 0)
            away_val = away_team_stats.get(col, 0)
            features[col] = home_val - away_val if 'diff' not in col else home_val
        
        X = pd.DataFrame([features])
        prediction, probabilities = self.predict(X)
        
        return self.RESULT_MAPPING[prediction], probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Zwraca waznosc poszczegolnych cech
        
        Returns:
            DataFrame z cechami i ich waznoscia
        """
        if not self.is_trained:
            raise ValueError("Model nie został wytrenowany!")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, path: str) -> None:
        """Zapisuje model do pliku"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
        logger.info(f"Model zapisany do {path}")
    
    def load_model(self, path: str) -> None:
        """Wczytuje model z pliku"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model wczytany z {path}")
