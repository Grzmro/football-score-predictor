"""
Modul z modelami predykcyjnymi do analizy pilkarskiej
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger()


class AnalysisPredictor:
    """Klasa do predykcji i grupowania klastrowania danych"""
    
    def __init__(self):
        """Inicjalizacja predyktora"""
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.features_used = None
    
    def __repr__(self):
        """Reprezentacja tekstowa obiektu"""
        status = "Wytrenowany" if self.kmeans_model else "Niewytrenowany"
        features = f", Cechy: {len(self.features_used)}" if self.features_used else ""
        return f"<AnalysisPredictor: {status}{features}>"
    
    def cluster_teams(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Klasteryzacja druzyn uzywajac kmeans
        
        Args:
            df: DataFrame z danymi drużyn
            features:Lista cech do klasteryzacji
            n_clusters: Liczba klastrów
            random_state: Seed dla powtarzalności wyników
        """
        X = df[features].copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init=10
        )
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        self.features_used = features
        
        df_result = df.copy()
        df_result['cluster'] = clusters
        
        cluster_centers = df_result.groupby('cluster')[features].mean()
        
        return df_result, cluster_centers.values
    
    def predict_cluster(self, df: pd.DataFrame) -> np.ndarray:
        """
        Przewiduje klastery dla nowych danych
        
        Args:
            df: DataFrame z danymi do predykcji
            
        Returns:
            Array z przypisanymi klastrami
        
        Raises:
            ValueError: Jeśli model nie został jeszcze wytrenowany
        """
        if self.kmeans_model is None:
            logger.error("Attempted to predict cluster without training")
            raise ValueError("Model nie został wytrenowany! Wywołaj najpierw cluster_teams()")
        
        if self.features_used is None:
            raise ValueError("Brak informacji o użytych cechach!")
        
        X = df[self.features_used].copy()
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans_model.predict(X_scaled)
    

