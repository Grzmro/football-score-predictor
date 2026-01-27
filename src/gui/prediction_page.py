"""
Strona Streamlit do przewidywania wynikow meczy
"""

import streamlit as st
import pandas as pd
import os
from src.loader.loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.model.match_predictor import MatchPredictor
from src.preprocessing.features import add_advanced_features
from src.utils.logger import get_logger

logger = get_logger()


def get_team_latest_stats(df: pd.DataFrame, team_name: str) -> dict:
    """Pobiera najnowsze statystyki czyli rolling features druzyny"""
    team_df = df[df['team'] == team_name].sort_values('date', ascending=False)
    
    if team_df.empty:
        return {}
    
    latest = team_df.iloc[0]
    rolling_cols = [c for c in df.columns if 'rolling_avg' in c]
    
    return {col: latest[col] for col in rolling_cols if col in latest.index}

def get_available_leagues(raw_dir: str) -> list:
    """Zwraca liste dostepnych lig z plikow z meczami"""
    if not os.path.exists(raw_dir):
        return []
    
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    match_prefixes = ['passing_', 'defense_', 'shots_', 'possession_']
    
    leagues = set()
    for f in files:
        if 'season_stats' in f:
            continue
        
        is_match_file = any(f.startswith(prefix) for prefix in match_prefixes)
        if not is_match_file:
            continue
        
        parts = f.split('_')
        if len(parts) >= 3:
            league_parts = parts[1:-1]
            if league_parts:
                leagues.add('_'.join(league_parts))
    
    return sorted(leagues)


def show_prediction_page():
    """Wyswietla strone do przewidywania meczy"""
    
    st.header("Predykcja Wyników Meczów")
    
    from src.utils.config import RAW_DIR, MODELS_DIR
    
    if not os.path.exists(RAW_DIR):
        st.warning("Brak danych. Przejdź do 'Pobieranie Danych'.")
        return
    
    leagues = get_available_leagues(RAW_DIR)
    
    if not leagues:
        st.warning("Brak danych meczowych.")
        return
    
    if 'pred_league' not in st.session_state:
        st.session_state.pred_league = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'match_data' not in st.session_state:
        st.session_state.match_data = None
    
    with st.expander("Konfiguruj model"):
        selected_league = st.selectbox("Wybierz ligę do treningu", leagues, key="train_league")
        
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_{selected_league}.pkl")

    def load_data_for_prediction(league):
        loader = DataLoader(leagues=[league], seasons='all')
        match_df = loader.load_match_data()
        if match_df.empty:
            return None
        prepared = DataPreprocessor.prepare_match_data(match_df, rename_cols=True)
        return add_advanced_features(prepared)

    if st.session_state.pred_league != selected_league or st.session_state.match_data is None:
        with st.spinner("Przygotowywanie modelu i danych..."):
            try:
                prepared_df = load_data_for_prediction(selected_league)
                
                if prepared_df is None or prepared_df.empty:
                    logger.error(f"No data for prediction in league {selected_league}")
                    st.error("Brak danych dla tej ligi.")
                    return

                predictor = MatchPredictor()
                
                if os.path.exists(model_path) and st.session_state.pred_league != selected_league:
                    try:
                        predictor.load_model(model_path)
                        st.info(f"Wczytano zapisany model dla {selected_league}.")
                    except Exception:
                        st.warning("Nie udało się wczytać modelu. Trenowanie nowego...")
                        metrics = predictor.train(prepared_df)
                        predictor.save_model(model_path)
                        st.success(f"Model wytrenowany i zapisany! Accuracy: {metrics['accuracy']:.1%}")
                else:
                    metrics = predictor.train(prepared_df)
                    predictor.save_model(model_path)
                    st.success(f"Model wytrenowany i zapisany! Accuracy: {metrics['accuracy']:.1%}")
                
                st.session_state.predictor = predictor
                st.session_state.match_data = prepared_df
                st.session_state.pred_league = selected_league
                
            except Exception as e:
                st.error(f"Błąd: {str(e)}")
                if os.path.exists(model_path):
                    try: 
                        os.remove(model_path) 
                    except: 
                        pass
                return
    
    if st.session_state.match_data is None:
        return
    
    df = st.session_state.match_data
    teams = sorted(df['team'].unique()) if 'team' in df.columns else []
    
    if not teams:
        st.warning("Brak drużyn.")
        return
    
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        home_team = st.selectbox("Gospodarze", teams, key="home")
    
    with col2:
        st.markdown("<div style='text-align:center;padding-top:30px;font-size:24px'> VS </div>", unsafe_allow_html=True)
    
    with col3:
        away_team = st.selectbox("Goście", [t for t in teams if t != home_team], key="away")
    
    st.markdown("---")
    
    if st.button("Przewiduj", type="primary", use_container_width=True):
        home_stats = get_team_latest_stats(df, home_team)
        away_stats = get_team_latest_stats(df, away_team)
        
        if not home_stats or not away_stats:
            st.error("Brak danych dla wybranych drużyn.")
            return
        
        result, probs = st.session_state.predictor.predict_match(home_stats, away_stats)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{home_team}", f"{probs.get('Home Win', 0):.0%}",
                     delta="OK" if result == "Home Win" else None)
        
        with col2:
            st.metric("Remis", f"{probs.get('Draw', 0):.0%}",
                     delta="OK" if result == "Draw" else None)
        
        with col3:
            st.metric(f"{away_team}", f"{probs.get('Away Win', 0):.0%}",
                     delta="OK" if result == "Away Win" else None)
