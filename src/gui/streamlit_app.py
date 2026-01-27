import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.loader.loader import LeagueLoader, DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.analysis.team_analyzer import TeamAnalyzer
from src.analysis.player_analyzer import PlayerAnalyzer
from src.gui.prediction_page import show_prediction_page
from src.utils.logger import get_logger

logger = get_logger()

st.set_page_config(
    page_title="Football Score Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2ecc71;
        color: white;
        border: none;
    }
    h1, h2, h3 {
        color: #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_analysis_data(league, seasons):
    """Laduje i przetwarza dane do analizy z cachem"""
    if isinstance(seasons, str):
        seasons = [s.strip() for s in seasons.split(',')]
    
    loader = DataLoader(leagues=[league], seasons='all')
    
    team_df_raw = loader.load_team_season_data()
    if team_df_raw.empty:
        logger.warning(f"No team season data loaded for {league}")
        return None, None
        
    team_df = DataPreprocessor.prepare_team_data(team_df_raw, rename_cols=True)
    
    player_df_raw = loader.load_player_season_data()
    if player_df_raw.empty:
        return team_df, None
        
    player_df = DataPreprocessor.prepare_player_data(player_df_raw, rename_cols=True)
    
    return team_df, player_df

def show_download_page():
    st.header("Pobieranie Danych")
    
    st.warning("""
    Ta strona służy do importowania nowych danych. 
    Proces ten opiera się na scrapowaniu treści, 
    co może zająć od kilku godzin do nawet całego dnia. Ze względu na ryzyko blokady źródła, 
    prosimy o niewykonywanie żadnych operacji na tej podstronie.

    Jeśli chcesz jedynie sprawdzić działanie programu, 
    przejdź do sekcji Analiza lub Predykcja (menu po lewej stronie).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Konfiguracja")
        league = st.text_input("Liga (np. ESP-La Liga)", value="ESP-La Liga")
        seasons_input = st.text_input("Sezony (np. 23-24, 22-23)", value="23-24")
        
        if st.button("Pobierz Dane"):
            seasons = [s.strip() for s in seasons_input.split(',')]
            
            with st.status("Pobieranie danych...", expanded=True) as status:
                try:
                    loader = LeagueLoader()
                    st.write(f"Inicjalizowanie pobierania dla {league}...")
                    loader.download_all_data(leagues=[league], seasons=seasons)
                    status.update(label="Pobieranie zakończone!", state="complete", expanded=False)
                    st.success(f"Pomyślnie pobrano dane dla {league}!")
                    st.rerun()
                except Exception as e:
                    status.update(label="Błąd pobierania", state="error")
                    st.error(f"Wystąpił błąd: {str(e)}")

    with col2:
        st.subheader("Pobrane Pliki (data/raw)")
        loader = LeagueLoader()
        raw_dir = loader.raw_dir
        
        if os.path.exists(raw_dir):
            files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
            if files:
                df_files = pd.DataFrame(files, columns=["Nazwa Pliku"])
                st.dataframe(df_files, use_container_width=True, hide_index=True)
            else:
                st.info("Brak plików CSV w katalogu data/raw.")
        else:
            st.warning("Katalog data/raw nie istnieje.")

def get_available_leagues():
    """Skanuje folder data raw zeby znalezc dostepne ligi"""
    loader = LeagueLoader()
    raw_dir = loader.raw_dir
    
    if not os.path.exists(raw_dir):
        return []
        
    files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    leagues = set()
    
    known_prefixes = [
        'defense_', 'passing_', 'player_season_stats_', 
        'possession_', 'schedule_', 'shots_', 'team_season_stats_'
    ]
    
    for f in files:
        for prefix in known_prefixes:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                last_underscore_idx = rest.rfind('_')
                if last_underscore_idx != -1:
                    league_name = rest[:last_underscore_idx]
                    leagues.add(league_name)
                break
                
    return sorted(list(leagues))

def show_analysis_page():
    st.header("Analiza Danych (Team & Player)")
    
    st.sidebar.markdown("### Ustawienia Analizy")
    
    available_leagues = get_available_leagues()
    
    if not available_leagues:
        st.sidebar.warning("Brak pobranych lig. Przejdź do pobierania.")
        league = st.sidebar.text_input("Liga (wpisz ręcznie)", value="ESP_La_Liga", key="analysis_league")
    else:
        default_ix = 0
        if "ESP_La_Liga" in available_leagues:
            default_ix = available_leagues.index("ESP_La_Liga")
            
        league = st.sidebar.selectbox("Wybierz Ligę", available_leagues, index=default_ix, key="analysis_league")
        
    if st.sidebar.button("Załaduj / Odśwież Dane"):
        st.cache_data.clear()
        
    with st.spinner("Ładowanie i przetwarzanie danych..."):
        try:
            team_df, player_df = load_analysis_data(league, 'all')
        except Exception as e:
            st.error(f"Błąd ładowania danych: {str(e)}")
            return

    if team_df is None or team_df.empty:
        st.warning("Brak danych. Przejdź do zakładki 'Pobieranie Danych' i pobierz dane dla wybranej ligi.")
        return

    tab_team, tab_player = st.tabs(["Analiza Drużyn", "Analiza Zawodników"])
    
    latest_season = sorted(team_df['season'].unique(), reverse=True)[0]
    
    with tab_team:
        st.subheader("Analiza Drużynowa")
        selected_season = st.selectbox("Wybierz Sezon", sorted(team_df['season'].unique(), reverse=True), index=0, key="team_season")
        
        analyzer = TeamAnalyzer(team_df)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("#### Klasteryzacja (Podział na grupy)")
            st.caption("Grupowanie drużyn o podobnym profilu (np. defensywne, ofensywne, zbalansowane).")
            n_clusters = 3
            
            valid_season_df = team_df[team_df['season'] == selected_season]
            if not valid_season_df.empty:
                analyzer.cluster_teams(n_clusters=n_clusters, season=selected_season)
                fig_cluster = analyzer.plot_cluster_scatter_plotly(season=selected_season)
                if fig_cluster:
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.info("Brak danych do klasteryzacji dla wybranego sezonu.")
            else:
                st.info(f"Brak danych dla sezonu {selected_season}")

        with col_t2:
            st.markdown("#### Efektywność (Gole vs xG)")
            st.caption("Porównanie goli strzelonych do goli oczekiwanych (xG). Nad linią = ponadprzeciętna skuteczność.")
            fig_eff = analyzer.analyze_efficiency_plotly(season=selected_season)
            if fig_eff:
                st.plotly_chart(fig_eff, use_container_width=True)
            else:
                st.info("Brak danych efektywności.")
            
        st.markdown("---")
        
        col_t3, col_t4 = st.columns(2)
        
        with col_t3:
            st.markdown("#### Styl Gry (Posiadanie vs Progresja)")
            st.caption("Zależność między posiadaniem piłki a umiejętnością przesuwania akcji do przodu.")
            teams_list = sorted(team_df[team_df['season'] == selected_season]['team'].unique())
            selected_teams = st.multiselect("Wybierz drużyny do podświetlenia", teams_list, default=None)
            
            fig_style = analyzer.analyze_playstyle_plotly(
                teams=selected_teams if selected_teams else None, 
                season=selected_season
            )
            if fig_style:
                st.plotly_chart(fig_style, use_container_width=True)
            else:
                 st.info("Brak danych stylu gry.")
            
        with col_t4:
            st.markdown("#### Rzuty Karne")
            st.caption("Jaki procent wszystkich goli danej drużyny stanowiły rzuty karne.")
            fig_pens = analyzer.analyze_penalties_figure(season=selected_season)
            st.pyplot(fig_pens)
            
        st.markdown("---")
        st.markdown("#### Ewolucja Drużyny (Wielosezonowa)")
        st.caption("Zmiana kluczowych współczynników na przestrzeni lat dla wybranej drużyny.")
        
        all_teams = sorted(team_df['team'].unique())
        evo_team = st.selectbox("Wybierz drużynę do analizy historycznej", all_teams, key="team_evolution_select")
        
        fig_evo_team = analyzer.analyze_team_evolution_plotly(evo_team)
        if fig_evo_team:
            st.plotly_chart(fig_evo_team, use_container_width=True)
        else:
            st.info("Brak danych historycznych dla tej drużyny.")

    with tab_player:
        st.subheader("Analiza Indywidualna")
        
        if player_df is None:
            st.warning("Brak danych zawodników. Poberz dane 'Player Season Stats'.")
        else:
            p_analyzer = PlayerAnalyzer(player_df)
            
            col_p_search, col_p_season = st.columns([2, 1])
            with col_p_season:
                 p_season = st.selectbox("Sezon", sorted(player_df['season'].unique(), reverse=True), key="player_season")
            with col_p_search:
                 players_in_season = sorted(player_df[player_df['season'] == p_season]['player'].unique())
                 default_player = "Robert Lewandowski" if "Robert Lewandowski" in players_in_season else players_in_season[0] if players_in_season else None
                 selected_player = st.selectbox("Wybierz Zawodnika", players_in_season, index=players_in_season.index(default_player) if default_player in players_in_season else 0)

            if selected_player:
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.markdown(f"#### Pizza Chart: {selected_player}")
                    st.caption("Profil gracza na tle ligi (percentyle). Im dalej od środka, tym lepiej w danej kategorii.")
                    fig_pizza = p_analyzer.player_pizza_chart_figure(selected_player, p_season)
                    st.pyplot(fig_pizza)
                    
                with col_p2:
                    st.markdown(f"#### Porównanie z Ligą (npxG vs G+A)")
                    st.caption("Efektywność gracza (bez karnych) na tle innych zawodników w tym samym sezonie.")
                    fig_comp = p_analyzer.plot_player_comparison_plotly(selected_player, season=p_season)
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        st.info("Brak danych do porównania.")
                
                st.markdown("---")
                
                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    st.markdown("#### Krzywe Starzenia (Cała Liga)")
                    st.caption("Średnia wydajność zawodników w zależności od wieku (analiza całej ligi).")
                    fig_aging = p_analyzer.plot_aging_curves_figure()
                    st.pyplot(fig_aging)
                    
                
            else:
                st.info("Wybierz zawodnika aby zobaczyć detale.")

def main():
    st.sidebar.title("Menu Główne")
    
    page = st.sidebar.radio(
        "Nawigacja",
        ["Pobieranie Danych", "Analiza i Predykcje", "Predykcja Meczów"]
    )
    
    st.sidebar.markdown("---")
    
    if page == "Pobieranie Danych":
        show_download_page()
    elif page == "Analiza i Predykcje":
        show_analysis_page()
    elif page == "Predykcja Meczów":
        show_prediction_page()

if __name__ == "__main__":
    main()
