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
    """Loads and processes data for analysis with caching"""
    # If seasons is a list (from multiselect), used directly. If string (legacy), split it.
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
    st.header("Data Download")
    
    st.warning("""
    This page is used to import new data. 
    This process relies on scraping content, 
    which may take from a few hours to a whole day. Due to the risk of source blocking, 
    please do not perform any operations on this subpage.

    If you only want to check how the program works, 
    go to the Analysis or Prediction section (menu on the left).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Top 5 Leagues
        top_5_leagues = [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1"
        ]
        
        league = st.selectbox("League", top_5_leagues, index=1)
        
        # Generate seasons list (e.g. from 24-25 back to 10-11)
        # Current season assumption: 24-25. 
        # Let's generate last 15 seasons.
        years = range(24, 9, -1) # 24 down to 10
        available_seasons = [f"{y}-{y+1}" for y in years]
        
        seasons = st.multiselect("Seasons", available_seasons, default=["23-24"])
        
        if st.button("Download Data"):
            if not seasons:
                st.error("Please select at least one season.")
            else:
                with st.status("Downloading data...", expanded=True) as status:
                    try:
                        loader = LeagueLoader()
                        st.write(f"Initializing download for {league}...")
                        loader.download_all_data(leagues=[league], seasons=seasons)
                        status.update(label="Download complete!", state="complete", expanded=False)
                        st.success(f"Successfully downloaded data for {league}!")
                        st.rerun()
                    except Exception as e:
                        status.update(label="Download error", state="error")
                        st.error(f"An error occurred: {str(e)}")

    with col2:
        st.subheader("Downloaded Files (data/raw)")
        loader = LeagueLoader()
        raw_dir = loader.raw_dir
        
        if os.path.exists(raw_dir):
            files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
            if files:
                df_files = pd.DataFrame(files, columns=["File Name"])
                st.dataframe(df_files, use_container_width=True, hide_index=True)
            else:
                st.info("No CSV files in data/raw directory.")
        else:
            st.warning("Directory data/raw does not exist.")

def get_available_leagues():
    """Scans data/raw directory to find available leagues"""
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
    st.header("Data Analysis (Team & Player)")
    
    st.sidebar.markdown("### Analysis Settings")
    
    available_leagues = get_available_leagues()
    
    if not available_leagues:
        st.sidebar.warning("No downloaded leagues. Go to Download Data.")
        league = st.sidebar.text_input("League (enter manually)", value="ESP_La_Liga", key="analysis_league")
    else:
        default_ix = 0
        if "ESP_La_Liga" in available_leagues:
            default_ix = available_leagues.index("ESP_La_Liga")
            
        league = st.sidebar.selectbox("Select League", available_leagues, index=default_ix, key="analysis_league")
        
    if st.sidebar.button("Load / Refresh Data"):
        st.cache_data.clear()
        
    with st.spinner("Loading and processing data..."):
        try:
            team_df, player_df = load_analysis_data(league, 'all')
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    if team_df is None or team_df.empty:
        st.warning("No data. Go to 'Download Data' tab and download data for the selected league.")
        return

    tab_team, tab_player = st.tabs(["Team Analysis", "Player Analysis"])
    
    latest_season = sorted(team_df['season'].unique(), reverse=True)[0]
    
    with tab_team:
        st.subheader("Team Analysis")
        selected_season = st.selectbox("Select Season", sorted(team_df['season'].unique(), reverse=True), index=0, key="team_season")
        
        analyzer = TeamAnalyzer(team_df)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("#### Clustering (Group Division)")
            st.caption("Grouping teams with similar performance profiles (e.g., defensive, offensive, balanced).")
            n_clusters = 3
            
            valid_season_df = team_df[team_df['season'] == selected_season]
            if not valid_season_df.empty:
                analyzer.cluster_teams(n_clusters=n_clusters, season=selected_season)
                fig_cluster = analyzer.plot_cluster_scatter_plotly(season=selected_season)
                if fig_cluster:
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.info("No clustering data for selected season.")
            else:
                st.info(f"No data for season {selected_season}")

        with col_t2:
            st.markdown("#### Efficiency (Goals vs xG)")
            st.caption("Comparison of actual goals to expected goals (xG). Above line = above average efficiency.")
            fig_eff = analyzer.analyze_efficiency_plotly(season=selected_season)
            if fig_eff:
                st.plotly_chart(fig_eff, use_container_width=True)
            else:
                st.info("No efficiency data.")
            
        st.markdown("---")
        
        col_t3, col_t4 = st.columns(2)
        
        with col_t3:
            st.markdown("#### Play Style (Possession vs Progression)")
            st.caption("Relationship between ball possession and ability to progress the action.")
            teams_list = sorted(team_df[team_df['season'] == selected_season]['team'].unique())
            selected_teams = st.multiselect("Select teams to highlight", teams_list, default=None)
            
            fig_style = analyzer.analyze_playstyle_plotly(
                teams=selected_teams if selected_teams else None, 
                season=selected_season
            )
            if fig_style:
                st.plotly_chart(fig_style, use_container_width=True)
            else:
                 st.info("No play style data.")
            
        with col_t4:
            st.markdown("#### Penalties")
            st.caption("Percentage of team's total goals scored from penalties.")
            fig_pens = analyzer.analyze_penalties_figure(season=selected_season)
            st.pyplot(fig_pens)
            
        st.markdown("---")
        st.markdown("#### Team Evolution (Multi-season)")
        st.caption("Change in key metrics over the years for the selected team.")
        
        all_teams = sorted(team_df['team'].unique())
        evo_team = st.selectbox("Select team for historical analysis", all_teams, key="team_evolution_select")
        
        fig_evo_team = analyzer.analyze_team_evolution_plotly(evo_team)
        if fig_evo_team:
            st.plotly_chart(fig_evo_team, use_container_width=True)
        else:
            st.info("No historical data for this team.")

    with tab_player:
        st.subheader("Individual Analysis")
        
        if player_df is None:
            st.warning("No player data. Download 'Player Season Stats'.")
        else:
            p_analyzer = PlayerAnalyzer(player_df)
            
            col_p_search, col_p_season = st.columns([2, 1])
            with col_p_season:
                 p_season = st.selectbox("Season", sorted(player_df['season'].unique(), reverse=True), key="player_season")
            with col_p_search:
                 players_in_season = sorted(player_df[player_df['season'] == p_season]['player'].unique())
                 default_player = "Robert Lewandowski" if "Robert Lewandowski" in players_in_season else players_in_season[0] if players_in_season else None
                 selected_player = st.selectbox("Select Player", players_in_season, index=players_in_season.index(default_player) if default_player in players_in_season else 0)

            if selected_player:
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.markdown(f"#### Pizza Chart: {selected_player}")
                    st.caption("Player profile against the league (percentiles). Further from center is better.")
                    fig_pizza = p_analyzer.player_pizza_chart_figure(selected_player, p_season)
                    st.pyplot(fig_pizza)
                    
                with col_p2:
                    st.markdown(f"#### League Comparison (npxG vs G+A)")
                    st.caption("Player efficiency (non-penalty) against other players in the same season.")
                    fig_comp = p_analyzer.plot_player_comparison_plotly(selected_player, season=p_season)
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        st.info("No comparison data.")
                
                st.markdown("---")
                
                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    st.markdown("#### Aging Curves (Whole League)")
                    st.caption("Average player performance by age (league-wide analysis).")
                    fig_aging = p_analyzer.plot_aging_curves_figure()
                    st.pyplot(fig_aging)
                    
                
            else:
                st.info("Select a player to see details.")

def main():
    st.sidebar.title("Main Menu")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Data Download", "Analysis & Predictions", "Match Prediction"]
    )
    
    st.sidebar.markdown("---")
    
    if page == "Data Download":
        show_download_page()
    elif page == "Analysis & Predictions":
        show_analysis_page()
    elif page == "Match Prediction":
        show_prediction_page()

if __name__ == "__main__":
    main()
