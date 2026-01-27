import os
from pathlib import Path

# Project Paths
# src/utils/config.py -> src/utils -> src -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SOCCERDATA_DIR = os.path.join(DATA_DIR, "soccerdata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# FBRef Column Mapping
FBREF_COLUMN_MAPPING = {
    'league': 'league',
    'season': 'season',
    'team': 'team',
    'players_used': 'players_count',
    'Poss': 'possession_pct',

    'Playing Time': 'matches_played',
    'Playing Time.1': 'starts',
    'Playing Time.2': 'total_minutes',
    'Playing Time.3': 'minutes_90s',

    'Performance': 'goals',
    'Performance.1': 'assists',
    'Performance.2': 'goals_assists',
    'Performance.3': 'goals_non_penalty',
    'Performance.4': 'penalty_goals',
    'Performance.5': 'penalty_attempts',
    'Performance.6': 'yellow_cards',
    'Performance.7': 'red_cards',

    'Expected': 'xg',
    'Expected.1': 'npxg',
    'Expected.2': 'xag',
    'Expected.3': 'npxg_plus_xag',

    'Progression': 'progressive_carries',
    'Progression.1': 'progressive_passes',

    'Per 90 Minutes': 'goals_per90',
    'Per 90 Minutes.1': 'assists_per90',
    'Per 90 Minutes.2': 'goals_assists_per90',
    'Per 90 Minutes.3': 'goals_non_penalty_per90',
    'Per 90 Minutes.4': 'goals_assists_non_penalty_per90',
    'Per 90 Minutes.5': 'xg_per90',
    'Per 90 Minutes.6': 'xag_per90',
    'Per 90 Minutes.7': 'xg_plus_xag_per90',
    'Per 90 Minutes.8': 'npxg_per90',
    'Per 90 Minutes.9': 'npxg_plus_xag_per90',

    'GF': 'goals_for',
    'GA': 'goals_against',

    'Tackles': 'tackles_attempted',
    'Tackles.1': 'tackles_won',
    'Tackles.2': 'tackles_def_3rd',
    'Tackles.3': 'tackles_mid_3rd',
    'Tackles.4': 'tackles_att_3rd',

    'Challenges': 'challenges_won',
    'Challenges.1': 'challenges_attempted',
    'Challenges.2': 'challenges_pct',
    'Challenges.3': 'challenges_lost',

    'Total': 'passes_completed',
    'Total.1': 'passes_attempted',
    'Total.2': 'pass_accuracy_pct',
    'Total.3': 'pass_total_dist',
    'Total.4': 'pass_prog_dist',

    'Ast': 'assists',
    'xAG': 'expected_assisted_goals',
    'xA': 'expected_assists',
    'KP': 'key_passes',
    '1/3': 'passes_to_final_third',
    'PPA': 'passes_into_penalty_area',
    'CrsPA': 'crosses_into_penalty_area',
    'PrgP': 'progressive_passes',

    'Touches': 'touches_total',
    'Touches.1': 'touches_def_pen',
    'Touches.2': 'touches_def_3rd',
    'Touches.3': 'touches_mid_3rd',
    'Touches.4': 'touches_att_3rd',
    'Touches.5': 'touches_att_pen',
    'Touches.6': 'touches_live_ball',

    'Carries': 'carries_total',
    'Carries.1': 'carries_dist',
    'Carries.2': 'carries_prog_dist',
    'Carries.3': 'progressive_carries',
    'Carries.4': 'carries_to_final_third',
    'Carries.5': 'carries_into_penalty_area',
    'Carries.6': 'miscontrols',
    'Carries.7': 'dispossessed',

    'Blocks': 'blocks_total',
    'Blocks.1': 'blocks_shots',
    'Blocks.2': 'blocks_passes',

    'Short': 'pass_short_cmp',
    'Short.1': 'pass_short_att',
    'Short.2': 'pass_short_pct',

    'Medium': 'pass_med_cmp',
    'Medium.1': 'pass_med_att',
    'Medium.2': 'pass_med_pct',

    'Long': 'pass_long_cmp',
    'Long.1': 'pass_long_att',
    'Long.2': 'pass_long_pct',

    'Take-Ons': 'take_ons_att',
    'Take-Ons.1': 'take_ons_succ',
    'Take-Ons.2': 'take_ons_succ_pct',
    'Take-Ons.3': 'take_ons_tackled',
    'Take-Ons.4': 'take_ons_tackled_pct',

    'Receiving': 'passes_received',
    'Receiving.1': 'progressive_passes_received'
}
