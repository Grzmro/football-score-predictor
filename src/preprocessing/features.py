"""
Moduł zawierajacy funcje do inzynierii cech dla modelu predykcji
"""

import pandas as pd
import numpy as np
from src.utils.exceptions import InvalidDataError
from src.utils.logger import get_logger

logger = get_logger()

def calculate_rolling_averages(df: pd.DataFrame, window: int = 5, columns: list = None) -> pd.DataFrame:
    """
    Oblicza srednie kroczace dla podanych kolumn uzywajac numpy i pandasa
    
    Args:
        df: Input DataFrame
        window: Rozmiar okna kroczacego
        columns: Lista kolumn dla ktorych liczymy srednie
        
    Returns:
        DataFrame z nowymi kolumnami rolling_avg_...
        
    Raises:
        InvalidDataError: Jak danych jest za malo (< window size) albo brakuje kolumn
    """
    if len(df) < window:
        raise InvalidDataError(f"Insufficient data: {len(df)} rows, expected at least {window} for rolling calculations.")

    if not columns:
        logger.warning("No columns specified for rolling averages.")
        return df

    df_copy = df.copy()

    for col in columns:
        if col not in df_copy.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping.")
            continue
        
        try:
            if 'team' in df_copy.columns:
                df_copy[f'rolling_avg_{col}_{window}'] = df_copy.groupby('team')[col].transform(
                    lambda x: x.shift().rolling(window, min_periods=1).mean()
                )
            else:
                df_copy[f'rolling_avg_{col}_{window}'] = df_copy[col].shift().rolling(window, min_periods=1).mean()
                 
        except Exception as e:
            logger.error(f"Error calculating rolling avg for {col}: {e}")
            # raise FeatureCalculationError(f"Computation failed for {col}: {e}")

    return df_copy


def calculate_multi_window_rolling(df: pd.DataFrame, columns: list = None, windows: list = [5, 10]) -> pd.DataFrame:
    """
    Wylicza srednie kroczace dla wielu okien naraz
    
    Args:
        df: Input DataFrame
        columns: Lista kolumn do policzenia srednich
        windows: Lista rozmiarow okien do uzycia
        
    Returns:
        DataFrame z nowymi kolumnami dla kazdego rozmiaru okna
    """
    if not columns:
        logger.warning("No columns specified for multi-window rolling.")
        return df
    
    df_copy = df.copy()
    
    for window in windows:
        if len(df_copy) >= window:
            df_copy = calculate_rolling_averages(df_copy, window=window, columns=columns)
        else:
            logger.warning(f"Insufficient data for window size {window}. Skipping.")
    
    return df_copy


def add_xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje statystyki xG czyli overperformance i trendy
    
    Mierzy jak bardzo druzyna gra ponad stan (overperforms) swoje xG
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}

    gf_col = col_map.get('gf') or col_map.get('goals_for')
    xg_col = col_map.get('xg')
    ga_col = col_map.get('ga') or col_map.get('goals_against')
    xga_col = col_map.get('xga') or col_map.get('expected_goals_against')
    
    if gf_col and xg_col:
        df_copy['xg_overperformance'] = df_copy[gf_col] - df_copy[xg_col]

        if 'team' in df_copy.columns:
            df_copy['xg_overperformance_rolling_5'] = df_copy.groupby('team')['xg_overperformance'].transform(
                lambda x: x.shift().rolling(5, min_periods=1).mean()
            )
        else:
            df_copy['xg_overperformance_rolling_5'] = df_copy['xg_overperformance'].shift().rolling(5, min_periods=1).mean()
    else:
        logger.warning("xG columns not found. Skipping xG overperformance features.")
    
    if ga_col and xga_col:

        df_copy['xga_overperformance'] = df_copy[ga_col] - df_copy[xga_col]
        
        if 'team' in df_copy.columns:
            df_copy['xga_overperformance_rolling_5'] = df_copy.groupby('team')['xga_overperformance'].transform(
                lambda x: x.shift().rolling(5, min_periods=1).mean()
            )
        else:
            df_copy['xga_overperformance_rolling_5'] = df_copy['xga_overperformance'].shift().rolling(5, min_periods=1).mean()
    
    return df_copy

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje inforamcje o formie czyli win rate i punkty na mecz
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}
    
    gf_col = col_map.get('gf') or col_map.get('goals_for')
    ga_col = col_map.get('ga') or col_map.get('goals_against')
    
    if not gf_col or not ga_col:
        logger.warning("Goals columns not found. Skipping form features.")

        return df_copy

    if 'match_points' not in df_copy.columns:
        conditions = [
            df_copy[gf_col] > df_copy[ga_col],
            df_copy[gf_col] == df_copy[ga_col],
            df_copy[gf_col] < df_copy[ga_col]
        ]
        points = [3, 1, 0]
        df_copy['match_points'] = np.select(conditions, points)

    df_copy['is_win'] = (df_copy[gf_col] > df_copy[ga_col]).astype(int)
    df_copy['is_draw'] = (df_copy[gf_col] == df_copy[ga_col]).astype(int)
    df_copy['is_loss'] = (df_copy[gf_col] < df_copy[ga_col]).astype(int)
    
    if 'team' in df_copy.columns:
        for window in [5, 10]:
            df_copy[f'win_rate_{window}'] = df_copy.groupby('team')['is_win'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
            df_copy[f'ppg_{window}'] = df_copy.groupby('team')['match_points'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
        
        df_copy['form_momentum'] = df_copy['ppg_5'] - df_copy['ppg_10']
        
        df_copy['unbeaten'] = (df_copy['is_loss'] == 0).astype(int)

        df_copy['unbeaten_streak'] = df_copy.groupby('team')['unbeaten'].transform(
            lambda x: ((x.groupby((x != x.shift()).cumsum()).cumcount() + 1) * x).shift()
        )
    else:
        for window in [5, 10]:
            df_copy[f'win_rate_{window}'] = df_copy['is_win'].shift().rolling(window, min_periods=1).mean()

            df_copy[f'ppg_{window}'] = df_copy['match_points'].shift().rolling(window, min_periods=1).mean()
        
        df_copy['form_momentum'] = df_copy['ppg_5'] - df_copy['ppg_10']
    
    return df_copy




def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje zlozone cechy jak efektywnosc podan czy presja w obronie i kontrola posiadania
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}
    
    pass_acc = col_map.get('pass_accuracy_pct') or col_map.get('total.2')

    prog_pass = col_map.get('progressive_passes') or col_map.get('prgp')

    key_pass = col_map.get('key_passes') or col_map.get('kp')
    
    if pass_acc and prog_pass and key_pass:
        pass_acc_norm = df_copy[pass_acc] / 100
        prog_pass_norm = df_copy[prog_pass] / df_copy[prog_pass].max() if df_copy[prog_pass].max() > 0 else 0
        key_pass_norm = df_copy[key_pass] / df_copy[key_pass].max() if df_copy[key_pass].max() > 0 else 0
        
        df_copy['passing_efficiency'] = (pass_acc_norm * 0.3) + (prog_pass_norm * 0.4) + (key_pass_norm * 0.3)
    else:
        logger.warning(f"Passing columns not found. Available: {list(col_map.keys())[:20]}...")
    
    tackles_won = col_map.get('tackles_won') or col_map.get('tackles.1')

    tackles_att = col_map.get('tackles_attempted') or col_map.get('tackles')
    challenges_won = col_map.get('challenges_won') or col_map.get('challenges')

    blocks_total = col_map.get('blocks_total') or col_map.get('blocks')
    
    if tackles_won and tackles_att and challenges_won and blocks_total:
        tackle_rate = df_copy[tackles_won] / df_copy[tackles_att].replace(0, 1)
        df_copy['defensive_pressure'] = tackle_rate * df_copy[challenges_won] + df_copy[blocks_total]
    else:
        logger.warning("Defensive columns not found for defensive_pressure.")
    
    poss = col_map.get('possession_pct') or col_map.get('poss')

    touches_att = col_map.get('touches_att_3rd') or col_map.get('touches.4')

    touches_total = col_map.get('touches_total') or col_map.get('touches')
    prog_carries = col_map.get('progressive_carries') or col_map.get('prgc')
    
    if poss and touches_att and touches_total and prog_carries:
        poss_norm = df_copy[poss] / 100
        touch_ratio = df_copy[touches_att] / df_copy[touches_total].replace(0, 1)
        prog_carries_norm = df_copy[prog_carries] / df_copy[prog_carries].max() if df_copy[prog_carries].max() > 0 else 0
        
        df_copy['possession_control'] = poss_norm * touch_ratio * (1 + prog_carries_norm)
    else:
        logger.warning("Possession columns not found for possession_control.")
    
    return df_copy


def add_defensive_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje statystyki stabilnosci obrony czyste konta i stracone gole
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}
    
    ga_col = col_map.get('ga') or col_map.get('goals_against')
    
    if not ga_col:
        logger.warning("Goals against column not found. Skipping defensive stability features.")
        return df_copy
    
    if 'clean_sheet' not in df_copy.columns:
        df_copy['clean_sheet'] = (df_copy[ga_col] == 0).astype(int)
    
    if 'team' in df_copy.columns:
        for window in [5, 10]:
            df_copy[f'clean_sheet_rate_{window}'] = df_copy.groupby('team')['clean_sheet'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
            df_copy[f'goals_against_avg_{window}'] = df_copy.groupby('team')[ga_col].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
        
        df_copy['goals_against_trend'] = df_copy['goals_against_avg_5'] - df_copy['goals_against_avg_10']
        
        df_copy['clean_sheet_streak'] = df_copy.groupby('team')['clean_sheet'].transform(
            lambda x: ((x.groupby((x != x.shift()).cumsum()).cumcount() + 1) * x).shift()
        )
    else:
        for window in [5, 10]:
            df_copy[f'clean_sheet_rate_{window}'] = df_copy['clean_sheet'].shift().rolling(window, min_periods=1).mean()
            df_copy[f'goals_against_avg_{window}'] = df_copy[ga_col].shift().rolling(window, min_periods=1).mean()
        
        df_copy['goals_against_trend'] = df_copy['goals_against_avg_5'] - df_copy['goals_against_avg_10']
    
    return df_copy

def add_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje oddzielne staty dla meczy u siebie i na wyjezdzie
    Wymaga kolumny 'venue' z wartosciami 'Home'/'Away'
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}
    
    venue_col = col_map.get('venue')
    gf_col = col_map.get('gf') or col_map.get('goals_for')
    
    if not venue_col:
        logger.warning("Venue column not found. Skipping home/away splits.")
        return df_copy
    
    if not gf_col:
        logger.warning("Goals for column not found. Skipping home/away splits.")
        return df_copy
    
    is_home = df_copy[venue_col].str.lower() == 'home'
    is_away = df_copy[venue_col].str.lower() == 'away'
    
    if 'team' in df_copy.columns:
        df_copy['is_home'] = is_home.astype(int)
        df_copy['is_away'] = is_away.astype(int)
    
    return df_copy

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje wszystkie zaawansowane funkcje do trenowania modelu
    To glowny punkt wejscia dla inzynierii cech
    """
    df_copy = df.copy()
    col_map = {c.lower(): c for c in df_copy.columns}
    
    gf_col = col_map.get('gf') or col_map.get('goals_for')
    ga_col = col_map.get('ga') or col_map.get('goals_against')
    
    if not gf_col or not ga_col:
        logger.warning(f"Goals columns not found. Available: {df_copy.columns.tolist()}")
        logger.warning("Skipping advanced features that depend on goals.")
        return df_copy

    try:
        conditions = [
            df_copy[gf_col] > df_copy[ga_col],
            df_copy[gf_col] == df_copy[ga_col],
            df_copy[gf_col] < df_copy[ga_col]
        ]
        points = [3, 1, 0]
        df_copy['match_points'] = np.select(conditions, points)
        df_copy['goal_diff'] = df_copy[gf_col] - df_copy[ga_col]
        df_copy['clean_sheet'] = (df_copy[ga_col] == 0).astype(int)
        
        df_copy = add_composite_features(df_copy)

        rolling_cols = []
        for col_name in ['goals_for', 'goals_against', 'xg', 'npxg', 'possession_pct',
                         'tackles_won', 'challenges_won', 'blocks_total',
                         'pass_accuracy_pct', 'progressive_passes', 'key_passes',
                         'passing_efficiency', 'defensive_pressure', 'possession_control']:
            
            actual_col = col_map.get(col_name)
            if actual_col:
                rolling_cols.append(actual_col)
            elif col_name in df_copy.columns:
                 rolling_cols.append(col_name)
        
        if rolling_cols:
            df_copy = calculate_multi_window_rolling(df_copy, columns=rolling_cols, windows=[5, 10])
        
        df_copy = add_xg_features(df_copy)
        df_copy = add_form_features(df_copy)
        df_copy = add_defensive_stability_features(df_copy)
        df_copy = add_home_away_splits(df_copy)
        
        logger.info(f"Added {len(df_copy.columns) - len(df.columns)} new features.")
        
    except Exception as e:
        logger.error(f"Error calculating advanced features: {e}")
        # raise FeatureCalculationError(f"Error calculating advanced features: {e}")

    return df_copy
