import os
import pandas as pd
import soccerdata as sd
from pathlib import Path
from typing import List, Dict, Union
from src.utils.logger import get_logger
from src.utils.exceptions import DataSourceError, DataEmptyError




logger = get_logger()


from src.utils.config import DATA_DIR, RAW_DIR, PROJECT_ROOT

class LeagueLoader:
    def __init__(self, data_dir: str = str(DATA_DIR)):
        """
        Inicjalizacja LeagueLoader.

        Args:
            data_dir (str): Katalog bazowy dla danych.
        """
        self.data_dir = data_dir
        self.soccerdata_dir = os.path.join(data_dir, "soccerdata")
        self.raw_dir = os.path.join(data_dir, "raw")
        self.loaded_leagues: Dict[str, List[str]] = {}

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.soccerdata_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

    def download_all_data(self, leagues: Union[str, List[str]], seasons: Union[str, List[str]], no_cache: bool = False):
        """
        Pobiera WSZYSTKIE dostępne typy danych dla podanych lig i sezonów.
        To może zająć dużo czasu i wygenerować duże pliki.
        
        Args:
            leagues: Lista lig do pobrania.
            seasons: Lista sezonów do pobrania.
            no_cache: Jeśli True, wymusza nowe pobranie danych ze źródła. Jeśli False, używa danych z cache jeśli dostępne.
        """

        if isinstance(leagues, str): leagues = [leagues]
        if isinstance(seasons, str): seasons = [seasons]

        tasks = [
            ("read_player_season_stats","player_season_stats",{}),
            ("read_team_season_stats", "team_season_stats", {}),
            ("read_team_match_stats", "shots", {"stat_type": "shooting"}),
            ("read_team_match_stats", "passing", {"stat_type": "passing"}),
            ("read_team_match_stats", "defense", {"stat_type": "defense"}),
            ("read_team_match_stats", "possession", {"stat_type": "possession"}),
        ]
        
        for method_name, prefix, kwargs in tasks:
            missing_leagues = []
            missing_seasons = []
            
            pairs_to_fetch = []
            
            for league in leagues:
                for season in seasons:
                    sanitized_league = str(league).replace(" ", "_").replace("-", "_")
                    sanitized_season = str(season).replace("-", "_")
                    filename = f"{prefix}_{sanitized_league}_{sanitized_season}.csv"
                    filepath = os.path.join(self.raw_dir, filename)
                    
                    if no_cache or not os.path.exists(filepath):
                        pairs_to_fetch.append((league, season))
                    else:
                        logger.info(f"Skipping {filename}, already exists.")

            if not pairs_to_fetch:
                print(f"Komunikat: Dane dla {prefix} są już pobrane.")
                continue

            from collections import defaultdict
            league_seasons_map = defaultdict(list)
            for l, s in pairs_to_fetch:
                league_seasons_map[l].append(s)
            
            for league, seas_list in league_seasons_map.items():
                logger.info(f"Downloading {prefix} for {league}: {seas_list}")
                try:
                    fbref = sd.FBref(leagues=[league], seasons=seas_list, data_dir=Path(self.soccerdata_dir), no_cache=no_cache)
                    self._fetch_and_save(fbref, method_name, prefix, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to process {league} {seas_list} for {prefix}: {e}")


    def _fetch_and_save(self, fbref, method_name: str, prefix: str, **kwargs):
        """
        Pomocnicza funkcja do wołania metody soccerdata i zapisywania wyników z podziałem na ligę/sezon.
        """
        logger.info(f"Fetching {prefix} data using {method_name} with args {kwargs}...")
        try:
            method = getattr(fbref, method_name)
            df = method(**kwargs)

            if df.empty:
                logger.warning(f"Data frame for {method_name} is empty. Skipping save.")
                raise DataEmptyError(f"Fetched data for {method_name} is empty.")

            df_reset = df.reset_index()

            if 'league' not in df_reset.columns or 'season' not in df_reset.columns:
                logger.warning(
                    f"Could not find 'league' or 'season' columns in {method_name} result. Saving as single file.")
                self._save_df(df, f"{prefix}_combined.csv")
                return

            grouped = df_reset.groupby(['league', 'season'])

            for (league, season), group in grouped:
                sanitized_league = str(league).replace(" ", "_").replace("-", "_")
                sanitized_season = str(season).replace("-", "_")

                filename = f"{prefix}_{sanitized_league}_{sanitized_season}.csv"
                self._save_df(group, filename)

                if league not in self.loaded_leagues:
                    self.loaded_leagues[league] = []
                if season not in self.loaded_leagues[league]:
                    self.loaded_leagues[league].append(season)

        except Exception as e:
            logger.error(f"Error executing {method_name}: {e}")
            raise DataSourceError(f"Failed to fetch {prefix} data using {method_name}: {e}", original_exception=e)

    def _save_df(self, df: pd.DataFrame, filename: str):
        filepath = os.path.join(self.raw_dir, filename)
        logger.info(f"Saving {len(df)} rows to {filepath}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                df.to_csv(f, index=False)
        except Exception as e:
            logger.error(f"Failed to write to {filepath}: {e}")
            raise DataSourceError(f"Failed to save data to {filepath}: {e}", original_exception=e)


class DataLoader:
    def __init__(self,
                 data_dir: str = str(RAW_DIR),
                 seasons: Union[List[str], str] = None,
                 stat_types: Union[List[str], str] = None,
                 leagues: Union[List[str], str] = None):
        """
        DataLoader do obsługi czytania i łączenia plików CSV.
        Może działać jako Context Manager.
        
        Args:
            data_dir: Katalog z surowymi plikami CSV.
            seasons: Lista sezonów do załadowania lub "all".
            stat_types: Lista typów statystyk do załadowania (np. ['passing', 'defense']) lub "all".
            leagues: Lista lig do załadowania (np. ['Premier League']) lub "all".
        """
        
        if Path(data_dir).is_absolute():
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(str(PROJECT_ROOT), data_dir)
        
        if seasons == "all":
            self.seasons = None
        elif isinstance(seasons, str):
            self.seasons = [seasons]
        else:
            self.seasons = seasons
            
        if stat_types == "all":
            self.stat_types = None
        elif isinstance(stat_types, str):
            self.stat_types = [stat_types]
        else:
            self.stat_types = stat_types

        if leagues == "all":
            self.leagues = None
        elif isinstance(leagues, str):
            self.leagues = [leagues]
        else:
            self.leagues = leagues
            
        self.data: pd.DataFrame = pd.DataFrame()
        self._files_handle = None

    def __enter__(self):
        """Wejście do context managera."""
        logger.info("Entering DataLoader context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Wyjście z context managera."""
        logger.info("Exiting DataLoader context. Cleaning up temporary resources if any.")
        pass

    def load_match_data(self) -> pd.DataFrame:
        """
        Ładuje statystyki meczowe (passing, defense, shots itp.).
        POMIJA dane terminarza i statystyki sezonowe.
        Łączy na podstawie [league, season, date, team, opponent].
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Directory not found: {self.data_dir}")
            return pd.DataFrame()

        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        dfs = []

        logger.info(f"Loading MATCH data from {self.data_dir}. Filters - S: {self.seasons}, T: {self.stat_types}, L: {self.leagues}")

        for f in all_files:
            if "schedule" in f or "season_stats" in f:
                continue

            if not self._check_season(f): continue
            
            if not self._check_stat_type(f): continue

            if not self._check_league(f): continue

            self._load_file(f, dfs)

        if not dfs:
            logger.warning("No MATCH data loaded.")
            self.data = pd.DataFrame()
            return self.data

        mandatory_keys = ['date', 'team', 'opponent']
        valid_dfs = [d for d in dfs if all(k in d.columns for k in mandatory_keys)]
        
        if len(valid_dfs) < len(dfs):
            logger.warning(f"Skipped {len(dfs) - len(valid_dfs)} files lacking match keys {mandatory_keys}.")

        if not valid_dfs:
            return pd.DataFrame()

        join_keys = [c for c in ['league', 'season', 'date', 'team', 'opponent'] if all(c in d.columns for d in valid_dfs)]
        logger.info(f"Merging MATCH data on keys: {join_keys}")
        
        big_concat = pd.concat(valid_dfs, ignore_index=True)
        self.data = big_concat.groupby(join_keys, as_index=False).first()
        

        
        return self.data

    def load_schedule_data(self) -> pd.DataFrame:
        """
        Ładuje TYLKO dane terminarza.
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Dir not found: {self.data_dir}")
            return pd.DataFrame()
        all_files = os.listdir(self.data_dir)
        dfs = []
        
        for f in all_files:
            if "schedule" not in f or not f.endswith(".csv"): continue
            if not self._check_season(f): continue
            if not self._check_league(f): continue
            
            self._load_file(f, dfs)
            

            
        self.data = pd.concat(dfs, ignore_index=True).drop_duplicates()
        return self.data

    def load_team_season_data(self) -> pd.DataFrame:
        """
        Ładuje team_season_stats. Łączy po [league, season, team].
        """
        df = self._load_specialized("team_season_stats", ['league', 'season', 'team'])

        return df


    def load_player_season_data(self) -> pd.DataFrame:
        """
        Ładuje player_season_stats. Łączy po [league, season, team, player].
        """
        df = self._load_specialized("player_season_stats", ['league', 'season', 'team', 'player'])

        return df

    def _load_specialized(self, prefix_marker: str, merge_keys: List[str]) -> pd.DataFrame:
        if not os.path.exists(self.data_dir):
            logger.error(f"Dir not found: {self.data_dir}")
            return pd.DataFrame()
        all_files = os.listdir(self.data_dir)
        dfs = []
        
        for f in all_files:
            if prefix_marker not in f or not f.endswith(".csv"): continue
            if not self._check_season(f): continue
            if not self._check_league(f): continue
            
            self._load_file(f, dfs)
            
        if not dfs: return pd.DataFrame()
        
        valid_dfs = [d for d in dfs if all(k in d.columns for k in merge_keys)]
        if not valid_dfs: return pd.DataFrame()
        
        big_concat = pd.concat(valid_dfs, ignore_index=True)
        self.data = big_concat.groupby(merge_keys, as_index=False).first()
        return self.data

    def _check_season(self, filename: str) -> bool:
        if not self.seasons: return True
        return any(s.replace("-", "_") in filename for s in self.seasons)

    def _check_stat_type(self, filename: str) -> bool:
        if not self.stat_types: return True
        return any(filename.startswith(t) for t in self.stat_types)

    def _check_league(self, filename: str) -> bool:
        if not self.leagues: return True
        for l in self.leagues:
            sanitized = l.replace(" ", "_").replace("-", "_")
            if sanitized in filename: return True
        return False

    def _load_file(self, filename: str, dfs_list: List[pd.DataFrame]):
        path = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_csv(path)
            dfs_list.append(df)
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")



    def __add__(self, other):
        """
        Przeciąża operator + aby połączyć dane z dwóch instancji DataLoader.
        Używa tej samej logiki łączenia (Concat + GroupBy Keys) do łączenia atrybutów.
        """
        if not isinstance(other, DataLoader):
            raise TypeError("Operands must be of type DataLoader")

        new_seasons = list(set((self.seasons or []) + (other.seasons or []))) or None
        new_stats = list(set((self.stat_types or []) + (other.stat_types or []))) or None
        new_leagues = list(set((self.leagues or []) + (other.leagues or []))) or None
        
        new_loader = DataLoader(self.data_dir, seasons=new_seasons, stat_types=new_stats, leagues=new_leagues)
        
        df1 = self.data
        df2 = other.data
        
        if df1.empty and df2.empty:
            return new_loader
        
        if df1.empty: 
            new_loader.data = df2.copy()
            return new_loader
        if df2.empty: 
            new_loader.data = df1.copy()
            return new_loader

        join_keys = [c for c in ['league', 'season', 'date', 'team', 'opponent'] if c in df1.columns and c in df2.columns]
        
        if not join_keys:
            new_loader.data = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        else:
            big_concat = pd.concat([df1, df2], ignore_index=True)
            new_loader.data = big_concat.groupby(join_keys, as_index=False).first()
            
        return new_loader
