# Football Score Predictor

A tool for analyzing football statistics and predicting match results based on historical data and machine learning.

## Usage

The main entry point for the application is the `main.py` file, which launches the graphical interface based on the **Streamlit** library.

```bash
python main.py
```

## GUI Features

The application offers three main modules:

### 1. Data Download

Allows downloading the latest statistical data from FBRef (via the `soccerdata` library). You can select a league (e.g., `ENG-Premier League`, `ESP-La Liga`) and specific seasons.

### 2. Statistical Analysis

Interactive data analysis dashboard for teams and players:

- **Team Clustering**: Grouping teams with similar performance profiles using K-Means algorithm.
- **Efficiency (Goals vs xG)**: Comparison of actual scoring efficiency against expected goals.
- **Play Style**: Analysis of the relationship between ball possession and action progression.
- **Penalties**: Share of penalty goals in the team's total scoring.
- **Player Profiles (Pizza Charts)**: Percentile visualization of specific player statistics.
- **Aging Curves**: Analysis of the impact of age on player performance in the league.
- **Team Evolution**: Historical chart of key team metrics changes over seasons.

### 3. Match Prediction

Module using a **Random Forest** model to predict the probability of:

- Home Win
- Draw
- Away Win

## Programmatic Usage

### Downloading Data (`LeagueLoader`):

```python
from src.loader import LeagueLoader

loader = LeagueLoader()
loader.download_all_data(
    leagues=['ENG-Premier League'],
    seasons=['2324']
)
```

### Loading and Merging Data (`DataLoader`):

```python
from src.loader import DataLoader

# Loading passing and defense statistics for Premier League
loader = DataLoader(
    seasons=["2223"],
    leagues=["Premier League"],
    stat_types=["passing", "defense"]
)

df = loader.load_data()
```

## Data Source

The application uses FBRef data provided by [soccerdata](https://soccerdata.readthedocs.io/en/latest/reference/fbref.html).
