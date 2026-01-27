# Football Score Predictor

Narzędzie do analizy statystyk piłkarskich oraz przewidywania wyników meczów na podstawie danych historycznych i uczenia maszynowego.

## Uruchamianie

Głównym punktem wejścia do aplikacji jest plik `main.py`, który uruchamia interfejs graficzny oparty na bibliotece **Streamlit**.

```bash
python main.py
```

## Funkcjonalności GUI

Aplikacja oferuje trzy główne moduły:

### 1. Pobieranie Danych

Pozwala na pobieranie najświeższych danych statystycznych z serwisu FBRef (za pośrednictwem biblioteki `soccerdata`). Możesz wybrać ligę (np. `ENG-Premier League`, `ESP-La Liga`) oraz konkretne sezony.

### 2. Analiza Statystyczna

Interaktywny panel analizy danych dla drużyn i zawodników:

- **Klasteryzacja Drużyn**: Grupowanie zespołów o podobnych profilach wydajnościowych za pomocą algorytmu K-Means.
- **Efektywność (Gole vs xG)**: Porównanie rzeczywistej skuteczności z galami oczekiwanymi.
- **Styl Gry**: Analiza zależności między posiadaniem piłki a progresją akcji.
- **Rzuty Karne**: Udział goli z rzutów karnych w całkowitym dorobku drużyny.
- **Profile Zawodników (Pizza Charts)**: Wizualizacja percentylowa statystyk konkretnych graczy.
- **Krzywe Starzenia**: Analiza wpływu wieku na wydajność zawodników w lidze.
- **Ewolucja Drużyn**: Historyczny wykres zmian kluczowych wskaźników zespołu na przestrzeni sezonów.

### 3. Predykcja Meczów

Moduł wykorzystujący model **Random Forest** do przewidywania prawdopodobieństwa:

- Zwycięstwa gospodarzy
- Remisu
- Zwycięstwa gości

Model jest trenowany "w locie" na dostępnych danych historycznych dla wybranej ligi, uwzględniając zaawansowane cechy (rolling averages, formę zespołów).

## Wykorzystanie Programistyczne

### Pobieranie danych (`LeagueLoader`):

```python
from src.loader import LeagueLoader

loader = LeagueLoader()
loader.download_all_data(
    leagues=['ENG-Premier League'],
    seasons=['2324']
)
```

### Ładowanie i łączenie danych (`DataLoader`):

```python
from src.loader import DataLoader

# Ładowanie statystyk podań i obrony dla Premier League
loader = DataLoader(
    seasons=["2223"],
    leagues=["Premier League"],
    stat_types=["passing", "defense"]
)

df = loader.load_data()
```

## Źródło Danych

Aplikacja korzysta z danych FBRef dostarczanych przez [soccerdata](https://soccerdata.readthedocs.io/en/latest/reference/fbref.html).
