class FootballPredictorError(Exception):
    """Klasa bazowa dla błędów w tym module"""
    def __init__(self, message: str = "Wystąpił nieoczekiwany błąd w aplikacji Football Predictor"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"[Football_Score_Predictor] {self.message}"

class InvalidDataError(FootballPredictorError):
    """Wyjątek gdy dane wejściowe są źle sformatowane lub mają zły typ"""
    def __init__(self, message: str = "Dane są nieprawidłowe lub uszkodzone"):
        super().__init__(message)

class DataSourceError(FootballPredictorError):
    """Wyjątek gdy wystąpi problem ze źródłem danych (API, soccerdata)"""
    def __init__(self, message: str = "Nie udało się pobrać danych ze źródła zewnętrznego", original_exception: Exception = None):
        if original_exception:
            message = f"{message} | Oryginalny błąd: {str(original_exception)}"
        super().__init__(message)
        self.original_exception = original_exception

class DataEmptyError(FootballPredictorError):
    """Wyjątek gdy brak danych dla podanych kryteriów"""
    def __init__(self, message: str = "Nie znaleziono danych dla podanych kryteriów"):
        super().__init__(message)

