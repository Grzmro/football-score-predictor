import time
from functools import wraps

from src.utils.logger import get_logger

logger = get_logger()

def measure_time(func):
    """dekorator do mierzenia czasu wykonania funkcji"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"Function {func.__name__} failed after {end_time - start_time:.4f} seconds. Error: {str(e)}")
            raise e
    return wrapper
