import logging

def setup_logger(name='app_logger', log_file='app.log', level=logging.INFO):
    """konfiguruje loggera zeby pisal do pliku i konsoli"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name='app_logger'):
    """zwraca instancje loggera"""
    return setup_logger(name)
