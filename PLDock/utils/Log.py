import os
#import logging
from loguru import logger

def log(log_file: str = None):
    """Log configuration
    Args:
        log_file (str, optional): Path to a log file. Defaults to None.

    Returns:
        loguru.logger class
    """
    #logger = logging.getLogger(__name__)
    #logger.setLevel(level=logging.INFO)
    #formatter = logging.Formatter(
    #    '%(asctime)s - %(process)d - %(pathname)s - %(funcName)s - %(levelname)s - %(message)s')

    if log_file:
        logger.add(log_file,enqueue=True, backtrace=True, diagnose=True)
    #    handler = logging.FileHandler(log_file)
    #    handler.setLevel(logging.INFO)
    #    handler.setFormatter(formatter)
    #    logger.addHandler(handler)
    #console = logging.StreamHandler()
    #console.setLevel(logging.INFO)
    #console.setFormatter(formatter)
    #logger.addHandler(console)
    return logger
