"""
Logging Configuration Module for Energy Gnome Library.
"""

import sys

from loguru import Logger, logger


def formatter(record):
    if record["level"].no >= 20:
        return (
            "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: ^8}</level> ] "
            "<level>{message}</level> "
            "\n{exception}"
        )
    else:
        return (
            "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: ^8}</level> > "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> ] "
            "<level>{message}</level> "
            "\n{exception}"
        )


def setup_logging(level="DEBUG", logfile=False) -> Logger:
    """
    Configure the Loguru logger for the Energy Gnome library.

    Returns:
        None
    """
    # Remove the default Loguru logger to avoid duplicate logs
    logger.remove()

    # If tqdm is installed, configure loguru with tqdm.write
    # https://github.com/Delgan/loguru/issues/135
    try:
        from tqdm import tqdm

        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            level=level,
            colorize=True,
            format=formatter,
        )
    except ModuleNotFoundError:
        logger.remove(0)
        logger.add(
            sys.stdout,
            level=level,
            colorize=True,
            format=formatter,
        )
    # Add a file logger that writes to 'energy_gnome.log' with rotation and retention
    if logfile:
        logger.add(
            "energy_gnome.log",
            rotation="10 MB",  # Rotate log after it reaches 10 MB
            level=level,
            colorize=False,
            format=formatter,
        )
    return logger


logger = setup_logging()  # noqa: F401, F811
