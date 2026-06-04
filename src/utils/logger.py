import logging
import sys


def setup_logger(name: str = "infra_insight") -> logging.Logger:
    """Zentralisiertes Logging-Setup für das gesamte Projekt."""
    logger = logging.getLogger(name)

    # Verhindert doppelte Handler, falls die Funktion mehrfach aufgerufen wird
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Formatierung für lesbare Konsolen-Ausgaben
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # StreamHandler für die Konsole
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger