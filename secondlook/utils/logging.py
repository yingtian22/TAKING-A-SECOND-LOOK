from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "secondlook", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def setup_file_logging(log_path: str | Path, name: str = "secondlook") -> logging.Logger:
    """Attach a file handler to the project logger."""
    logger = get_logger(name)
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger
