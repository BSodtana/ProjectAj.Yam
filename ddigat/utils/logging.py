from __future__ import annotations

import logging
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "ddigat", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_log_level(level: int, names: Optional[list[str]] = None) -> None:
    targets = names or list(_LOGGERS.keys())
    for name in targets:
        if name in _LOGGERS:
            _LOGGERS[name].setLevel(level)

