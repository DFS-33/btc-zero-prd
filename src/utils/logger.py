"""Structured logging setup for the Passos Magicos ML pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _load_config_logging() -> dict:
    """Attempt to read logging config from config.yaml without circular imports."""
    try:
        import yaml
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("logging", {})
    except Exception:
        return {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Override log level (e.g. 'DEBUG'). Reads from config.yaml if None.

    Returns:
        Configured Logger instance with a StreamHandler writing to stdout.
    """
    cfg = _load_config_logging()
    resolved_level = level or cfg.get("level", _DEFAULT_LEVEL)
    log_format = cfg.get("format", _DEFAULT_FORMAT)

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))
    logger.propagate = False

    return logger
