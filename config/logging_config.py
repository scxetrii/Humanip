"""
config/logging_config.py
────────────────────────
Sets up Loguru as the application-wide logger.
Call configure_logging() once at startup (done in backend/main.py and
frontend/app.py).  Any module can then do:

    from loguru import logger
    logger.info("hello")
"""
from __future__ import annotations

import sys

from loguru import logger

from config.settings import settings


def configure_logging() -> None:
    """Remove default Loguru sink and add our own configured sinks."""
    logger.remove()  # wipe the default stderr sink

    # ── Console sink ──────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=settings.debug,
        diagnose=settings.debug,
    )

    # ── File sink ─────────────────────────────────────────────────────────────
    logger.add(
        str(settings.log_file),
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        serialize=False,
        backtrace=True,
        diagnose=False,   # never log locals to file (security)
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

    logger.info(
        "Logging configured | level={} | file={}",
        settings.log_level,
        settings.log_file,
    )
