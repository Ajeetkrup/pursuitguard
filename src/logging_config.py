"""Centralized, secret-safe CLI logging configuration for PursuitGuard."""

from __future__ import annotations

import logging
import sys

_CONSOLE_HANDLER_NAME = "pursuitguard_console"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure idempotent stderr console logging for the `src` package."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    pkg_logger = logging.getLogger("src")
    pkg_logger.setLevel(numeric)
    pkg_logger.propagate = False

    console_handler = next(
        (h for h in pkg_logger.handlers if h.name == _CONSOLE_HANDLER_NAME),
        None,
    )
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.name = _CONSOLE_HANDLER_NAME
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        pkg_logger.addHandler(console_handler)

    console_handler.setLevel(numeric)
    return pkg_logger
