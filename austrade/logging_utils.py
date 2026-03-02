from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: str = "austrade.log", debug: bool = False) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    level = logging.DEBUG if debug else logging.INFO
    Path(log_path).touch(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
