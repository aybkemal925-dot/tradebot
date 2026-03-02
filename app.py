from __future__ import annotations

from austrade.config import load_settings
from austrade.logging_utils import setup_logging
from austrade.ui import run_app


if __name__ == "__main__":
    # debug bayrağını config'den oku → loglama seviyesini ayarla
    _settings = load_settings("config.json")
    setup_logging("austrade.log", debug=_settings.app.debug)
    run_app()
