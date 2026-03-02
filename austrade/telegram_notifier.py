from __future__ import annotations

"""Basit Telegram bildirim modülü.

Kurulum gerektirmez — sadece standart kütüphane (urllib) kullanır.
config.json'da telegram.enabled=true, token ve chat_id doldurulunca aktif olur.
"""

import urllib.parse
import urllib.request
from datetime import datetime, timezone
from threading import Thread

from .config import TelegramConfig
from .logging_utils import get_logger

logger = get_logger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig) -> None:
        self.cfg = cfg
        self._last_daily_summary: str | None = None

    def send(self, text: str, blocking: bool = False) -> None:
        """Mesaj gönder. Varsayılan olarak arka planda (non-blocking)."""
        if not self.cfg.enabled:
            return
        if not self.cfg.token or not self.cfg.chat_id:
            return

        if blocking:
            self._send_sync(text)
        else:
            Thread(target=self._send_sync, args=(text,), daemon=True).start()

    def _send_sync(self, text: str) -> None:
        url = _API.format(token=self.cfg.token)
        data = urllib.parse.urlencode({
            "chat_id": self.cfg.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Telegram API returned %s", resp.status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: %s", exc)

    def send_daily_summary(self, equity: float, daily_pnl: float, open_count: int) -> None:
        """Her sabah 08:00 için günlük özet gönder (engine tarafından çağrılır)."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self._last_daily_summary == today:
            return  # Bugün zaten gönderildi
        self._last_daily_summary = today

        sign = "🟢" if daily_pnl >= 0 else "🔴"
        msg = (
            f"📊 <b>Günlük Özet — {today}</b>\n"
            f"Equity: <b>${equity:.2f}</b>\n"
            f"Günlük PNL: {sign} <b>${daily_pnl:.4f}</b>\n"
            f"Açık pozisyon: {open_count}"
        )
        self.send(msg)
