"""watchdog.py — Austrade otomatik yeniden başlatma scripti.

Kullanım:
    python watchdog.py

Her 60 saniyede bir bot'un çalışıp çalışmadığını kontrol eder.
Çalışmıyorsa yeniden başlatır. PID dosyası yoksa veya process ölüyse restart.

Windows Task Scheduler'a eklemek için:
    Tetikleyici: Bilgisayar başlatılınca
    Eylem: python C:\\path\\to\\tradebot\\watchdog.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BOT_SCRIPT = Path(__file__).parent / "app.py"
PID_FILE = Path(__file__).parent / "austrade.pid"
CHECK_INTERVAL = 60  # saniye


def _bot_running(pid: int) -> bool:
    """PID'in hâlâ aktif bir process olup olmadığını kontrol et."""
    try:
        # Windows: tasklist ile kontrol
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10,
        )
        return str(pid) in result.stdout
    except Exception:  # noqa: BLE001
        try:
            # Fallback: os.kill(pid, 0) — Unix'te çalışır
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _start_bot() -> int:
    """Bot'u başlat, PID'i döndür."""
    proc = subprocess.Popen(
        [sys.executable, str(BOT_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    pid = proc.pid
    PID_FILE.write_text(str(pid), encoding="utf-8")
    print(f"[Watchdog] Bot başlatıldı — PID: {pid}")
    return pid


def main() -> None:
    print("[Watchdog] Başlatıldı. Kontrol aralığı:", CHECK_INTERVAL, "sn")
    pid: int | None = None

    # Mevcut PID dosyasını oku
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            if not _bot_running(pid):
                print(f"[Watchdog] Eski PID {pid} artık çalışmıyor.")
                pid = None
        except (ValueError, IOError):
            pid = None

    # İlk başlatma
    if pid is None:
        pid = _start_bot()

    while True:
        time.sleep(CHECK_INTERVAL)
        if not _bot_running(pid):
            print("[Watchdog] Bot çökmüş, yeniden başlatılıyor...")
            pid = _start_bot()
        else:
            print(f"[Watchdog] Bot çalışıyor — PID: {pid}")


if __name__ == "__main__":
    main()
