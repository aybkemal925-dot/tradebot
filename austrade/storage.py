from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .models import Position, TradeRecord

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    qty REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_closed_date ON trades(DATE(closed_at));

CREATE TABLE IF NOT EXISTS balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    daily_pnl REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS open_positions (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
"""

OPEN_POSITION_EXTRA_COLUMNS: dict[str, str] = {
    "initial_qty": "REAL NOT NULL DEFAULT 0",
    "tp1_price": "REAL NOT NULL DEFAULT 0",
    "tp1_hit": "INTEGER NOT NULL DEFAULT 0",
    "break_even_price": "REAL NOT NULL DEFAULT 0",
    "trail_callback_pct": "REAL NOT NULL DEFAULT 0",
    "trail_activation_price": "REAL NOT NULL DEFAULT 0",
    "trailing_active": "INTEGER NOT NULL DEFAULT 0",
}


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._cleanup_counter = 0

    def _init_db(self) -> None:
        self.conn.executescript(SCHEMA)
        self._ensure_open_position_columns()
        self.conn.commit()

    def _ensure_open_position_columns(self) -> None:
        cur = self.conn.execute("PRAGMA table_info(open_positions)")
        existing = {str(row["name"]) for row in cur.fetchall()}
        for name, ddl in OPEN_POSITION_EXTRA_COLUMNS.items():
            if name not in existing:
                self.conn.execute(f"ALTER TABLE open_positions ADD COLUMN {name} {ddl}")

    def add_trade(self, trade: TradeRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO trades (
                position_id, symbol, side, entry_price, exit_price, qty,
                pnl_usd, pnl_pct, opened_at, closed_at, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.position_id,
                trade.symbol,
                trade.side,
                trade.entry_price,
                trade.exit_price,
                trade.qty,
                trade.pnl_usd,
                trade.pnl_pct,
                trade.opened_at.isoformat(),
                trade.closed_at.isoformat(),
                trade.reason,
            ),
        )
        self.conn.commit()
        logger.info(
            "Trade saved: pos=%s symbol=%s side=%s pnl=%.4f",
            trade.position_id,
            trade.symbol,
            trade.side,
            trade.pnl_usd,
        )

    def recent_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def today_pnl(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS s FROM trades WHERE DATE(closed_at) = ?",
            (today,),
        )
        row = cur.fetchone()
        return float(row["s"] if row else 0.0)

    def total_pnl(self) -> float:
        cur = self.conn.execute("SELECT COALESCE(SUM(pnl_usd), 0) AS s FROM trades")
        row = cur.fetchone()
        return float(row["s"] if row else 0.0)

    def peak_equity(self) -> float:
        cur = self.conn.execute("SELECT MAX(equity) AS m FROM balance_snapshots")
        row = cur.fetchone()
        return float(row["m"] if row and row["m"] is not None else 0.0)

    def add_snapshot(self, equity: float, daily_pnl: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO balance_snapshots (ts, equity, daily_pnl) VALUES (?, ?, ?)",
            (now, equity, daily_pnl),
        )
        self.conn.commit()
        self._cleanup_counter += 1
        if self._cleanup_counter >= 1000:
            self.cleanup_old_snapshots(days=7)
            self._cleanup_counter = 0

    def cleanup_old_snapshots(self, days: int = 7) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        self.conn.execute("DELETE FROM balance_snapshots WHERE ts < ?", (cutoff,))
        self.conn.commit()

    def equity_curve(self, limit: int = 200) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT ts, equity FROM balance_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows

    def save_position(self, pos: Position) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO open_positions (
                id, symbol, side, qty, entry_price, stop_loss, take_profit, opened_at, status,
                initial_qty, tp1_price, tp1_hit, break_even_price,
                trail_callback_pct, trail_activation_price, trailing_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pos.id,
                pos.symbol,
                pos.side,
                pos.qty,
                pos.entry_price,
                pos.stop_loss,
                pos.take_profit,
                pos.opened_at.isoformat(),
                pos.status,
                pos.initial_qty,
                pos.tp1_price,
                int(pos.tp1_hit),
                pos.break_even_price,
                pos.trail_callback_pct,
                pos.trail_activation_price,
                int(pos.trailing_active),
            ),
        )
        self.conn.commit()

    def update_position(self, pos: Position) -> None:
        self.save_position(pos)

    def update_position_sl(self, pos_id: int, new_sl: float) -> None:
        self.conn.execute(
            "UPDATE open_positions SET stop_loss=? WHERE id=?",
            (new_sl, pos_id),
        )
        self.conn.commit()

    def delete_position(self, pos_id: int) -> None:
        self.conn.execute("DELETE FROM open_positions WHERE id=?", (pos_id,))
        self.conn.commit()

    def load_open_positions(self) -> list[Position]:
        cur = self.conn.execute(
            "SELECT * FROM open_positions WHERE status='open' ORDER BY id"
        )
        positions: list[Position] = []
        for row in cur.fetchall():
            try:
                opened_at = datetime.fromisoformat(row["opened_at"])
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                opened_at = datetime.now(timezone.utc)

            pos = Position(
                id=int(row["id"]),
                symbol=row["symbol"],
                side=row["side"],
                qty=float(row["qty"]),
                entry_price=float(row["entry_price"]),
                stop_loss=float(row["stop_loss"]),
                take_profit=float(row["take_profit"]),
                opened_at=opened_at,
                status=row["status"],
                initial_qty=float(row["initial_qty"] or 0.0),
                tp1_price=float(row["tp1_price"] or 0.0),
                tp1_hit=bool(row["tp1_hit"]),
                break_even_price=float(row["break_even_price"] or 0.0),
                trail_callback_pct=float(row["trail_callback_pct"] or 0.0),
                trail_activation_price=float(row["trail_activation_price"] or 0.0),
                trailing_active=bool(row["trailing_active"]),
            )
            if pos.initial_qty <= 0:
                pos.initial_qty = pos.qty
            if pos.tp1_price <= 0:
                pos.tp1_price = pos.take_profit
            if pos.break_even_price <= 0:
                pos.break_even_price = pos.entry_price
            positions.append(pos)

        logger.info("Loaded %s open positions from DB", len(positions))
        return positions
