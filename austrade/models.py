from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["long", "short"]
SignalKind = Literal["bos", "choch", "none", "triple"]


@dataclass(slots=True)
class Signal:
    ts: datetime
    side: Side
    kind: SignalKind
    price: float
    note: str
    score: float = 0.0
    context: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Position:
    id: int
    symbol: str
    side: Side
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    status: str = "open"


@dataclass(slots=True)
class TradeRecord:
    position_id: int
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    opened_at: datetime
    closed_at: datetime
    reason: str
