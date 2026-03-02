from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import RiskConfig
from .logging_utils import get_logger
from .models import Signal

if TYPE_CHECKING:
    from .models import Position

logger = get_logger(__name__)


@dataclass
class SizingResult:
    qty: float
    stop_loss: float
    take_profit: float


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def _atr(self, df: pd.DataFrame) -> float:
        """ATR hesaplama — numpy ile hızlandırılmış versiyon."""
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        if len(h) < 2:
            return float(h[-1] - l[-1]) if len(h) > 0 else 0.0

        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )
        period = self.cfg.atr_period
        if len(tr) >= period:
            atr_val = float(tr[-period:].mean())
        else:
            atr_val = float(tr.mean()) if len(tr) > 0 else 0.0

        return atr_val if atr_val > 0 else float(h[-1] - l[-1])

    def _volatility_scale(self, df: pd.DataFrame) -> float:
        """ATR/price oranına göre pozisyon büyüklüğü ölçeği: 0.7–1.2"""
        atr = self._atr(df)
        price = float(df["close"].iloc[-1])
        if price <= 0:
            return 1.0
        vol_pct = (atr / price) * 100.0
        if vol_pct > 5.0:
            return 0.7   # yüksek volatilite → küçült
        if vol_pct < 1.0:
            return 1.2   # düşük volatilite → büyüt
        return 1.0

    def size_position(
        self,
        signal: Signal,
        equity: float,
        df: pd.DataFrame,
        open_positions: list[Position] | None = None,
    ) -> SizingResult | None:
        atr = self._atr(df)
        if atr <= 0:
            return None

        lev = max(self.cfg.leverage, 1)
        risk_usd = equity * (self.cfg.risk_per_trade_pct / 100.0)
        stop_distance = atr * self.cfg.atr_stop_mult
        if stop_distance <= 0:
            return None

        # Leverage dahil: risk_usd = qty * stop_distance * lev
        qty = risk_usd / (stop_distance * lev)

        # Volatilite ölçeği uygula
        qty *= self._volatility_scale(df)

        # Multi-position margin takibi: kullanılan margin'i düş
        used_margin = sum(
            (p.qty * p.entry_price) / lev
            for p in (open_positions or [])
        )
        available_margin = max(0.0, equity - used_margin)
        max_notional = available_margin * lev
        if qty * signal.price > max_notional:
            logger.warning(
                "Position qty capped by available margin: %.6f → %.6f "
                "(equity=%.2f used_margin=%.2f lev=%sx)",
                qty, max_notional / signal.price, equity, used_margin, lev,
            )
            qty = max_notional / signal.price if signal.price > 0 else 0.0

        if signal.side == "long":
            sl = signal.price - stop_distance
            tp = signal.price + stop_distance * self.cfg.target_rr
        else:
            sl = signal.price + stop_distance
            tp = signal.price - stop_distance * self.cfg.target_rr

        # SL/TP doğrulama
        if signal.side == "long":
            if sl <= 0 or sl >= signal.price:
                logger.error("Invalid long SL: %.4f >= entry %.4f", sl, signal.price)
                return None
            if tp <= signal.price:
                logger.error("Invalid long TP: %.4f <= entry %.4f", tp, signal.price)
                return None
        else:
            if sl <= signal.price:
                logger.error("Invalid short SL: %.4f <= entry %.4f", sl, signal.price)
                return None
            if tp <= 0 or tp >= signal.price:
                logger.error("Invalid short TP: %.4f >= entry %.4f", tp, signal.price)
                return None

        if qty <= 0:
            return None

        actual_risk = qty * stop_distance
        notional = qty * signal.price
        logger.info(
            "Position sized: side=%s price=%.4f qty=%.6f sl=%.4f tp=%.4f "
            "lev=%sx notional=%.2f risk_usd=%.2f",
            signal.side, signal.price, qty, sl, tp, lev, notional, actual_risk,
        )
        return SizingResult(qty=qty, stop_loss=sl, take_profit=tp)

    def fee_cost(self, notional: float) -> float:
        return notional * (self.cfg.fee_pct / 100.0)
