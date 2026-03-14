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
    tp1_price: float
    atr: float
    stop_distance: float
    trail_callback_pct: float
    trail_activation_price: float
    break_even_price: float


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def _atr(self, df: pd.DataFrame) -> float:
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        if len(h) < 2:
            return float(h[-1] - l[-1]) if len(h) > 0 else 0.0

        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )
        period = max(int(self.cfg.atr_period), 2)
        if len(tr) >= period:
            atr_val = float(tr[-period:].mean())
        else:
            atr_val = float(tr.mean()) if len(tr) > 0 else 0.0
        return atr_val if atr_val > 0 else float(h[-1] - l[-1])

    def _clamp_pct(self, value_pct: float, min_pct: float, max_pct: float) -> float:
        return float(min(max(value_pct, min_pct), max_pct))

    def _score_size_multiplier(self, score: float) -> float:
        """Return position size multiplier based on signal score (0-100).

        High-conviction signals (score > 80) get up to 1.3x,
        low-conviction signals (score < 50) get 0.7x.
        """
        if score >= 85.0:
            return 1.30
        if score >= 70.0:
            return 1.15
        if score >= 55.0:
            return 1.00
        if score >= 40.0:
            return 0.85
        return 0.70

    def _volatility_size_multiplier(self, atr: float, price: float) -> float:
        """Reduce position size in high-volatility environments.

        ATR% < 0.5 → 1.1x (low vol, tight stops)
        ATR% 0.5-1.0 → 1.0x (normal)
        ATR% 1.0-2.0 → 0.85x (elevated)
        ATR% > 2.0 → 0.70x (high vol, wider stops = bigger risk per qty)
        """
        if price <= 0:
            return 1.0
        atr_pct = (atr / price) * 100.0
        if atr_pct < 0.5:
            return 1.10
        if atr_pct < 1.0:
            return 1.00
        if atr_pct < 2.0:
            return 0.85
        return 0.70

    def size_position(
        self,
        signal: Signal,
        equity: float,
        df: pd.DataFrame,
        open_positions: list[Position] | None = None,
        consecutive_losses: int = 0,
    ) -> SizingResult | None:
        atr = self._atr(df)
        if atr <= 0 or signal.price <= 0:
            return None

        lev = max(self.cfg.leverage, 1)
        base_risk_pct = self.cfg.risk_per_trade_pct

        # Reduce risk after consecutive losses (progressive de-risking)
        if consecutive_losses >= 3:
            base_risk_pct *= 0.50  # 50% reduction after 3+ losses
        elif consecutive_losses >= 2:
            base_risk_pct *= 0.70  # 30% reduction after 2 losses

        risk_usd = equity * (base_risk_pct / 100.0)
        stop_distance = atr * self.cfg.atr_stop_mult
        if risk_usd <= 0 or stop_distance <= 0:
            return None

        # Dynamic sizing: adjust by signal quality and market volatility
        score_mult = self._score_size_multiplier(float(signal.score))
        vol_mult = self._volatility_size_multiplier(atr, signal.price)
        qty = (risk_usd / stop_distance) * score_mult * vol_mult

        used_margin = sum(
            (p.qty * p.entry_price) / lev
            for p in (open_positions or [])
        )
        available_margin = max(0.0, equity - used_margin)
        max_notional = available_margin * lev
        if qty * signal.price > max_notional and signal.price > 0:
            capped_qty = max_notional / signal.price
            logger.warning(
                "Position qty capped by available margin: %.6f -> %.6f "
                "(equity=%.2f used_margin=%.2f lev=%sx)",
                qty,
                capped_qty,
                equity,
                used_margin,
                lev,
            )
            qty = capped_qty

        if qty <= 0:
            return None

        if signal.side == "long":
            sl = signal.price - stop_distance
            tp1 = signal.price + (atr * self.cfg.tp1_atr_mult)
            activation_pct = self._clamp_pct(
                (atr * self.cfg.trailing_activation_atr_mult / signal.price) * 100.0,
                self.cfg.trailing_activation_min_pct,
                self.cfg.trailing_activation_max_pct,
            )
            trail_activation_price = signal.price * (1.0 + (activation_pct / 100.0))
            if sl <= 0 or sl >= signal.price or tp1 <= signal.price:
                return None
        else:
            sl = signal.price + stop_distance
            tp1 = signal.price - (atr * self.cfg.tp1_atr_mult)
            activation_pct = self._clamp_pct(
                (atr * self.cfg.trailing_activation_atr_mult / signal.price) * 100.0,
                self.cfg.trailing_activation_min_pct,
                self.cfg.trailing_activation_max_pct,
            )
            trail_activation_price = signal.price * (1.0 - (activation_pct / 100.0))
            if sl <= signal.price or tp1 <= 0 or tp1 >= signal.price:
                return None

        callback_pct = self._clamp_pct(
            (atr * self.cfg.trailing_callback_atr_mult / signal.price) * 100.0,
            self.cfg.trailing_callback_min_pct,
            self.cfg.trailing_callback_max_pct,
        )
        actual_risk = qty * stop_distance
        notional = qty * signal.price
        logger.info(
            "Position sized: side=%s price=%.4f qty=%.6f sl=%.4f tp1=%.4f "
            "lev=%sx notional=%.2f risk_usd=%.2f score_mult=%.2f vol_mult=%.2f consec_losses=%s",
            signal.side,
            signal.price,
            qty,
            sl,
            tp1,
            lev,
            notional,
            actual_risk,
            score_mult,
            vol_mult,
            consecutive_losses,
        )
        return SizingResult(
            qty=qty,
            stop_loss=sl,
            take_profit=tp1,
            tp1_price=tp1,
            atr=atr,
            stop_distance=stop_distance,
            trail_callback_pct=callback_pct,
            trail_activation_price=trail_activation_price,
            break_even_price=signal.price,
        )

    def fee_cost(self, notional: float) -> float:
        return notional * (self.cfg.fee_pct / 100.0)
