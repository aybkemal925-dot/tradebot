"""LuxAlgo strategy implementation for structure, order blocks, and FVGs.

Signal conditions:
  - Swing CHoCH/BOS break (leg-based, close crossover)
  - Price inside active OB and trend filter (EMA50/200) aligned
  - Entry after waiting for confirm_bars
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .logging_utils import get_logger
from .models import Signal

logger = get_logger(__name__)


@dataclass(slots=True)
class Pivot:
    level: float = float("nan")
    bar_idx: int = -1
    crossed: bool = False


@dataclass(slots=True)
class OB:
    side: str
    top: float
    bottom: float
    bar_idx: int
    active: bool = True


@dataclass(slots=True)
class FVG:
    side: str
    top: float
    bottom: float
    active: bool = True


@dataclass(slots=True)
class PendingSignal:
    side: str
    kind: str
    price: float
    note: str
    remaining: int
    score: float = 0.0
    context: dict[str, object] | None = None


class LuxAlgoStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.swing_lb = max(cfg.swing_lookback, 5)
        self.internal_lb = max(cfg.pivot_lookback, 3)

        self._swing_high = Pivot()
        self._swing_low = Pivot()
        self._internal_high = Pivot()
        self._internal_low = Pivot()

        self._swing_leg: int = -1
        self._internal_leg: int = -1

        self._swing_trend: int = 0
        self._internal_trend: int = 0

        self.order_blocks: list[OB] = []
        self.fvgs: list[FVG] = []
        self.pending_signal: PendingSignal | None = None
        self.last_bar_ts: int | None = None

    def _calc_leg(self, highs: np.ndarray, lows: np.ndarray, size: int) -> int:
        if len(highs) < size + 2:
            return -1
        pivot_h = highs[-(size + 1)]
        pivot_l = lows[-(size + 1)]
        window_h = highs[-size:]
        window_l = lows[-size:]
        new_leg_high = pivot_h > window_h.max()
        new_leg_low = pivot_l < window_l.min()
        if new_leg_high:
            return 0
        if new_leg_low:
            return 1
        return -1

    def _parsed_hl(
        self, highs: np.ndarray, lows: np.ndarray, atr200: float
    ) -> tuple[np.ndarray, np.ndarray]:
        bar_range = highs - lows
        high_vol = bar_range >= 2 * atr200
        parsed_h = np.where(high_vol, lows, highs)
        parsed_l = np.where(high_vol, highs, lows)
        return parsed_h, parsed_l

    def _store_ob(
        self,
        side: str,
        pivot_idx: int,
        current_idx: int,
        parsed_h: np.ndarray,
        parsed_l: np.ndarray,
    ) -> None:
        if pivot_idx < 0 or current_idx <= pivot_idx:
            return
        start = pivot_idx
        end = current_idx

        if side == "long":
            sub = parsed_l[start:end]
            if len(sub) == 0:
                return
            local_idx = int(np.argmin(sub)) + start
        else:
            sub = parsed_h[start:end]
            if len(sub) == 0:
                return
            local_idx = int(np.argmax(sub)) + start

        ob_top = float(parsed_h[local_idx])
        ob_bottom = float(parsed_l[local_idx])

        for existing in self.order_blocks:
            if existing.bar_idx == local_idx and existing.side == side:
                return

        self.order_blocks.append(
            OB(side=side, top=ob_top, bottom=ob_bottom, bar_idx=local_idx, active=True)
        )
        if len(self.order_blocks) > 20:
            self.order_blocks.pop(0)

    def _invalidate_obs(self, close: float, high: float, low: float) -> None:
        for ob in self.order_blocks:
            if not ob.active:
                continue
            if ob.side == "short" and high >= ob.top:
                ob.active = False
            elif ob.side == "long" and low <= ob.bottom:
                ob.active = False
        self.order_blocks = [ob for ob in self.order_blocks if ob.active]

    def _update_fvg(self, df: pd.DataFrame) -> None:
        if len(df) < 3:
            return
        h2 = float(df["high"].iloc[-3])
        l2 = float(df["low"].iloc[-3])
        h0 = float(df["high"].iloc[-1])
        l0 = float(df["low"].iloc[-1])
        c1 = float(df["close"].iloc[-2])

        if l0 > h2 and c1 > h2:
            self.fvgs.append(FVG(side="long", top=l0, bottom=h2))
        if h0 < l2 and c1 < l2:
            self.fvgs.append(FVG(side="short", top=l2, bottom=h0))

        close = float(df["close"].iloc[-1])
        for g in self.fvgs:
            if g.side == "long" and close <= g.bottom:
                g.active = False
            if g.side == "short" and close >= g.top:
                g.active = False
        self.fvgs = [g for g in self.fvgs if g.active][-20:]

    def _trend(self, close_s: pd.Series) -> int:
        if len(close_s) < 200:
            return 0
        e50 = close_s.ewm(span=50, adjust=False).mean().iloc[-1]
        e200 = close_s.ewm(span=200, adjust=False).mean().iloc[-1]
        c = close_s.iloc[-1]
        if c > e50 > e200:
            return 1
        if c < e50 < e200:
            return -1
        return 0

    def _in_ob(self, side: str, price: float) -> bool:
        return any(
            ob.side == side and ob.bottom <= price <= ob.top
            for ob in self.order_blocks
            if ob.active
        )

    def _has_fvg(self, side: str) -> bool:
        return any(g.side == side for g in self.fvgs if g.active)

    def _rsi(self, close_s: pd.Series) -> float:
        period = max(int(self.cfg.rsi_period), 2)
        if len(close_s) < period + 1:
            return 50.0
        delta = close_s.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        if "volume" not in df or len(df) < max(int(self.cfg.volume_ma_period), 2):
            return 0.0
        period = max(int(self.cfg.volume_ma_period), 2)
        vol = df["volume"]
        avg = vol.rolling(period).mean().iloc[-1]
        if pd.isna(avg) or avg <= 0:
            return 0.0
        return float(vol.iloc[-1] / avg)

    def _adx(self, df: pd.DataFrame) -> float:
        period = max(int(self.cfg.adx_period), 2)
        if len(df) < period + 2:
            return 0.0

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        if atr.iloc[-1] <= 0 or pd.isna(atr.iloc[-1]):
            return 0.0

        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        di_sum = plus_di + minus_di
        if di_sum.iloc[-1] <= 0 or pd.isna(di_sum.iloc[-1]):
            return 0.0
        dx = ((plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)) * 100.0
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return 0.0 if pd.isna(adx) else float(adx)

    def _cvd_bias(self, df: pd.DataFrame) -> float:
        lookback = max(int(self.cfg.cvd_lookback), 5)
        if "volume" not in df or len(df) < lookback:
            return 0.0
        window = df.iloc[-lookback:]
        signed_volume = np.where(
            window["close"] > window["open"],
            window["volume"],
            np.where(window["close"] < window["open"], -window["volume"], 0.0),
        )
        total_abs = float(np.abs(signed_volume).sum())
        if total_abs <= 0:
            return 0.0
        return float(np.clip(signed_volume.sum() / total_abs, -1.0, 1.0))

    def next_signal(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> Signal | None:
        min_bars = max(self.swing_lb, self.internal_lb) * 2 + 10
        if len(df) < min_bars:
            return None

        bar_ts = int(df["ts"].iloc[-1])
        if self.last_bar_ts == bar_ts:
            return None
        self.last_bar_ts = bar_ts

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        close = closes[-1]
        high = highs[-1]
        low = lows[-1]

        atr200 = float(
            np.mean(
                np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
                )[-200:]
            )
        )
        parsed_h, parsed_l = self._parsed_hl(highs, lows, atr200)

        current_idx = len(df) - 1
        new_swing_leg = self._calc_leg(highs, lows, self.swing_lb)

        if new_swing_leg != -1 and new_swing_leg != self._swing_leg:
            self._swing_leg = new_swing_leg
            pivot_bar = current_idx - self.swing_lb
            if new_swing_leg == 0:
                self._swing_high = Pivot(level=float(highs[pivot_bar]), bar_idx=pivot_bar, crossed=False)
            else:
                self._swing_low = Pivot(level=float(lows[pivot_bar]), bar_idx=pivot_bar, crossed=False)

        new_int_leg = self._calc_leg(highs, lows, self.internal_lb)

        if new_int_leg != -1 and new_int_leg != self._internal_leg:
            self._internal_leg = new_int_leg
            pivot_bar = current_idx - self.internal_lb
            if new_int_leg == 0:
                self._internal_high = Pivot(
                    level=float(highs[pivot_bar]), bar_idx=pivot_bar, crossed=False
                )
            else:
                self._internal_low = Pivot(level=float(lows[pivot_bar]), bar_idx=pivot_bar, crossed=False)

        self._update_fvg(df)
        self._invalidate_obs(close, high, low)

        raw_signal: Signal | None = None
        sh_level = self._swing_high.level
        sl_level = self._swing_low.level

        if not np.isnan(sh_level) and not self._swing_high.crossed and close > sh_level:
            kind = "choch" if self._swing_trend == -1 else "bos"
            self._swing_high.crossed = True
            self._swing_trend = 1
            self._store_ob("long", self._swing_high.bar_idx, current_idx, parsed_h, parsed_l)
            raw_signal = self._make_signal("long", kind, df, htf_df)
        elif not np.isnan(sl_level) and not self._swing_low.crossed and close < sl_level:
            kind = "choch" if self._swing_trend == 1 else "bos"
            self._swing_low.crossed = True
            self._swing_trend = -1
            self._store_ob("short", self._swing_low.bar_idx, current_idx, parsed_h, parsed_l)
            raw_signal = self._make_signal("short", kind, df, htf_df)

        if self.pending_signal is not None:
            ps = self.pending_signal
            ps.remaining -= 1
            if ps.remaining <= 0:
                self.pending_signal = None
                sig = self._make_signal(ps.side, ps.kind, df, htf_df)
                if sig is None:
                    logger.info("LuxAlgo signal rejected after confirm: %s %s", ps.side, ps.kind)
                    return None
                logger.info("LuxAlgo signal confirmed: %s %s @ %.2f", sig.side, sig.kind, sig.price)
                return sig
            return None

        if raw_signal is not None:
            confirm = max(1, int(self.cfg.confirm_bars))
            self.pending_signal = PendingSignal(
                side=raw_signal.side,
                kind=raw_signal.kind,
                price=raw_signal.price,
                note=raw_signal.note,
                remaining=confirm,
                score=raw_signal.score,
                context=dict(raw_signal.context),
            )
            logger.info(
                "LuxAlgo signal queued: %s %s @ %.2f (confirm=%d)",
                raw_signal.side,
                raw_signal.kind,
                raw_signal.price,
                confirm,
            )

        return None

    def _make_signal(
        self,
        side: str,
        kind: str,
        df: pd.DataFrame,
        htf_df: pd.DataFrame | None = None,
    ) -> Signal | None:
        if self.cfg.use_choch_only and kind != "choch":
            return None
        if side == "long" and not self.cfg.allow_long:
            return None
        if side == "short" and not self.cfg.allow_short:
            return None

        close = float(df["close"].iloc[-1])
        close_s = df["close"]
        trend = self._trend(close_s)
        htf_close_s = htf_df["close"] if htf_df is not None and not htf_df.empty else close_s
        htf_trend = self._trend(htf_close_s)

        if self.cfg.ema_trend_filter:
            if side == "long" and trend == -1:
                return None
            if side == "short" and trend == 1:
                return None

        if self.cfg.htf_trend_filter:
            if side == "long" and htf_trend != 1:
                return None
            if side == "short" and htf_trend != -1:
                return None

        rsi = self._rsi(close_s)
        adx = self._adx(df)
        cvd_bias = self._cvd_bias(df)
        volume_ratio = self._volume_ratio(df)

        if self.cfg.volume_filter:
            if volume_ratio < float(self.cfg.min_volume_ratio):
                return None

        ob_ok = self._in_ob(side, close)
        fvg_ok = self._has_fvg(side)

        if self.cfg.use_ob_fvg_filter and (self.order_blocks or self.fvgs):
            if not ob_ok and not fvg_ok:
                return None

        note_parts = [f"LuxAlgo {kind.upper()}"]
        if ob_ok:
            note_parts.append("OB")
        if fvg_ok:
            note_parts.append("FVG")
        note_parts.append(f"RSI {rsi:.1f}")
        note_parts.append(f"ADX {adx:.1f}")
        note_parts.append(f"CVD {cvd_bias:+.2f}")
        note_parts.append(f"VOL x{volume_ratio:.2f}")

        return Signal(
            ts=datetime.now(timezone.utc),
            side=side,
            kind=kind,
            price=close,
            note=" | ".join(note_parts),
            context={
                "rsi": rsi,
                "adx": adx,
                "cvd_bias": cvd_bias,
                "volume_ratio": volume_ratio,
                "trend": trend,
                "htf_trend": htf_trend,
                "ob_ok": ob_ok,
                "fvg_ok": fvg_ok,
            },
        )
