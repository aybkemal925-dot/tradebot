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
class PendingSignal:
    side: str
    kind: str
    remaining: int
    note: str
    context: dict[str, object]


@dataclass(slots=True)
class OrderBlock:
    idx: int
    top: float
    bottom: float
    side: str  # bull or bear

class LuxAlgoStrategy:
    """Lux Algo + Smart Money Concepts (SMC) hybrid strategy.

    Uses market structure (BOS/CHOCH), order blocks, trend filters,
    and candle confirmation to generate high-probability signals.
    Targets ~63%+ win rate with confluence scoring system.
    """

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.pending_signal: PendingSignal | None = None
        self.last_bar_ts: int | None = None
        self.cooldown_remaining = 0
        # SMC state
        self.swing_highs: list[tuple[int, float]] = []
        self.swing_lows: list[tuple[int, float]] = []
        self.last_sh_price = 0.0
        self.last_sl_price = 0.0
        self.last_sh_idx = -1
        self.last_sl_idx = -1
        self.current_trend = 0  # 1=bull, -1=bear
        self.last_break_bar = -20
        self.last_break_type: str | None = None
        self.order_blocks: list[OrderBlock] = []
        self.bar_count = 0

    # ── Indicator helpers ──

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=max(period, 2), adjust=False).mean()

    def _atr_series(self, df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / max(period, 2), adjust=False).mean()

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
        period = max(int(self.cfg.volume_ma_period), 2)
        if "volume" not in df or len(df) < period:
            return 0.0
        vol = df["volume"].astype(float)
        avg = vol.rolling(period).mean().iloc[-1]
        if pd.isna(avg) or avg <= 0:
            return 0.0
        return float(vol.iloc[-1] / avg)

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index) for trend strength."""
        period = max(period, 2)
        if len(df) < period * 2:
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        # Directional movement
        up_move = high.diff()
        down_move = -low.diff()
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
        pos_di = 100.0 * pd.Series(pos_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
        neg_di = 100.0 * pd.Series(neg_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
        dx = (100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return float(adx.iloc[-1])

    def _ema_slope(self, series: pd.Series, period: int, lookback: int = 5) -> float:
        """Return EMA slope as pct change over lookback bars (positive = rising)."""
        ema = self._ema(series, period)
        if len(ema) < lookback + 1:
            return 0.0
        prev = float(ema.iloc[-(lookback + 1)])
        curr = float(ema.iloc[-1])
        if prev <= 0:
            return 0.0
        return (curr - prev) / prev * 100.0

    # ── SMC: Swing detection ──

    def _detect_swing(self, df: pd.DataFrame, lookback: int = 5) -> None:
        """Detect swing highs/lows and update structure state."""
        n = len(df)
        if n < lookback * 2 + 1:
            return
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        idx = n - lookback - 1  # candidate index
        if idx < lookback:
            return
        # Check swing high
        h_val = float(high.iloc[idx])
        is_sh = all(h_val > float(high.iloc[idx + d]) for d in range(-lookback, lookback + 1) if d != 0 and 0 <= idx + d < n)
        if is_sh:
            self.swing_highs.append((idx, h_val))
            self.last_sh_price = h_val
            self.last_sh_idx = idx
            self.swing_highs = self.swing_highs[-50:]
        # Check swing low
        l_val = float(low.iloc[idx])
        is_sl = all(l_val < float(low.iloc[idx + d]) for d in range(-lookback, lookback + 1) if d != 0 and 0 <= idx + d < n)
        if is_sl:
            self.swing_lows.append((idx, l_val))
            self.last_sl_price = l_val
            self.last_sl_idx = idx
            self.swing_lows = self.swing_lows[-50:]

    # ── SMC: Market Structure (BOS / CHOCH) ──

    def _detect_structure_break(self, close_val: float, bar_idx: int) -> str | None:
        """Detect BOS or CHOCH. Returns break type or None."""
        if self.last_sh_price <= 0 or self.last_sl_price <= 0:
            return None
        if bar_idx - self.last_break_bar < 3:
            return None

        if close_val > self.last_sh_price and bar_idx > self.last_sh_idx:
            if self.current_trend >= 0:
                break_type = "BOS_UP"
            else:
                break_type = "CHOCH_UP"
            self.current_trend = 1
            self.last_break_bar = bar_idx
            self.last_break_type = break_type
            self.last_sh_price = close_val
            self.last_sh_idx = bar_idx
            return break_type

        if close_val < self.last_sl_price and bar_idx > self.last_sl_idx:
            if self.current_trend <= 0:
                break_type = "BOS_DOWN"
            else:
                break_type = "CHOCH_DOWN"
            self.current_trend = -1
            self.last_break_bar = bar_idx
            self.last_break_type = break_type
            self.last_sl_price = close_val
            self.last_sl_idx = bar_idx
            return break_type

        return None

    # ── SMC: Order Block detection ──

    def _detect_order_block(self, df: pd.DataFrame, break_type: str, break_idx: int) -> None:
        """Find order block before a structure break."""
        close_arr = df["close"].astype(float)
        open_arr = df["open"].astype(float)
        if break_type in ("BOS_UP", "CHOCH_UP"):
            for j in range(break_idx - 1, max(break_idx - 10, 0), -1):
                if float(close_arr.iloc[j]) < float(open_arr.iloc[j]):
                    o_val = float(open_arr.iloc[j])
                    c_val = float(close_arr.iloc[j])
                    self.order_blocks.append(OrderBlock(idx=j, top=max(o_val, c_val), bottom=min(o_val, c_val), side="bull"))
                    break
        elif break_type in ("BOS_DOWN", "CHOCH_DOWN"):
            for j in range(break_idx - 1, max(break_idx - 10, 0), -1):
                if float(close_arr.iloc[j]) > float(open_arr.iloc[j]):
                    o_val = float(open_arr.iloc[j])
                    c_val = float(close_arr.iloc[j])
                    self.order_blocks.append(OrderBlock(idx=j, top=max(o_val, c_val), bottom=min(o_val, c_val), side="bear"))
                    break
        self.order_blocks = self.order_blocks[-60:]

    # ── Candle pattern helpers ──

    def _is_pin_bar(self, o: float, h: float, l: float, c: float, atr_val: float) -> int:
        body = abs(c - o)
        rng = h - l
        if rng < atr_val * 0.3:
            return 0
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if lower_wick > body * 2 and lower_wick > rng * 0.5:
            return 1  # bullish pin
        if upper_wick > body * 2 and upper_wick > rng * 0.5:
            return -1  # bearish pin
        return 0

    def _is_engulfing(self, o1: float, c1: float, o2: float, c2: float) -> int:
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            return 1  # bullish engulfing
        if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
            return -1  # bearish engulfing
        return 0

    # ── Confluence scoring ──

    def _calc_confluence(self, df: pd.DataFrame, htf_df: pd.DataFrame | None, bar_idx: int, atr_val: float) -> tuple[int, int, dict]:
        """Calculate confluence score for long and short. Returns (long_score, short_score, context)."""
        close_s = df["close"].astype(float)
        close_val = float(close_s.iloc[-1])
        open_val = float(df["open"].astype(float).iloc[-1])
        high_val = float(df["high"].astype(float).iloc[-1])
        low_val = float(df["low"].astype(float).iloc[-1])
        sl_score = 0
        ss_score = 0
        ctx = {}

        # 1. Market structure trend (2 pts)
        if self.current_trend == 1:
            sl_score += 2
        elif self.current_trend == -1:
            ss_score += 2
        ctx["smc_trend"] = self.current_trend

        # 2. Recent BOS/CHOCH (1 pt)
        if self.last_break_type and bar_idx - self.last_break_bar <= 20:
            if self.last_break_type in ("BOS_UP", "CHOCH_UP"):
                sl_score += 1
            else:
                ss_score += 1
        ctx["last_break"] = self.last_break_type or "none"

        # 3. Order Block retest (2 pts)
        ob_buffer = atr_val * 0.3
        ob_max_age = 50
        ob_long = None
        ob_short = None
        for ob in self.order_blocks:
            if bar_idx - ob.idx > ob_max_age:
                continue
            if ob.side == "bull" and low_val <= ob.top + ob_buffer and close_val >= ob.bottom - ob_buffer:
                ob_long = ob
                break
        for ob in self.order_blocks:
            if bar_idx - ob.idx > ob_max_age:
                continue
            if ob.side == "bear" and high_val >= ob.bottom - ob_buffer and close_val <= ob.top + ob_buffer:
                ob_short = ob
                break
        if ob_long:
            sl_score += 2
        if ob_short:
            ss_score += 2
        ctx["ob_long"] = ob_long is not None
        ctx["ob_short"] = ob_short is not None

        # 4. EMA trend filter (1 pt)
        ema_fast = self._ema(close_s, int(self.cfg.ema_fast_period))
        ema_slow = self._ema(close_s, int(self.cfg.ema_slow_period))
        ema_fast_val = float(ema_fast.iloc[-1])
        ema_slow_val = float(ema_slow.iloc[-1])
        ema_bull = ema_fast_val > ema_slow_val
        ema_bear = ema_fast_val < ema_slow_val
        if ema_bull:
            sl_score += 1
        if ema_bear:
            ss_score += 1
        ctx["ema_fast"] = ema_fast_val
        ctx["ema_slow"] = ema_slow_val

        # 5. EMA200 major trend (1 pt)
        ema200 = self._ema(close_s, 200)
        ema200_val = float(ema200.iloc[-1]) if len(close_s) >= 200 else close_val
        if close_val > ema200_val:
            sl_score += 1
        elif close_val < ema200_val:
            ss_score += 1
        ctx["ema200"] = ema200_val

        # 6. HTF trend alignment (1 pt)
        htf_trend = 0
        if htf_df is not None and not htf_df.empty and len(htf_df) > 60:
            htf_close = htf_df["close"].astype(float)
            htf_ema_f = self._ema(htf_close, int(self.cfg.ema_fast_period))
            htf_ema_s = self._ema(htf_close, int(self.cfg.ema_slow_period))
            hc = float(htf_close.iloc[-1])
            hf = float(htf_ema_f.iloc[-1])
            hs = float(htf_ema_s.iloc[-1])
            if hc > hf > hs:
                htf_trend = 1
            elif hc < hf < hs:
                htf_trend = -1
        if htf_trend == 1:
            sl_score += 1
        elif htf_trend == -1:
            ss_score += 1
        ctx["htf_trend"] = htf_trend

        # 7. Candle confirmation (2 pts)
        pin = self._is_pin_bar(open_val, high_val, low_val, close_val, atr_val)
        eng = 0
        if len(df) >= 2:
            eng = self._is_engulfing(float(df["open"].iloc[-2]), float(df["close"].iloc[-2]), open_val, close_val)
        if pin == 1 or eng == 1:
            sl_score += 2
        if pin == -1 or eng == -1:
            ss_score += 2
        ctx["pin_bar"] = pin
        ctx["engulfing"] = eng

        # 8. Bullish/bearish candle (1 pt)
        candle_range = high_val - low_val
        body_ratio = abs(close_val - open_val) / candle_range if candle_range > 0 else 0
        candle_ok_l = pin == 1 or eng == 1 or (close_val > open_val and body_ratio > 0.5)
        candle_ok_s = pin == -1 or eng == -1 or (close_val < open_val and body_ratio > 0.5)
        if close_val > open_val:
            sl_score += 1
        if close_val < open_val:
            ss_score += 1

        # 9. Volume (1 pt)
        vol_ratio = self._volume_ratio(df)
        vol_threshold = max(float(self.cfg.min_volume_ratio), 0.5)
        if vol_ratio >= vol_threshold:
            sl_score += 1
            ss_score += 1
        ctx["volume_ratio"] = vol_ratio

        # 10. ADX trend strength (1 pt) — only score if ADX is strong
        adx_val = self._adx(df, int(self.cfg.adx_period))
        adx_threshold = float(self.cfg.adx_threshold)
        if adx_val >= adx_threshold:
            sl_score += 1
            ss_score += 1
        ctx["adx"] = adx_val

        # 11. EMA slope (1 pt) — reward rising/falling EMA momentum
        slope_lookback = max(int(self.cfg.ema_slope_lookback), 2)
        flat_thr = float(self.cfg.ema_flat_threshold_pct)
        ema_slope = self._ema_slope(close_s, int(self.cfg.ema_fast_period), slope_lookback)
        if ema_slope > flat_thr:
            sl_score += 1
        elif ema_slope < -flat_thr:
            ss_score += 1
        ctx["ema_slope"] = ema_slope

        # RSI
        rsi_val = self._rsi(close_s)
        ctx["rsi"] = rsi_val
        ctx["atr"] = atr_val
        ctx["atr_pct"] = (atr_val / close_val) * 100 if close_val > 0 else 0
        ctx["long_score"] = sl_score
        ctx["short_score"] = ss_score
        ctx["ema_bull"] = ema_bull
        ctx["ema_bear"] = ema_bear
        ctx["ob_long_obj"] = ob_long
        ctx["ob_short_obj"] = ob_short
        ctx["candle_ok_l"] = candle_ok_l
        ctx["candle_ok_s"] = candle_ok_s

        return sl_score, ss_score, ctx

    # ── Main signal generation ──

    def next_signal(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> Signal | None:
        warmup = max(200, int(self.cfg.ema_slow_period) + 10, int(self.cfg.breakout_lookback) + 10)
        if len(df) < warmup:
            return None

        bar_ts = int(df["ts"].iloc[-1])
        if self.last_bar_ts == bar_ts:
            return None
        self.last_bar_ts = bar_ts
        self.bar_count = len(df) - 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # Step 1: Detect swings
        self._detect_swing(df, lookback=int(self.cfg.pivot_lookback))

        # Step 2: Detect structure breaks
        close_val = float(df["close"].astype(float).iloc[-1])
        break_type = self._detect_structure_break(close_val, self.bar_count)

        # Step 3: Detect order blocks on break
        if break_type:
            self._detect_order_block(df, break_type, self.bar_count)

        # Step 4: Calculate ATR
        atr_series = self._atr_series(df, 14)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if atr_val <= 0:
            return None
        atr_pct = (atr_val / close_val) * 100 if close_val > 0 else 0

        # Min volatility filter
        if atr_pct < float(self.cfg.min_atr_pct):
            return None

        # Cooldown check
        if self.cooldown_remaining > 0:
            return None

        # Step 5: Confluence scoring
        long_score, short_score, ctx = self._calc_confluence(df, htf_df, self.bar_count, atr_val)

        # Step 6: Entry decision
        # min_score: normalize lux_signal_threshold (0-100 scale) to raw score (max ~15)
        # lux_signal_threshold=70 → min_score=7, =80 → 8, =90 → 9
        max_possible = 15  # updated with ADX and slope additions
        min_score = max(5, round(float(self.cfg.lux_signal_threshold) / 100.0 * max_possible))
        rsi_val = float(ctx.get("rsi", 50))
        vol_ratio = float(ctx.get("volume_ratio", 1.0))
        adx_val = float(ctx.get("adx", 0.0))

        # RSI filters using config values (oversold for long, overbought for short)
        rsi_ok_long = rsi_val < float(self.cfg.rsi_long_max) or (20.0 < rsi_val < 70.0)
        rsi_ok_short = rsi_val > float(self.cfg.rsi_short_min) or (30.0 < rsi_val < 80.0)
        # Always ensure RSI is not at extreme (avoid chasing)
        rsi_ok_long = rsi_ok_long and rsi_val > 20.0
        rsi_ok_short = rsi_ok_short and rsi_val < 80.0

        # Minimum ADX requirement — only trade in trending markets
        adx_ok = adx_val >= max(float(self.cfg.adx_threshold) * 0.8, 12.0)

        side = ""
        score = 0.0

        # LONG entry
        if (self.cfg.allow_long and
                long_score >= min_score and
                self.current_trend == 1 and
                ctx.get("ob_long") and
                ctx.get("ema_bull") and
                ctx.get("candle_ok_l") and
                rsi_ok_long and
                adx_ok and
                vol_ratio >= 0.5):
            side = "long"
            score = (float(long_score) / max_possible) * 100.0

        # SHORT entry
        elif (self.cfg.allow_short and
                short_score >= min_score and
                self.current_trend == -1 and
                ctx.get("ob_short") and
                ctx.get("ema_bear") and
                ctx.get("candle_ok_s") and
                rsi_ok_short and
                adx_ok and
                vol_ratio >= 0.5):
            side = "short"
            score = (float(short_score) / max_possible) * 100.0

        if not side:
            return None

        # Build context
        context = {
            "rsi": rsi_val,
            "atr": atr_val,
            "atr_pct": atr_pct,
            "volume_ratio": vol_ratio,
            "trend": self.current_trend,
            "htf_trend": ctx.get("htf_trend", 0),
            "ema_fast": ctx.get("ema_fast", 0),
            "ema_slow": ctx.get("ema_slow", 0),
            "ema_slope": ctx.get("ema_slope", 0.0),
            "adx": adx_val,
            "lux_strength": score,
            "smc_break": self.last_break_type or "none",
            "long_confluence": long_score,
            "short_confluence": short_score,
        }
        note = (
            "SMC " + ("BUY" if side == "long" else "SELL")
            + " | score " + ("%.0f" % score)
            + " | " + (self.last_break_type or "")
            + " | RSI %.1f" % rsi_val
            + " | ADX %.1f" % adx_val
            + " | ATR%% %.2f" % atr_pct
        )

        signal = Signal(
            ts=datetime.now(timezone.utc),
            side=side,
            kind="mcp",
            price=close_val,
            note=note,
            score=score,
            context=context,
        )

        confirm = max(1, int(self.cfg.confirm_bars))
        if confirm <= 1:
            self.cooldown_remaining = max(int(self.cfg.cooldown_bars), 0)
            return signal

        if self.pending_signal is None:
            self.pending_signal = PendingSignal(
                side=signal.side, kind=signal.kind,
                remaining=confirm - 1, note=signal.note,
                context=dict(signal.context),
            )
            return None

        self.pending_signal.remaining -= 1
        if self.pending_signal.remaining <= 0 and self.pending_signal.side == signal.side:
            self.pending_signal = None
            self.cooldown_remaining = max(int(self.cfg.cooldown_bars), 0)
            return signal

        return None
