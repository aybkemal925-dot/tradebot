from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from .config import Settings
from .exchange import ExchangeClient
from .models import Signal


@dataclass(slots=True)
class LayerScore:
    name: str
    score: float
    detail: str


@dataclass(slots=True)
class ScoreCard:
    technical: LayerScore
    onchain: LayerScore
    sentiment: LayerScore
    regime: LayerScore
    total: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.total >= self.threshold

    def summary(self) -> str:
        return (
            f"SCORE {self.total:.1f}/100 | "
            f"Tech {self.technical.score:.0f} | "
            f"OnChain {self.onchain.score:.0f} | "
            f"Sent {self.sentiment.score:.0f} | "
            f"Regime {self.regime.score:.0f}"
        )


class SignalScorer:
    def __init__(self, settings: Settings, exchange: ExchangeClient) -> None:
        self.settings = settings
        self.exchange = exchange
        self._cache: dict[str, tuple[float, object]] = {}
        self._prev_open_interest: dict[str, float] = {}

    def _clip(self, value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return float(min(max(value, lo), hi))

    def _cached(self, key: str, ttl_sec: float, loader) -> object | None:
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] <= ttl_sec:
            return cached[1]
        value = loader()
        self._cache[key] = (now, value)
        return value

    def _adx(self, df: pd.DataFrame, period: int) -> float:
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

    def _cvd_bias(self, df: pd.DataFrame, lookback: int) -> float:
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

    def _score_rsi(self, side: str, rsi: float) -> float:
        if side == "long":
            if rsi <= 30:
                return 100.0
            if rsi <= 35:
                return 90.0
            if rsi <= 45:
                return 70.0
            if rsi <= 55:
                return 45.0
            return 15.0
        if rsi >= 70:
            return 100.0
        if rsi >= 65:
            return 90.0
        if rsi >= 55:
            return 70.0
        if rsi >= 45:
            return 45.0
        return 15.0

    def _score_adx(self, adx: float) -> float:
        threshold = float(self.settings.strategy.adx_threshold)
        if adx >= threshold + 12:
            return 100.0
        if adx >= threshold + 6:
            return 85.0
        if adx >= threshold:
            return 70.0
        if adx >= max(threshold - 4, 10):
            return 50.0
        return 25.0

    def _score_cvd(self, side: str, cvd_bias: float) -> float:
        if side == "long":
            return self._clip((cvd_bias + 1.0) * 50.0)
        return self._clip((1.0 - cvd_bias) * 50.0)

    def _technical_layer(self, signal: Signal, df: pd.DataFrame) -> LayerScore:
        ctx = signal.context
        rsi = float(ctx.get("rsi", 50.0))
        adx = float(ctx.get("adx", self._adx(df, int(self.settings.strategy.adx_period))))
        cvd_bias = float(ctx.get("cvd_bias", self._cvd_bias(df, int(self.settings.strategy.cvd_lookback))))

        rsi_score = self._score_rsi(signal.side, rsi)
        adx_score = self._score_adx(adx)
        cvd_score = self._score_cvd(signal.side, cvd_bias)
        total = (rsi_score * 0.40) + (adx_score * 0.35) + (cvd_score * 0.25)
        detail = f"RSI {rsi:.1f} ADX {adx:.1f} CVD {cvd_bias:+.2f}"
        return LayerScore(name="technical", score=total, detail=detail)

    def _funding_score(self, side: str, funding: float | None) -> float:
        if funding is None:
            return 50.0
        funding_bps = funding * 10000.0
        if side == "long":
            return self._clip(65.0 - (funding_bps * 8.0))
        return self._clip(65.0 + (funding_bps * 8.0))

    def _oi_score(self, symbol: str, open_interest: float | None) -> float:
        if open_interest is None or open_interest <= 0:
            return 50.0
        prev = self._prev_open_interest.get(symbol)
        self._prev_open_interest[symbol] = open_interest
        if prev is None or prev <= 0:
            return 55.0
        delta_pct = ((open_interest - prev) / prev) * 100.0
        if delta_pct >= 3.0:
            return 85.0
        if delta_pct >= 1.0:
            return 70.0
        if delta_pct > -1.0:
            return 55.0
        if delta_pct > -3.0:
            return 40.0
        return 25.0

    def _lsr_score(self, side: str, ratio: float | None) -> float:
        if ratio is None or ratio <= 0:
            return 50.0
        if side == "long":
            if ratio <= 0.9:
                return 85.0
            if ratio <= 1.1:
                return 70.0
            if ratio <= 1.3:
                return 55.0
            return 25.0
        if ratio >= 1.3:
            return 85.0
        if ratio >= 1.1:
            return 70.0
        if ratio >= 0.9:
            return 55.0
        return 25.0

    def _onchain_layer(self, signal: Signal, symbol: str) -> LayerScore:
        funding = self._cached(f"funding:{symbol}", 180.0, lambda: self.exchange.fetch_funding_rate_value(symbol))
        open_interest = self._cached(
            f"open_interest:{symbol}",
            180.0,
            lambda: self.exchange.fetch_open_interest_value(symbol),
        )
        long_short = self._cached(
            f"long_short:{symbol}",
            180.0,
            lambda: self.exchange.fetch_long_short_ratio(symbol),
        )
        funding_score = self._funding_score(signal.side, funding if isinstance(funding, float) else None)
        oi_score = self._oi_score(symbol, open_interest if isinstance(open_interest, float) else None)
        lsr_score = self._lsr_score(signal.side, long_short if isinstance(long_short, float) else None)
        total = (funding_score * 0.35) + (oi_score * 0.30) + (lsr_score * 0.35)
        detail = (
            f"funding={funding if funding is not None else 'na'} "
            f"oi={open_interest if open_interest is not None else 'na'} "
            f"lsr={long_short if long_short is not None else 'na'}"
        )
        return LayerScore(name="onchain", score=total, detail=detail)

    def _sentiment_layer(self, signal: Signal) -> LayerScore:
        fng = self._cached("fear_greed", 300.0, self.exchange.fetch_fear_greed_index)
        if not isinstance(fng, float):
            return LayerScore(name="sentiment", score=50.0, detail="fear_greed=na")

        if signal.side == "long":
            if fng <= 20:
                score = 90.0
            elif fng <= 35:
                score = 75.0
            elif fng <= 55:
                score = 55.0
            elif fng <= 75:
                score = 35.0
            else:
                score = 20.0
        else:
            if fng >= 80:
                score = 90.0
            elif fng >= 65:
                score = 75.0
            elif fng >= 45:
                score = 55.0
            elif fng >= 25:
                score = 35.0
            else:
                score = 20.0
        return LayerScore(name="sentiment", score=score, detail=f"fear_greed={fng:.0f}")

    def _regime_layer(self, signal: Signal, btc_df: pd.DataFrame) -> LayerScore:
        if btc_df is None or btc_df.empty or len(btc_df) < 60:
            return LayerScore(name="regime", score=50.0, detail="btc_regime=na")

        close = btc_df["close"].astype(float)
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        adx = self._adx(btc_df, int(self.settings.strategy.adx_period))
        last = float(close.iloc[-1])
        lookback = min(len(close) - 1, 96)
        ret_pct = ((last / float(close.iloc[-(lookback + 1)])) - 1.0) * 100.0 if lookback > 0 else 0.0

        regime = "range"
        if last > ema20 > ema50 and adx >= 18.0 and ret_pct >= 1.5:
            regime = "bull"
        elif last < ema20 < ema50 and adx >= 18.0 and ret_pct <= -1.5:
            regime = "bear"

        if regime == "bull":
            score = 90.0 if signal.side == "long" else 20.0
        elif regime == "bear":
            score = 90.0 if signal.side == "short" else 20.0
        else:
            score = 55.0
        detail = f"btc_regime={regime} ret={ret_pct:+.2f}% adx={adx:.1f}"
        return LayerScore(name="regime", score=score, detail=detail)

    def score_signal(self, signal: Signal, symbol: str, df: pd.DataFrame, btc_df: pd.DataFrame) -> ScoreCard:
        technical = self._technical_layer(signal, df)
        onchain = self._onchain_layer(signal, symbol)
        sentiment = self._sentiment_layer(signal)
        regime = self._regime_layer(signal, btc_df)

        weights = {
            "technical": float(self.settings.strategy.technical_weight),
            "onchain": float(self.settings.strategy.onchain_weight),
            "sentiment": float(self.settings.strategy.sentiment_weight),
            "regime": float(self.settings.strategy.regime_weight),
        }
        total_weight = sum(weights.values()) or 1.0
        total = (
            (technical.score * weights["technical"])
            + (onchain.score * weights["onchain"])
            + (sentiment.score * weights["sentiment"])
            + (regime.score * weights["regime"])
        ) / total_weight
        return ScoreCard(
            technical=technical,
            onchain=onchain,
            sentiment=sentiment,
            regime=regime,
            total=total,
            threshold=float(self.settings.strategy.score_threshold),
        )
