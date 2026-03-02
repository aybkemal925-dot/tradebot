from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.request import urlopen

import ccxt
import pandas as pd

from .config import ExchangeConfig
from .logging_utils import get_logger

logger = get_logger(__name__)

MAJOR_BASES = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "TRX", "BCH", "ATOM", "NEAR", "APT", "ARB", "OP", "SUI",
    "FIL", "ETC", "INJ", "AAVE", "UNI",
}


class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig) -> None:
        self.cfg = cfg
        self._leverage_set: set[str] = set()

        options = {"adjustForTimeDifference": True}
        if cfg.market_type.lower() == "futures":
            options["defaultType"] = "future"

        params = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
            "timeout": 20000,   # 20 sn — load_markets sonsuz beklemesin
            "options": options,
        }

        if cfg.name.lower() == "binance":
            self.ex = ccxt.binance(params)
        elif cfg.name.lower() == "mexc":
            self.ex = ccxt.mexc(params)
        else:
            raise ValueError(f"Unsupported exchange: {cfg.name}")

        self.ex.load_markets()
        logger.info("Exchange initialized: %s market_type=%s", cfg.name, cfg.market_type)

    def normalize_symbol(self, symbol: str) -> str:
        sym = symbol.strip().upper()
        if self.cfg.market_type.lower() == "futures" and self.cfg.name.lower() == "binance":
            if ":" not in sym and sym.endswith("/USDT"):
                return f"{sym}:USDT"
        return sym

    def base_symbol(self, market_symbol: str) -> str:
        if ":" in market_symbol:
            return market_symbol.split(":", 1)[0]
        return market_symbol

    def market_id(self, symbol: str) -> str:
        market_symbol = self.normalize_symbol(symbol)
        market = self.ex.markets.get(market_symbol, {})
        market_id = market.get("id")
        if market_id:
            return str(market_id)
        return market_symbol.replace("/", "").replace(":", "")

    def fetch_universe_symbols(self, count: int) -> list[str]:
        quote = self.cfg.scan_quote
        tickers = self.ex.fetch_tickers()
        ranked: list[tuple[str, float]] = []

        for m_symbol, market in self.ex.markets.items():
            if not market.get("active", True):
                continue
            if market.get("quote") != quote:
                continue

            if self.cfg.market_type.lower() == "futures":
                if not market.get("contract", False):
                    continue
                if market.get("linear") is False:
                    continue
            else:
                if market.get("spot") is False:
                    continue

            ticker = tickers.get(m_symbol, {})
            qv = ticker.get("quoteVolume")
            if qv is None:
                base_v = ticker.get("baseVolume") or 0.0
                last = ticker.get("last") or 0.0
                qv = float(base_v) * float(last)

            try:
                score = float(qv or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0

            if score <= 0:
                continue
            symbol = self.base_symbol(m_symbol)
            base = market.get("base")
            if self.cfg.majors_only and base not in MAJOR_BASES:
                continue
            ranked.append((symbol, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        out: list[str] = []
        seen: set[str] = set()
        for symbol, _ in ranked:
            if symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
            if len(out) >= count:
                break

        logger.info("Universe selected: %s symbols", len(out))
        return out

    def fetch_market_snapshot(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        normalized = [self.normalize_symbol(s) for s in symbols]
        data: dict[str, dict[str, float]] = {}

        try:
            tickers = self.ex.fetch_tickers(normalized)
        except Exception:  # noqa: BLE001
            tickers = {sym: self.ex.fetch_ticker(sym) for sym in normalized}

        for norm in normalized:
            t = tickers.get(norm, {})
            base = self.base_symbol(norm)
            last = float(t.get("last") or 0.0)
            change = t.get("percentage")
            if change is None:
                info = t.get("info") or {}
                raw = info.get("priceChangePercent") or info.get("change_rate") or 0.0
                try:
                    change = float(raw)
                except Exception:  # noqa: BLE001
                    change = 0.0
            data[base] = {"last": last, "change": float(change)}

        return data

    def fetch_ohlcv_for_timeframe(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        market_symbol = self.normalize_symbol(symbol)
        rows = self.ex.fetch_ohlcv(
            market_symbol,
            timeframe=timeframe,
            limit=limit or self.cfg.limit,
        )
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        return self.fetch_ohlcv_for_timeframe(symbol, self.cfg.timeframe, self.cfg.limit)

    def fetch_ohlcv_paginated(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int | None = None,
        batch_size: int = 1500,
        sleep_between: float = 0.15,
    ) -> pd.DataFrame:
        """Paginated OHLCV fetch — 12 aylık veri gibi büyük aralıklar için."""
        import time as _time

        market_symbol = self.normalize_symbol(symbol)
        all_rows: list[list] = []
        current_since = since_ms

        while True:
            rows = self.ex.fetch_ohlcv(
                market_symbol,
                timeframe=timeframe,
                since=current_since,
                limit=batch_size,
            )
            if not rows:
                break

            if until_ms is not None:
                rows = [r for r in rows if r[0] <= until_ms]
                all_rows.extend(rows)
                if not rows or rows[-1][0] >= until_ms:
                    break
            else:
                all_rows.extend(rows)
                if len(rows) < batch_size:
                    break

            current_since = rows[-1][0] + 1
            _time.sleep(sleep_between)

        if not all_rows:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.drop_duplicates(subset="ts", inplace=True)
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        logger.info("Paginated OHLCV: %s %s → %d bar", symbol, timeframe, len(df))
        return df

    def fetch_last_price(self, symbol: str) -> float:
        market_symbol = self.normalize_symbol(symbol)
        ticker = self.ex.fetch_ticker(market_symbol)
        return float(ticker["last"])

    def fetch_funding_rate_value(self, symbol: str) -> float | None:
        market_symbol = self.normalize_symbol(symbol)
        try:
            if hasattr(self.ex, "fetch_funding_rate"):
                data = self.ex.fetch_funding_rate(market_symbol)
                rate = data.get("fundingRate")
                if rate is not None:
                    return float(rate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fetch_funding_rate failed: %s %s", symbol, exc)

        try:
            if hasattr(self.ex, "fapiPublicGetPremiumIndex"):
                data = self.ex.fapiPublicGetPremiumIndex({"symbol": self.market_id(symbol)})
                rate = data.get("lastFundingRate")
                if rate is not None:
                    return float(rate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("premiumIndex funding fetch failed: %s %s", symbol, exc)
        return None

    def fetch_open_interest_value(self, symbol: str) -> float | None:
        try:
            if hasattr(self.ex, "fapiPublicGetOpenInterest"):
                data = self.ex.fapiPublicGetOpenInterest({"symbol": self.market_id(symbol)})
                value = data.get("openInterest")
                if value is not None:
                    return float(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("open interest fetch failed: %s %s", symbol, exc)
        return None

    def fetch_long_short_ratio(self, symbol: str, period: str = "5m") -> float | None:
        try:
            if hasattr(self.ex, "fapiPublicGetGlobalLongShortAccountRatio"):
                rows = self.ex.fapiPublicGetGlobalLongShortAccountRatio(
                    {"symbol": self.market_id(symbol), "period": period, "limit": 1}
                )
                if rows:
                    value = rows[-1].get("longShortRatio")
                    if value is not None:
                        return float(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("long/short ratio fetch failed: %s %s", symbol, exc)
        return None

    def fetch_fear_greed_index(self) -> float | None:
        try:
            with urlopen("https://api.alternative.me/fng/?limit=1&format=json", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") or []
            if data:
                return float(data[0].get("value"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear-greed fetch failed: %s", exc)
        return None

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        market_symbol = self.normalize_symbol(symbol)
        return float(self.ex.amount_to_precision(market_symbol, amount))

    def min_order_amount(self, symbol: str) -> float:
        """Return minimum order amount for symbol (0.0 if unknown)."""
        market_symbol = self.normalize_symbol(symbol)
        market = self.ex.markets.get(market_symbol, {})
        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        return float(amount_limits.get("min") or 0.0)

    def ensure_leverage(self, symbol: str) -> None:
        if self.cfg.market_type.lower() != "futures":
            return
        if self.cfg.leverage <= 0:
            return
        market_symbol = self.normalize_symbol(symbol)
        if market_symbol in self._leverage_set:
            return
        if hasattr(self.ex, "set_leverage"):
            self.ex.set_leverage(self.cfg.leverage, market_symbol)
            self._leverage_set.add(market_symbol)
            logger.info("Leverage set: %s %sx", market_symbol, self.cfg.leverage)

    def open_position(self, symbol: str, side: str, amount: float) -> float:
        market_symbol = self.normalize_symbol(symbol)
        self.ensure_leverage(symbol)
        order_side = "buy" if side == "long" else "sell"
        order = self.ex.create_order(market_symbol, "market", order_side, amount)
        price = self._extract_fill_price(order, symbol)
        logger.info("Order opened: %s %s qty=%.6f price=%.4f", symbol, side, amount, price)
        return price

    def close_position(self, symbol: str, side: str, amount: float) -> float:
        market_symbol = self.normalize_symbol(symbol)
        order_side = "sell" if side == "long" else "buy"
        params = {"reduceOnly": True} if self.cfg.market_type.lower() == "futures" else {}
        order = self.ex.create_order(market_symbol, "market", order_side, amount, None, params)
        price = self._extract_fill_price(order, symbol)
        logger.info("Order closed: %s %s qty=%.6f price=%.4f", symbol, side, amount, price)
        return price

    def place_sl_tp_orders(
        self, symbol: str, side: str, qty: float, sl_price: float, tp_price: float
    ) -> None:
        """Exchange'e SL (stop-market) ve TP (take_profit_market) order'ı gönder.

        Sadece live (non-paper) modda çağrılmalı.
        Binance Futures: closePosition=True ile reduceOnly order.
        """
        market_symbol = self.normalize_symbol(symbol)
        close_side = "sell" if side == "long" else "buy"

        # Stop-Loss
        try:
            self.ex.create_order(
                market_symbol,
                "stop_market",
                close_side,
                qty,
                params={
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                    "closePosition": True,
                },
            )
            logger.info("SL order placed: %s %s stopPrice=%.4f", symbol, side, sl_price)
        except Exception as exc:  # noqa: BLE001
            logger.error("SL order FAILED: %s — %s", symbol, exc)

        # Take-Profit
        try:
            self.ex.create_order(
                market_symbol,
                "take_profit_market",
                close_side,
                qty,
                params={
                    "stopPrice": tp_price,
                    "reduceOnly": True,
                    "closePosition": True,
                },
            )
            logger.info("TP order placed: %s %s stopPrice=%.4f", symbol, side, tp_price)
        except Exception as exc:  # noqa: BLE001
            logger.error("TP order FAILED: %s — %s", symbol, exc)

    def cancel_all_orders(self, symbol: str) -> None:
        """Sembol için tüm açık order'ları iptal et (pozisyon kapatılırken SL/TP temizleme)."""
        market_symbol = self.normalize_symbol(symbol)
        try:
            self.ex.cancel_all_orders(market_symbol)
            logger.info("All orders cancelled: %s", symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cancel all orders failed: %s — %s", symbol, exc)

    def fetch_open_positions(self) -> list[dict]:
        """Exchange'den mevcut açık pozisyonları çek (sadece futures)."""
        if self.cfg.market_type.lower() != "futures":
            return []
        try:
            positions = self.ex.fetch_positions()
            open_pos = [p for p in positions if abs(float(p.get("contracts") or 0)) > 0]
            logger.info("Fetched %s open positions from exchange", len(open_pos))
            return open_pos
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_open_positions failed: %s", exc)
            return []

    def _extract_fill_price(self, order: dict, symbol: str) -> float:
        average = order.get("average")
        price = order.get("price")
        if average is not None:
            return float(average)
        if price is not None:
            return float(price)
        return self.fetch_last_price(symbol)

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
