"""Austrade Backtest Modülü

Look-ahead bias önleme:
  Her i için bar_df = df.iloc[:i+1] — gelecek bar asla görülmez.
  SMCStrategy her backtest için taze instance (state sıfırlanmış).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Settings
from .exchange import ExchangeClient
from .logging_utils import get_logger
from .models import Position, Signal
from .risk import RiskManager
from .strategy_lux import LuxAlgoStrategy

logger = get_logger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache")


def _cache_path(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> Path:
    sym_clean = symbol.replace("/", "_").replace(":", "_").replace(" ", "")
    d_since = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    d_until = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)
    tag = f"{sym_clean}_{timeframe}_{d_since.strftime('%Y%m')}_{d_until.strftime('%Y%m')}"
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{tag}.csv"


def save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Cache kaydedildi: %s (%d bar)", path.name, len(df))


def load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        logger.info("Cache yüklendi: %s (%d bar)", path.name, len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache okunamadı, fresh fetch yapılacak: %s", exc)
        return None


# ─── DataClass'lar ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str = "5m"
    months: int = 12
    initial_equity: float = 1000.0
    use_cache: bool = True
    rsi_threshold: float | None = None
    allow_long: bool = True
    allow_short: bool = True


@dataclass
class BacktestTrade:
    position_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    entry_bar: int
    exit_bar: int
    entry_time: datetime
    exit_time: datetime
    reason: str   # "SL" | "TP" | "TIMEOUT"


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_equity: float
    final_equity: float
    total_bars: int
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    net_pnl_usd: float
    net_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_win_usd: float
    avg_loss_usd: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class MultiBacktestResult:
    summary: BacktestResult
    results: list[BacktestResult] = field(default_factory=list)


# ─── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, settings: Settings, exchange: ExchangeClient) -> None:
        self.settings = settings
        self.exchange = exchange
        self.risk = RiskManager(settings.risk)

    # ── Veri çekme ────────────────────────────────────────────────────────────

    def fetch_data(
        self,
        cfg: BacktestConfig,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> pd.DataFrame:
        until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_dt = datetime.now(timezone.utc) - timedelta(days=cfg.months * 30)
        since_ms = int(since_dt.timestamp() * 1000)

        cache_path = _cache_path(cfg.symbol, cfg.timeframe, since_ms, until_ms)

        if cfg.use_cache:
            cached = load_cache(cache_path)
            if cached is not None:
                if progress_callback:
                    progress_callback(f"Cache'den yüklendi — {len(cached):,} bar", 1.0)
                return cached

        if progress_callback:
            progress_callback("Binance'dan veri indiriliyor...", 0.0)

        df = self.exchange.fetch_ohlcv_paginated(
            cfg.symbol, cfg.timeframe, since_ms, until_ms
        )

        if df.empty:
            if progress_callback:
                progress_callback("Veri alınamadı!", 1.0)
            return df

        if cfg.use_cache:
            save_cache(df, cache_path)

        if progress_callback:
            progress_callback(f"{len(df):,} bar indirildi", 1.0)

        return df

    def fetch_htf_data(
        self,
        cfg: BacktestConfig,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> pd.DataFrame:
        htf_tf = self.settings.strategy.htf_timeframe
        if not self.settings.strategy.htf_trend_filter or htf_tf == cfg.timeframe:
            return pd.DataFrame()

        until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_dt = datetime.now(timezone.utc) - timedelta(days=cfg.months * 30)
        since_ms = int(since_dt.timestamp() * 1000)
        cache_path = _cache_path(cfg.symbol, htf_tf, since_ms, until_ms)

        if cfg.use_cache:
            cached = load_cache(cache_path)
            if cached is not None:
                return cached

        if progress_callback:
            progress_callback(f"HTF veri indiriliyor ({htf_tf})...", 0.0)

        df = self.exchange.fetch_ohlcv_paginated(cfg.symbol, htf_tf, since_ms, until_ms)
        if not df.empty and cfg.use_cache:
            save_cache(df, cache_path)
        return df

    # ── Backtest döngüsü ───────────────────────────────────────────────────────

    def run(
        self,
        cfg: BacktestConfig,
        df: pd.DataFrame,
        progress_callback: Callable[[str, float], None] | None = None,
        htf_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        if df.empty or len(df) < 50:
            return self._empty_result(cfg, df)

        # Strateji — her backtest için taze instance (look-ahead bias önleme)
        strategy_cfg = replace(
            self.settings.strategy,
            rsi_threshold=(
                float(cfg.rsi_threshold)
                if cfg.rsi_threshold is not None
                else self.settings.strategy.rsi_threshold
            ),
            allow_long=cfg.allow_long,
            allow_short=cfg.allow_short,
        )
        strategy = LuxAlgoStrategy(strategy_cfg)

        # Warmup: EMA200 + swing pivot için minimum bar
        warmup = max(200, strategy_cfg.swing_lookback * 2 + 10)
        if len(df) <= warmup:
            return self._empty_result(cfg, df)

        equity = cfg.initial_equity
        open_pos: Position | None = None
        open_pos_entry_bar: int = 0
        trail_dist: float = 0.0
        pos_id_seq = 1
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [equity]

        total = len(df)
        lev = self.settings.risk.leverage
        htf_ts = None
        if htf_df is not None and not htf_df.empty:
            htf_ts = htf_df["ts"].to_numpy(dtype=float)

        for i in range(warmup, total):
            if progress_callback and i % 1000 == 0:
                pct = (i - warmup) / max(1, total - warmup)
                progress_callback(f"Bar {i:,}/{total:,}...", pct)

            # Sadece o ana kadar görülen barlar — look-ahead bias yok
            bar_df = df.iloc[:i + 1]
            close = float(df["close"].iloc[i])
            bar_ts = float(df["ts"].iloc[i])
            bar_time = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
            htf_bar_df = None
            if htf_ts is not None:
                htf_end = int(np.searchsorted(htf_ts, bar_ts, side="right"))
                if htf_end > 0:
                    htf_bar_df = htf_df.iloc[:htf_end]

            # ── Açık pozisyon takibi ───────────────────────────────────────────
            if open_pos is not None:
                # Trailing SL güncelle
                if trail_dist > 0:
                    if open_pos.side == "long":
                        new_sl = close - trail_dist
                        if new_sl > open_pos.stop_loss:
                            open_pos.stop_loss = new_sl
                    else:
                        new_sl = close + trail_dist
                        if new_sl < open_pos.stop_loss:
                            open_pos.stop_loss = new_sl

                # SL / TP kontrolü
                close_reason = ""
                if open_pos.side == "long":
                    if close <= open_pos.stop_loss:
                        close_reason = "SL"
                    elif close >= open_pos.take_profit:
                        close_reason = "TP"
                else:
                    if close >= open_pos.stop_loss:
                        close_reason = "SL"
                    elif close <= open_pos.take_profit:
                        close_reason = "TP"

                # Timeout: max_position_duration_hours
                if not close_reason:
                    age_bars = i - open_pos_entry_bar
                    tf_mins = _tf_to_minutes(cfg.timeframe)
                    age_hours = (age_bars * tf_mins) / 60.0
                    max_h = self.settings.risk.max_position_duration_hours
                    if max_h > 0 and age_hours >= max_h:
                        close_reason = "TIMEOUT"

                if close_reason:
                    exit_price = close
                    if open_pos.side == "long":
                        raw_pnl = (exit_price - open_pos.entry_price) * open_pos.qty * lev
                    else:
                        raw_pnl = (open_pos.entry_price - exit_price) * open_pos.qty * lev

                    entry_notional = open_pos.entry_price * open_pos.qty
                    exit_notional = exit_price * open_pos.qty
                    fee = self.risk.fee_cost(entry_notional) + self.risk.fee_cost(exit_notional)
                    pnl_after_fee = raw_pnl - fee

                    equity += pnl_after_fee
                    equity_curve.append(equity)

                    trades.append(BacktestTrade(
                        position_id=open_pos.id,
                        symbol=open_pos.symbol,
                        side=open_pos.side,
                        entry_price=open_pos.entry_price,
                        exit_price=exit_price,
                        qty=open_pos.qty,
                        pnl_usd=pnl_after_fee,
                        pnl_pct=(pnl_after_fee / max(1e-9, open_pos.entry_price * open_pos.qty)) * 100,
                        entry_bar=open_pos_entry_bar,
                        exit_bar=i,
                        entry_time=open_pos.opened_at,
                        exit_time=bar_time,
                        reason=close_reason,
                    ))
                    open_pos = None

            # ── Sinyal & pozisyon açma ─────────────────────────────────────────
            if open_pos is None:
                signal = strategy.next_signal(bar_df, htf_df=htf_bar_df)
                if signal is not None:
                    sizing = self.risk.size_position(signal, equity, bar_df, [])
                    if sizing is not None:
                        open_pos = Position(
                            id=pos_id_seq,
                            symbol=cfg.symbol,
                            side=signal.side,
                            qty=sizing.qty,
                            entry_price=signal.price,
                            stop_loss=sizing.stop_loss,
                            take_profit=sizing.take_profit,
                            opened_at=bar_time,
                        )
                        open_pos_entry_bar = i
                        trail_dist = abs(signal.price - sizing.stop_loss)
                        pos_id_seq += 1

        # Son açık pozisyonu son bar kapanışında kapat
        if open_pos is not None:
            exit_price = float(df["close"].iloc[-1])
            if open_pos.side == "long":
                raw_pnl = (exit_price - open_pos.entry_price) * open_pos.qty * lev
            else:
                raw_pnl = (open_pos.entry_price - exit_price) * open_pos.qty * lev
            entry_notional = open_pos.entry_price * open_pos.qty
            exit_notional = exit_price * open_pos.qty
            fee = self.risk.fee_cost(entry_notional) + self.risk.fee_cost(exit_notional)
            pnl_after_fee = raw_pnl - fee
            equity += pnl_after_fee
            equity_curve.append(equity)
            trades.append(BacktestTrade(
                position_id=open_pos.id,
                symbol=open_pos.symbol,
                side=open_pos.side,
                entry_price=open_pos.entry_price,
                exit_price=exit_price,
                qty=open_pos.qty,
                pnl_usd=pnl_after_fee,
                pnl_pct=(pnl_after_fee / max(1e-9, open_pos.entry_price * open_pos.qty)) * 100,
                entry_bar=open_pos_entry_bar,
                exit_bar=total - 1,
                entry_time=open_pos.opened_at,
                exit_time=datetime.fromtimestamp(
                    float(df["ts"].iloc[-1]) / 1000, tz=timezone.utc
                ),
                reason="EOD",
            ))

        if progress_callback:
            progress_callback(f"Tamamlandı — {len(trades)} trade", 1.0)

        return self._build_result(cfg, df, trades, equity_curve, cfg.initial_equity)

    # ── Metrik hesabı ──────────────────────────────────────────────────────────

    def run_multi(
        self,
        symbols: list[str],
        cfg: BacktestConfig,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> MultiBacktestResult:
        clean_symbols = [s for s in symbols if s]
        results: list[BacktestResult] = []
        per_symbol_equity = cfg.initial_equity / max(1, len(clean_symbols))

        for idx, symbol in enumerate(clean_symbols, start=1):
            if progress_callback:
                progress_callback(
                    f"Sembol hazirlaniyor ({idx}/{len(clean_symbols)}): {symbol}",
                    (idx - 1) / max(1, len(clean_symbols)),
                )

            sym_cfg = replace(cfg, symbol=symbol, initial_equity=per_symbol_equity)
            df = self.fetch_data(sym_cfg)
            if df.empty:
                continue
            htf_df = self.fetch_htf_data(sym_cfg)
            result = self.run(sym_cfg, df, htf_df=htf_df)
            results.append(result)

            if progress_callback:
                progress_callback(
                    f"Sembol tamamlandi ({idx}/{len(clean_symbols)}): {symbol}",
                    idx / max(1, len(clean_symbols)),
                )

        summary = self._combine_results(cfg, results)
        return MultiBacktestResult(summary=summary, results=results)

    def _combine_results(
        self,
        cfg: BacktestConfig,
        results: list[BacktestResult],
    ) -> BacktestResult:
        if not results:
            return self._empty_result(cfg, pd.DataFrame())

        all_trades = sorted(
            [trade for result in results for trade in result.trades],
            key=lambda t: t.exit_time,
        )
        total_initial = sum(r.initial_equity for r in results)
        running_equity = total_initial
        equity_curve = [running_equity]
        for trade in all_trades:
            running_equity += trade.pnl_usd
            equity_curve.append(running_equity)

        start_date = min(r.start_date for r in results)
        end_date = max(r.end_date for r in results)
        total_bars = sum(r.total_bars for r in results)
        summary_cfg = replace(cfg, symbol=f"MAJOR-{len(results)}")

        summary = self._build_result(
            summary_cfg,
            pd.DataFrame({"ts": [start_date.timestamp() * 1000, end_date.timestamp() * 1000]}),
            all_trades,
            equity_curve,
            total_initial,
        )
        summary.start_date = start_date
        summary.end_date = end_date
        summary.total_bars = total_bars
        return summary

    def _build_result(
        self,
        cfg: BacktestConfig,
        df: pd.DataFrame,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        initial_equity: float,
    ) -> BacktestResult:
        total = len(trades)
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]

        win_rate = len(wins) / total if total > 0 else 0.0
        net_pnl = sum(t.pnl_usd for t in trades)
        net_pnl_pct = (net_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0
        final_equity = initial_equity + net_pnl

        # Max drawdown
        peak = initial_equity
        max_dd = 0.0
        running = initial_equity
        for t in trades:
            running += t.pnl_usd
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Sharpe (trade PNL % bazlı)
        sharpe = 0.0
        if len(trades) > 1:
            returns = np.array([t.pnl_pct for t in trades])
            std = returns.std()
            if std > 0:
                sharpe = float(returns.mean() / std * np.sqrt(252))

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0

        start_dt = datetime.fromtimestamp(float(df["ts"].iloc[0]) / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(float(df["ts"].iloc[-1]) / 1000, tz=timezone.utc)

        return BacktestResult(
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            start_date=start_dt,
            end_date=end_dt,
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_bars=len(df),
            total_trades=total,
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=win_rate,
            net_pnl_usd=net_pnl,
            net_pnl_pct=net_pnl_pct,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win_usd=avg_win,
            avg_loss_usd=avg_loss,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _empty_result(self, cfg: BacktestConfig, df: pd.DataFrame) -> BacktestResult:
        now = datetime.now(timezone.utc)
        return BacktestResult(
            symbol=cfg.symbol, timeframe=cfg.timeframe,
            start_date=now, end_date=now,
            initial_equity=cfg.initial_equity, final_equity=cfg.initial_equity,
            total_bars=len(df), total_trades=0,
            win_count=0, loss_count=0, win_rate=0.0,
            net_pnl_usd=0.0, net_pnl_pct=0.0,
            max_drawdown_pct=0.0, sharpe_ratio=0.0,
            profit_factor=0.0, avg_win_usd=0.0, avg_loss_usd=0.0,
        )


# ─── Yardımcı ─────────────────────────────────────────────────────────────────

def _tf_to_minutes(tf: str) -> float:
    """'5m' → 5.0, '1h' → 60.0, '1d' → 1440.0"""
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return float(tf[:-1])
    if tf.endswith("h"):
        return float(tf[:-1]) * 60
    if tf.endswith("d"):
        return float(tf[:-1]) * 1440
    return 5.0  # fallback
