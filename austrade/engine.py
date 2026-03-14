from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time

import ccxt

from .config import Settings
from .exchange import ExchangeClient
from .logging_utils import get_logger
from .models import Position, TradeRecord
from .risk import RiskManager
from .signal_scoring import ScoreCard, SignalScorer
from .storage import Storage
from .strategy_lux import LuxAlgoStrategy

logger = get_logger(__name__)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_sec: float = 300.0
    failure_count: int = 0
    state: str = "closed"
    open_at: float = 0.0

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.open_at = time.time()
            logger.error("Circuit breaker OPEN after %s failures", self.failure_count)

    def record_success(self) -> None:
        if self.state != "closed":
            logger.info("Circuit breaker CLOSED")
        self.failure_count = 0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state == "closed":
            return False
        if self.state == "open" and time.time() - self.open_at > self.recovery_sec:
            self.state = "half_open"
            logger.info("Circuit breaker HALF-OPEN, trying recovery")
            return False
        return self.state == "open"


@dataclass(slots=True)
class EngineSnapshot:
    running: bool
    exchange: str
    symbol: str
    symbols: list[str]
    paper_mode: bool
    leverage: int
    equity: float
    cash: float
    daily_pnl: float
    total_pnl: float
    last_prices: dict[str, float]
    last_changes: dict[str, float]
    open_positions: list[Position] = field(default_factory=list)
    recent_trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    last_signal: str = "-"
    last_error: str = ""
    daily_loss_halted: bool = False


_CORRELATED_CLUSTERS: list[set[str]] = [
    {"BTC/USDT", "ETH/USDT"},
    {"SOL/USDT", "AVAX/USDT", "NEAR/USDT"},
]


class TradeEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exchange = ExchangeClient(settings.exchange)
        self.risk = RiskManager(settings.risk)
        self.signal_scorer = SignalScorer(settings, self.exchange)
        self.storage = Storage(settings.storage.db_path)

        try:
            from .telegram_notifier import TelegramNotifier

            self.telegram: TelegramNotifier | None = TelegramNotifier(settings.telegram)
        except Exception:  # noqa: BLE001
            self.telegram = None

        self.symbols = self._initial_symbols()
        logger.info("Strateji: LuxAlgo MCP proxy")
        self.strategies: dict[str, LuxAlgoStrategy] = {
            symbol: LuxAlgoStrategy(settings.strategy)
            for symbol in self.symbols
        }

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.position_seq = 1
        self.cash = settings.app.starting_balance_usd
        self.equity = self.cash
        self.last_prices: dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_changes: dict[str, float] = {s: 0.0 for s in self.symbols}
        self.open_positions: list[Position] = []
        self.trail_distance: dict[int, float] = {}
        self.cooldown_until_ts: dict[str, int] = {}
        self.last_signal = "-"
        self.last_error = ""
        self.max_retries = 3
        self.daily_loss_halted = False
        self.circuit_breaker = CircuitBreaker()

        self._sync_positions()

    def _initial_symbols(self) -> list[str]:
        if self.settings.exchange.symbols:
            return list(self.settings.exchange.symbols[: self.settings.exchange.symbol_count])
        try:
            universe = self.exchange.fetch_universe_symbols(self.settings.exchange.symbol_count)
            if universe:
                return universe
        except Exception as exc:  # noqa: BLE001
            logger.warning("Universe selection failed, fallback symbol: %s", exc)
        return [self.settings.exchange.symbol]

    def _sync_positions(self) -> None:
        db_positions = self.storage.load_open_positions()
        if db_positions:
            max_hours = self.settings.risk.max_position_duration_hours
            now = datetime.now(timezone.utc)
            valid_positions: list[Position] = []
            for pos in db_positions:
                if max_hours > 0:
                    age_hours = (now - pos.opened_at).total_seconds() / 3600.0
                    if age_hours >= max_hours:
                        logger.warning(
                            "Startup: Eski pozisyon temizlendi (timeout) id=%s symbol=%s age=%.1fh",
                            pos.id,
                            pos.symbol,
                            age_hours,
                        )
                        self.storage.delete_position(pos.id)
                        continue
                valid_positions.append(pos)
            self.open_positions = valid_positions
            if valid_positions:
                self.position_seq = max(p.id for p in valid_positions) + 1
                for pos in valid_positions:
                    self.trail_distance[pos.id] = abs(pos.entry_price - pos.stop_loss)
            logger.info(
                "Synced %s positions from DB on startup (%s stale removed)",
                len(valid_positions),
                len(db_positions) - len(valid_positions),
            )

        if not self.settings.app.paper_mode and db_positions:
            try:
                ex_positions = self.exchange.fetch_open_positions()
                ex_symbols = {
                    self.exchange.base_symbol(p.get("symbol", ""))
                    for p in ex_positions
                }
                still_open: list[Position] = []
                for pos in self.open_positions:
                    if pos.symbol in ex_symbols:
                        still_open.append(pos)
                    else:
                        logger.warning(
                            "Position %s %s not found on exchange, marking closed",
                            pos.id,
                            pos.symbol,
                        )
                        self.storage.delete_position(pos.id)
                self.open_positions = still_open
            except Exception as exc:  # noqa: BLE001
                logger.warning("Exchange position sync failed: %s", exc)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Engine started")

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info("Engine stopped")

    def _fetch_all_ohlcv(self, symbol_list: list[str], timeframe: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {}
        tf = timeframe or self.settings.exchange.timeframe
        with ThreadPoolExecutor(max_workers=min(4, len(symbol_list))) as ex:
            futures = {
                ex.submit(self.exchange.fetch_ohlcv_for_timeframe, s, tf, self.settings.exchange.limit): s
                for s in symbol_list
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    result[sym] = fut.result(timeout=15)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fetch_ohlcv timeout/error: %s - %s", sym, exc)
        return result

    def _loop(self) -> None:
        retry = 0
        while self.running:
            try:
                if self.circuit_breaker.is_open():
                    time.sleep(self.settings.app.refresh_seconds)
                    continue

                with self.lock:
                    halted = self._check_daily_loss_limit() or self._check_drawdown_limit()
                if halted:
                    time.sleep(self.settings.app.refresh_seconds)
                    continue

                symbol_list = list(self.symbols)
                market_snap = self.exchange.fetch_market_snapshot(symbol_list)
                ohlcv_map = self._fetch_all_ohlcv(symbol_list, self.settings.exchange.timeframe)
                htf_tf = self.settings.strategy.htf_timeframe
                if self.settings.strategy.htf_trend_filter and htf_tf != self.settings.exchange.timeframe:
                    htf_map = self._fetch_all_ohlcv(symbol_list, htf_tf)
                else:
                    htf_map = ohlcv_map
                btc_df = ohlcv_map.get("BTC/USDT")

                for symbol in symbol_list:
                    df = ohlcv_map.get(symbol)
                    if df is None or df.empty:
                        continue

                    htf_df = htf_map.get(symbol)
                    signal = self.strategies[symbol].next_signal(df, htf_df=htf_df)
                    price = float(df["close"].iloc[-1])
                    bar_ts = int(df["ts"].iloc[-1])

                    snap_info = market_snap.get(symbol, {})
                    last = float(snap_info.get("last", 0.0))
                    change = float(snap_info.get("change", 0.0))

                    with self.lock:
                        self.last_prices[symbol] = last if last > 0 else price
                        self.last_changes[symbol] = change
                        self._update_open_positions_for_symbol(symbol, self.last_prices[symbol], bar_ts)
                        if signal:
                            scorecard = self.signal_scorer.score_signal(
                                signal,
                                symbol,
                                df,
                                btc_df if btc_df is not None else df,
                            )
                            signal.score = scorecard.total
                            signal.context["score_total"] = scorecard.total
                            signal.context["score_threshold"] = scorecard.threshold
                            signal.context["score_passed"] = scorecard.passed
                            signal.context["score_summary"] = scorecard.summary()
                            signal.note = f"{signal.note} | {scorecard.summary()}"
                            self.last_signal = (
                                f"{symbol} {signal.kind.upper()} {signal.side.upper()} "
                                f"@ {signal.price:.2f} | {scorecard.total:.1f}"
                            )
                            logger.info("Signal produced: %s", self.last_signal)
                            self._maybe_open_position(symbol, signal, df, scorecard)

                with self.lock:
                    self.equity = self.cash + self._unrealized_total()
                    self.storage.add_snapshot(self.equity, self.storage.today_pnl())
                    self.last_error = ""
                retry = 0
                self.circuit_breaker.record_success()
            except ccxt.NetworkError as exc:
                retry += 1
                self.last_error = f"NetworkError: {exc}"
                logger.warning("Network error (%s/%s): %s", retry, self.max_retries, exc)
                if retry >= self.max_retries:
                    self._notify(f"API hatasi: NetworkError: {exc}")
                self.circuit_breaker.record_failure()
            except ccxt.ExchangeError as exc:
                retry += 1
                self.last_error = f"ExchangeError: {exc}"
                logger.error("Exchange error (%s/%s): %s", retry, self.max_retries, exc)
                if retry >= self.max_retries:
                    self._notify(f"API hatasi: ExchangeError: {exc}")
                self.circuit_breaker.record_failure()
            except (ValueError, KeyError, IndexError) as exc:
                retry += 1
                self.last_error = f"DataError: {exc}"
                logger.warning("Data error (%s/%s): %s", retry, self.max_retries, exc)
            except Exception as exc:  # noqa: BLE001
                retry += 1
                self.last_error = str(exc)
                logger.exception("Unexpected engine error: %s", exc)
                if retry >= self.max_retries:
                    self._notify(f"Kritik hata: {exc}")
                self.circuit_breaker.record_failure()

            retry = min(retry, self.max_retries)
            backoff = min(2**retry, 20)
            sleep_for = self.settings.app.refresh_seconds if retry == 0 else backoff
            time.sleep(sleep_for)

    def _check_daily_loss_limit(self) -> bool:
        max_loss_pct = self.settings.risk.max_daily_loss_pct
        if max_loss_pct <= 0:
            return False
        daily_pnl = self.storage.today_pnl()
        # Denominator: günlük başlangıç equity'si olarak starting_balance veya
        # equity'nin en yüksek değerini kullan (daha doğru günlük kayıp hesabı)
        peak = self.storage.peak_equity()
        base_equity = max(self.settings.app.starting_balance_usd, self.equity, peak if peak > 0 else 0.0)
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / max(1.0, base_equity) * 100.0
            if loss_pct >= max_loss_pct:
                if not self.daily_loss_halted:
                    self.daily_loss_halted = True
                    msg = (
                        f"Gunluk zarar limiti asildi. "
                        f"PNL: ${daily_pnl:.2f} (-%{loss_pct:.1f} / base=${base_equity:.2f}). "
                        f"Bot yeni pozisyon acmiyor."
                    )
                    logger.warning(msg)
                    self._notify(msg)
                return True
        self.daily_loss_halted = False
        return False

    def _check_drawdown_limit(self) -> bool:
        max_dd_pct = self.settings.risk.max_drawdown_pct
        if max_dd_pct <= 0:
            return False
        peak = self.storage.peak_equity()
        if peak > 0 and self.equity > 0:
            dd_pct = (peak - self.equity) / peak * 100.0
            if dd_pct >= max_dd_pct:
                self._notify(
                    f"Max drawdown asildi. "
                    f"Dusus: %{dd_pct:.1f} (peak=${peak:.2f}, equity=${self.equity:.2f})"
                )
                return True
        return False

    def _count_consecutive_losses(self) -> int:
        trades = self.storage.recent_trades(20)
        count = 0
        for trade in trades:
            if trade.get("pnl_usd", 0) < 0:
                count += 1
            else:
                break
        return count

    def _is_correlated_with_open(self, symbol: str) -> bool:
        for cluster in _CORRELATED_CLUSTERS:
            if symbol in cluster:
                for pos in self.open_positions:
                    if pos.symbol in cluster and pos.symbol != symbol:
                        return True
        return False

    def _check_portfolio_notional_cap(self, symbol: str, new_qty: float, new_price: float) -> bool:
        portfolio = self.settings.portfolio
        if portfolio is None:
            return False
        max_notional_pct = portfolio.max_notional_pct
        total_notional = sum(
            p.qty * self.last_prices.get(p.symbol, p.entry_price)
            for p in self.open_positions
        )
        new_notional = new_qty * new_price
        cap = self.equity * (max_notional_pct / 100.0)
        return (total_notional + new_notional) > cap

    def _timeframe_minutes(self) -> float:
        tf = str(self.settings.exchange.timeframe).strip().lower()
        if tf.endswith("m"):
            return float(tf[:-1])
        if tf.endswith("h"):
            return float(tf[:-1]) * 60.0
        if tf.endswith("d"):
            return float(tf[:-1]) * 1440.0
        return 5.0

    def _start_cooldown(self, symbol: str, bar_ts: int) -> None:
        cooldown_bars = max(int(self.settings.strategy.cooldown_bars), 0)
        if cooldown_bars <= 0:
            return
        self.cooldown_until_ts[symbol] = int(
            bar_ts + (self._timeframe_minutes() * 60_000 * cooldown_bars)
        )

    def _cooldown_active(self, symbol: str, bar_ts: int) -> bool:
        return int(bar_ts) < self.cooldown_until_ts.get(symbol, 0)

    def _maybe_open_position(self, symbol: str, signal, df, scorecard: ScoreCard | None = None) -> None:
        bar_ts = int(df["ts"].iloc[-1])
        if self.daily_loss_halted:
            return
        if self._cooldown_active(symbol, bar_ts):
            return
        if len(self.open_positions) >= self.settings.risk.max_open_positions:
            return
        if any(p.symbol == symbol for p in self.open_positions):
            return
        if self._is_correlated_with_open(symbol):
            return

        max_consec = self.settings.risk.max_consecutive_losses
        if max_consec > 0 and self._count_consecutive_losses() >= max_consec:
            return

        if scorecard is not None:
            logger.info(
                "Signal score: %s %s score=%.1f [%s]",
                symbol,
                signal.side,
                scorecard.total,
                scorecard.summary(),
            )

        consecutive_losses = self._count_consecutive_losses()
        sizing = self.risk.size_position(signal, self.equity, df, self.open_positions, consecutive_losses)
        if not sizing:
            return
        if self._check_portfolio_notional_cap(symbol, sizing.qty, signal.price):
            return

        qty = sizing.qty
        entry_price = signal.price
        if not self.settings.app.paper_mode:
            min_amt = self.exchange.min_order_amount(symbol)
            rounded_qty = float(self.exchange.amount_to_precision(symbol, qty))
            if rounded_qty <= 0 or (min_amt > 0 and rounded_qty < min_amt):
                return
            qty = rounded_qty
            entry_price = self.exchange.open_position(symbol, signal.side, qty)

        pos = Position(
            id=self.position_seq,
            symbol=symbol,
            side=signal.side,
            qty=qty,
            entry_price=entry_price,
            stop_loss=sizing.stop_loss,
            take_profit=sizing.tp1_price,
            opened_at=datetime.now(timezone.utc),
            initial_qty=qty,
            tp1_price=sizing.tp1_price,
            tp1_hit=False,
            break_even_price=entry_price,
            trail_callback_pct=sizing.trail_callback_pct,
            trail_activation_price=sizing.trail_activation_price,
            trailing_active=False,
        )
        self.position_seq += 1
        self.open_positions.append(pos)
        self.trail_distance[pos.id] = sizing.stop_distance
        self.storage.save_position(pos)
        self._start_cooldown(symbol, bar_ts)

        strategy_tag = signal.kind.upper()
        logger.info(
            "Position opened [%s]: id=%s symbol=%s side=%s qty=%.6f entry=%.4f sl=%.4f tp1=%.4f",
            strategy_tag,
            pos.id,
            pos.symbol,
            pos.side,
            pos.qty,
            pos.entry_price,
            pos.stop_loss,
            pos.tp1_price,
        )
        self._notify(
            f"Pozisyon acildi [{strategy_tag}]\n"
            f"Sembol: {symbol} | Yon: {signal.side.upper()}\n"
            f"Giris: {entry_price:.4f} | SL: {pos.stop_loss:.4f} | TP1: {pos.tp1_price:.4f}\n"
            f"Skor: {signal.score:.1f}/100"
        )

    def _close_trade(self, pos: Position, exit_price: float, reason: str, qty: float) -> float:
        pnl = (
            (exit_price - pos.entry_price) * qty
            if pos.side == "long"
            else (pos.entry_price - exit_price) * qty
        )
        entry_notional = pos.entry_price * qty
        exit_notional = exit_price * qty
        fee = self.risk.fee_cost(entry_notional) + self.risk.fee_cost(exit_notional)
        pnl_after_fee = pnl - fee
        self.cash += pnl_after_fee
        self.storage.add_trade(
            TradeRecord(
                position_id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=qty,
                pnl_usd=pnl_after_fee,
                pnl_pct=(pnl_after_fee / max(1e-9, pos.entry_price * qty)) * 100,
                opened_at=pos.opened_at,
                closed_at=datetime.now(timezone.utc),
                reason=reason,
            )
        )
        return pnl_after_fee

    def _update_trailing_stop(self, pos: Position, price: float) -> None:
        if not pos.trailing_active or pos.trail_callback_pct <= 0:
            return
        callback_ratio = pos.trail_callback_pct / 100.0
        # Orijinal ATR stop mesafesini de dikkate al (trail_distance sözlüğünden)
        atr_distance = self.trail_distance.get(pos.id, 0.0)
        if pos.side == "long":
            sl_pct = price * (1.0 - callback_ratio)
            # ATR tabanlı mesafe varsa ikisinin maksimumunu kullan (daha dar stop)
            sl_atr = (price - atr_distance) if atr_distance > 0 else sl_pct
            new_sl = max(sl_pct, sl_atr)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                self.storage.update_position(pos)
        else:
            sl_pct = price * (1.0 + callback_ratio)
            sl_atr = (price + atr_distance) if atr_distance > 0 else sl_pct
            new_sl = min(sl_pct, sl_atr)
            if new_sl < pos.stop_loss:
                pos.stop_loss = new_sl
                self.storage.update_position(pos)

    def _update_open_positions_for_symbol(self, symbol: str, price: float, bar_ts: int) -> None:
        remaining: list[Position] = []
        for pos in self.open_positions:
            if pos.symbol != symbol:
                remaining.append(pos)
                continue

            if not pos.tp1_hit:
                tp1_reached = price >= pos.tp1_price if pos.side == "long" else price <= pos.tp1_price
                if tp1_reached:
                    close_qty = min(
                        pos.qty,
                        max(pos.initial_qty * (self.settings.risk.tp1_close_pct / 100.0), 0.0),
                    )
                    if close_qty > 0:
                        exit_price = price
                        if not self.settings.app.paper_mode:
                            exit_price = self.exchange.close_position(pos.symbol, pos.side, close_qty)
                        pnl_after_fee = self._close_trade(pos, exit_price, "TP1", close_qty)
                        pos.qty -= close_qty
                        pos.tp1_hit = True
                        pos.stop_loss = pos.break_even_price
                        # Trailing aktif et: TP1 seviyesinde zaten aktivasyon fiyatı geçildi
                        pos.trailing_active = True
                        if pos.qty <= 0:
                            self.storage.delete_position(pos.id)
                            self.trail_distance.pop(pos.id, None)
                            self._start_cooldown(pos.symbol, bar_ts)
                            self._notify(
                                f"Pozisyon kapandi\n"
                                f"Sembol: {pos.symbol} | Yon: {pos.side.upper()} | Sebep: TP1\n"
                                f"PNL: ${pnl_after_fee:.4f}"
                            )
                            continue
                        self.storage.update_position(pos)
                        self._notify(
                            f"Pozisyon parcali kapandi\n"
                            f"Sembol: {pos.symbol} | Yon: {pos.side.upper()} | Sebep: TP1\n"
                            f"Kalan miktar: {pos.qty:.6f} | PNL: ${pnl_after_fee:.4f}"
                        )

            if not pos.trailing_active:
                activation_hit = (
                    price >= pos.trail_activation_price
                    if pos.side == "long"
                    else price <= pos.trail_activation_price
                )
                if activation_hit:
                    pos.trailing_active = True
                    self.storage.update_position(pos)

            self._update_trailing_stop(pos, price)

            close_reason = ""
            if pos.side == "long":
                if price <= pos.stop_loss:
                    close_reason = "SL"
            else:
                if price >= pos.stop_loss:
                    close_reason = "SL"

            if not close_reason:
                max_hours = self.settings.risk.max_position_duration_hours
                if max_hours > 0:
                    age_hours = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600.0
                    if age_hours >= max_hours:
                        close_reason = "TIMEOUT"

            if not close_reason:
                remaining.append(pos)
                continue

            exit_price = price
            if not self.settings.app.paper_mode:
                exit_price = self.exchange.close_position(pos.symbol, pos.side, pos.qty)
            pnl_after_fee = self._close_trade(pos, exit_price, close_reason, pos.qty)
            self.storage.delete_position(pos.id)
            self.trail_distance.pop(pos.id, None)
            self._start_cooldown(pos.symbol, bar_ts)
            logger.info(
                "Position closed: id=%s symbol=%s reason=%s pnl=%.4f",
                pos.id,
                pos.symbol,
                close_reason,
                pnl_after_fee,
            )
            self._notify(
                f"Pozisyon kapandi\n"
                f"Sembol: {pos.symbol} | Yon: {pos.side.upper()} | Sebep: {close_reason}\n"
                f"PNL: ${pnl_after_fee:.4f}"
            )

        self.open_positions = remaining

    def _realized_pnl(self, pos: Position, price: float) -> float:
        if pos.side == "long":
            return (price - pos.entry_price) * pos.qty
        return (pos.entry_price - price) * pos.qty

    def _unrealized_total(self) -> float:
        total = 0.0
        for pos in self.open_positions:
            current = self.last_prices.get(pos.symbol, 0.0)
            if current > 0:
                total += self._realized_pnl(pos, current)
        return total

    def _notify(self, message: str) -> None:
        if self.telegram:
            try:
                self.telegram.send(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Telegram notify failed: %s", exc)

    def snapshot(self) -> EngineSnapshot:
        with self.lock:
            return EngineSnapshot(
                running=self.running,
                exchange=self.settings.exchange.name,
                symbol=self.settings.exchange.symbol,
                symbols=list(self.symbols),
                paper_mode=self.settings.app.paper_mode,
                leverage=self.settings.risk.leverage,
                equity=self.equity,
                cash=self.cash,
                daily_pnl=self.storage.today_pnl(),
                total_pnl=self.storage.total_pnl(),
                last_prices=dict(self.last_prices),
                last_changes=dict(self.last_changes),
                open_positions=list(self.open_positions),
                recent_trades=self.storage.recent_trades(100),
                equity_curve=self.storage.equity_curve(200),
                last_signal=self.last_signal,
                last_error=self.last_error,
                daily_loss_halted=self.daily_loss_halted,
            )
