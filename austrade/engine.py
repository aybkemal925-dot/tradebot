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
            logger.info("Circuit breaker HALF-OPEN — trying recovery")
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


# Korelasyonlu sembol kümeleri — aynı kümeden aynı anda 1 pozisyon
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

        # Telegram notifier (lazy import to avoid breaking without requests)
        try:
            from .telegram_notifier import TelegramNotifier
            self.telegram: TelegramNotifier | None = TelegramNotifier(settings.telegram)
        except Exception:  # noqa: BLE001
            self.telegram = None

        self.symbols = self._initial_symbols()

        # Strateji seçimi: Triple Confirmation aktifse onu, değilse SMC kullan
        logger.info("Strateji: LuxAlgo")
        self.strategies: dict = {
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
        self.last_signal = "-"
        self.last_error = ""
        self.max_retries = 3
        self.daily_loss_halted = False
        self.circuit_breaker = CircuitBreaker()

        # Restart'ta pozisyonları DB'den yükle
        self._sync_positions()

    # ─── Startup ─────────────────────────────────────────────────────────────

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
        """Bot yeniden başlatılınca açık pozisyonları yükle."""
        db_positions = self.storage.load_open_positions()

        if db_positions:
            self.open_positions = db_positions
            self.position_seq = max(p.id for p in db_positions) + 1
            for pos in db_positions:
                self.trail_distance[pos.id] = abs(pos.entry_price - pos.stop_loss)
            logger.info("Synced %s positions from DB on startup", len(db_positions))

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
                            "Position %s %s not found on exchange — marking closed",
                            pos.id, pos.symbol,
                        )
                        self.storage.delete_position(pos.id)
                self.open_positions = still_open
            except Exception as exc:  # noqa: BLE001
                logger.warning("Exchange position sync failed: %s", exc)

    # ─── Engine control ───────────────────────────────────────────────────────

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

    # ─── Main loop ────────────────────────────────────────────────────────────

    def _fetch_all_ohlcv(self, symbol_list: list[str], timeframe: str | None = None) -> dict[str, object]:
        """Sembolleri paralel olarak fetch et (4x hız artışı)."""
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
                except Exception as e:  # noqa: BLE001
                    logger.warning("fetch_ohlcv timeout/error: %s — %s", sym, e)
        return result

    def _loop(self) -> None:
        retry = 0
        while self.running:
            try:
                if self.circuit_breaker.is_open():
                    logger.warning("Circuit breaker open — skipping loop iteration")
                    time.sleep(self.settings.app.refresh_seconds)
                    continue

                # Günlük zarar limiti ve drawdown kontrolü
                with self.lock:
                    halted = self._check_daily_loss_limit() or self._check_drawdown_limit()
                if halted:
                    time.sleep(self.settings.app.refresh_seconds)
                    continue

                symbol_list = list(self.symbols)
                market_snap = self.exchange.fetch_market_snapshot(symbol_list)

                # Paralel OHLCV fetch
                ohlcv_map = self._fetch_all_ohlcv(symbol_list, self.settings.exchange.timeframe)
                htf_tf = self.settings.strategy.htf_timeframe
                if self.settings.strategy.htf_trend_filter and htf_tf != self.settings.exchange.timeframe:
                    htf_map = self._fetch_all_ohlcv(symbol_list, htf_tf)
                else:
                    htf_map = ohlcv_map
                btc_df = ohlcv_map.get("BTC/USDT")
                if btc_df is None or btc_df.empty:
                    try:
                        btc_df = self.exchange.fetch_ohlcv_for_timeframe(
                            "BTC/USDT",
                            self.settings.exchange.timeframe,
                            self.settings.exchange.limit,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("BTC regime fetch failed: %s", exc)
                        btc_df = None

                for symbol in symbol_list:
                    df = ohlcv_map.get(symbol)
                    if df is None or df.empty:
                        logger.warning("Empty/missing OHLCV for %s — skipping", symbol)
                        continue

                    htf_df = htf_map.get(symbol)
                    signal = self.strategies[symbol].next_signal(df, htf_df=htf_df)
                    price = float(df["close"].iloc[-1])

                    snap_info = market_snap.get(symbol, {})
                    last = float(snap_info.get("last", 0.0))
                    change = float(snap_info.get("change", 0.0))

                    with self.lock:
                        self.last_prices[symbol] = last if last > 0 else price
                        self.last_changes[symbol] = change
                        self._update_open_positions_for_symbol(symbol, self.last_prices[symbol])
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
                self.circuit_breaker.record_failure()
            except ccxt.ExchangeError as exc:
                retry += 1
                self.last_error = f"ExchangeError: {exc}"
                logger.error("Exchange error (%s/%s): %s", retry, self.max_retries, exc)
                self.circuit_breaker.record_failure()
            except (ValueError, KeyError, IndexError) as exc:
                retry += 1
                self.last_error = f"DataError: {exc}"
                logger.warning("Data error (%s/%s): %s", retry, self.max_retries, exc)
            except Exception as exc:  # noqa: BLE001
                retry += 1
                self.last_error = str(exc)
                logger.exception("Unexpected engine error: %s", exc)
                self.circuit_breaker.record_failure()

            retry = min(retry, self.max_retries)
            backoff = min(2**retry, 20)
            sleep_for = self.settings.app.refresh_seconds if retry == 0 else backoff
            logger.info("Engine loop sleep: %ss", sleep_for)
            time.sleep(sleep_for)

    # ─── Daily loss + drawdown limits ─────────────────────────────────────────

    def _check_daily_loss_limit(self) -> bool:
        """Günlük zarar limiti aşıldıysa True döndür. Lock içinde çağrılmalı."""
        max_loss_pct = self.settings.risk.max_daily_loss_pct
        if max_loss_pct <= 0:
            return False

        daily_pnl = self.storage.today_pnl()
        # starting_balance sabiti — dinamik cash değil
        starting_balance = self.settings.app.starting_balance_usd
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / max(1.0, starting_balance) * 100.0
            if loss_pct >= max_loss_pct:
                if not self.daily_loss_halted:
                    self.daily_loss_halted = True
                    msg = (
                        f"Günlük zarar limiti aşıldı! "
                        f"PNL: ${daily_pnl:.2f} (-%{loss_pct:.1f}). "
                        f"Bot yeni pozisyon açmıyor."
                    )
                    logger.warning(msg)
                    self._notify(msg)
                return True

        self.daily_loss_halted = False
        return False

    def _check_drawdown_limit(self) -> bool:
        """Max drawdown kontrolü. Lock içinde çağrılmalı."""
        max_dd_pct = self.settings.risk.max_drawdown_pct
        if max_dd_pct <= 0:
            return False
        peak = self.storage.peak_equity()
        if peak > 0 and self.equity > 0:
            dd_pct = (peak - self.equity) / peak * 100.0
            if dd_pct >= max_dd_pct:
                logger.error(
                    "Max drawdown reached: %.1f%% (peak=%.2f current=%.2f)",
                    dd_pct, peak, self.equity,
                )
                self._notify(
                    f"🔴 Max Drawdown Aşıldı!\n"
                    f"Düşüş: %{dd_pct:.1f} (peak=${peak:.2f}, şu an=${self.equity:.2f})\n"
                    f"Bot yeni pozisyon açmıyor."
                )
                return True
        return False

    def _count_consecutive_losses(self) -> int:
        """Son işlemlerden ardışık zarar sayısını döndür."""
        trades = self.storage.recent_trades(20)
        count = 0
        for t in trades:
            if t.get("pnl_usd", 0) < 0:
                count += 1
            else:
                break
        return count

    # ─── Position management ──────────────────────────────────────────────────

    def _is_correlated_with_open(self, symbol: str) -> bool:
        """Aynı korelasyon kümesinden zaten açık pozisyon varsa True döndür."""
        for cluster in _CORRELATED_CLUSTERS:
            if symbol in cluster:
                for pos in self.open_positions:
                    if pos.symbol in cluster and pos.symbol != symbol:
                        logger.info(
                            "Korelasyon filtresi: %s ile %s çakışıyor", symbol, pos.symbol
                        )
                        return True
        return False

    def _check_portfolio_notional_cap(self, symbol: str, new_qty: float, new_price: float) -> bool:
        """Toplam notional cap aşılıyorsa True döndür (pozisyon açma)."""
        portfolio = self.settings.portfolio
        if portfolio is None:
            return False
        max_notional_pct = portfolio.max_notional_pct
        lev = self.settings.risk.leverage
        total_notional = sum(
            p.qty * self.last_prices.get(p.symbol, p.entry_price) * lev
            for p in self.open_positions
        )
        new_notional = new_qty * new_price * lev
        cap = self.equity * (max_notional_pct / 100.0)
        if (total_notional + new_notional) > cap:
            logger.info(
                "Portfolio notional cap aşılıyor: mevcut=%.2f + yeni=%.2f > cap=%.2f — atlanıyor",
                total_notional, new_notional, cap,
            )
            return True
        return False

    def _maybe_open_position(self, symbol: str, signal, df, scorecard: ScoreCard | None = None) -> None:
        if self.daily_loss_halted:
            return
        if len(self.open_positions) >= self.settings.risk.max_open_positions:
            return
        if any(p.symbol == symbol for p in self.open_positions):
            return
        if self._is_correlated_with_open(symbol):
            return

        # Ardışık kayıp kontrolü
        max_consec = self.settings.risk.max_consecutive_losses
        if max_consec > 0 and self._count_consecutive_losses() >= max_consec:
            logger.info(
                "Ardışık kayıp limiti (%s) aşıldı — pozisyon atlanıyor", max_consec
            )
            return

        if scorecard is not None and not scorecard.passed:
            logger.info(
                "Signal skipped by score gate: %s %s score=%.1f threshold=%.1f [%s]",
                symbol,
                signal.side,
                scorecard.total,
                scorecard.threshold,
                scorecard.summary(),
            )
            return

        sizing = self.risk.size_position(signal, self.equity, df, self.open_positions)
        if not sizing:
            return

        # Toplam notional cap kontrolü
        if self._check_portfolio_notional_cap(symbol, sizing.qty, signal.price):
            return

        # ── Triple Confirmation: SL/TP'yi override et (ATR*1.5 / ATR*4.5) ──
        qty = sizing.qty
        entry_price = signal.price

        if not self.settings.app.paper_mode:
            min_amt = self.exchange.min_order_amount(symbol)
            rounded_qty = float(self.exchange.amount_to_precision(symbol, qty))
            if rounded_qty <= 0 or (min_amt > 0 and rounded_qty < min_amt):
                logger.warning(
                    "Order size too small: %s qty=%.6f min=%.6f — skipping",
                    symbol, rounded_qty, min_amt,
                )
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
            take_profit=sizing.take_profit,
            opened_at=datetime.now(timezone.utc),
        )
        self.position_seq += 1
        self.open_positions.append(pos)
        self.trail_distance[pos.id] = abs(entry_price - sizing.stop_loss)

        self.storage.save_position(pos)

        if not self.settings.app.paper_mode:
            self.exchange.place_sl_tp_orders(
                symbol, signal.side, qty, sizing.stop_loss, sizing.take_profit
            )

        strategy_tag = "TRIPLE" if signal.kind == "triple" else signal.kind.upper()
        logger.info(
            "Position opened [%s]: id=%s symbol=%s side=%s qty=%.6f entry=%.4f sl=%.4f tp=%.4f",
            strategy_tag, pos.id, pos.symbol, pos.side, pos.qty,
            pos.entry_price, pos.stop_loss, pos.take_profit,
        )

        self._notify(
            f"🟢 Pozisyon Açıldı [{strategy_tag}]\n"
            f"Sembol: {symbol} | Yön: {signal.side.upper()}\n"
            f"Giriş: {entry_price:.4f} | SL: {sizing.stop_loss:.4f} | TP: {sizing.take_profit:.4f}\n"
            f"Skor: {signal.score:.1f}/100"
        )

    def _update_trailing_stop(self, pos: Position, price: float) -> None:
        d = self.trail_distance.get(pos.id, abs(pos.entry_price - pos.stop_loss))
        if d <= 0:
            return
        if pos.side == "long":
            new_sl = price - d
            if new_sl > pos.stop_loss:          # entry koşulu kaldırıldı (DÜZELTME)
                old_sl = pos.stop_loss
                pos.stop_loss = new_sl
                self.storage.update_position_sl(pos.id, new_sl)
                logger.debug("Trailing SL updated (LONG): %.4f → %.4f", old_sl, new_sl)
        else:
            new_sl = price + d
            if new_sl < pos.stop_loss:          # entry koşulu kaldırıldı (DÜZELTME)
                old_sl = pos.stop_loss
                pos.stop_loss = new_sl
                self.storage.update_position_sl(pos.id, new_sl)
                logger.debug("Trailing SL updated (SHORT): %.4f → %.4f", old_sl, new_sl)

    def _update_open_positions_for_symbol(self, symbol: str, price: float) -> None:
        remaining: list[Position] = []
        for pos in self.open_positions:
            if pos.symbol != symbol:
                remaining.append(pos)
                continue

            self._update_trailing_stop(pos, price)

            close_reason = ""
            if pos.side == "long":
                if price <= pos.stop_loss:
                    close_reason = "SL"
                elif price >= pos.take_profit:
                    close_reason = "TP"
            else:
                if price >= pos.stop_loss:
                    close_reason = "SL"
                elif price <= pos.take_profit:
                    close_reason = "TP"

            # Max pozisyon süresi kontrolü
            if not close_reason:
                max_hours = self.settings.risk.max_position_duration_hours
                if max_hours > 0:
                    age_hours = (
                        datetime.now(timezone.utc) - pos.opened_at
                    ).total_seconds() / 3600.0
                    if age_hours >= max_hours:
                        close_reason = "TIMEOUT"
                        logger.info(
                            "Position %s timed out after %.1f hours", pos.id, age_hours
                        )

            if not close_reason:
                remaining.append(pos)
                continue

            exit_price = price
            if not self.settings.app.paper_mode:
                self.exchange.cancel_all_orders(pos.symbol)
                exit_price = self.exchange.close_position(pos.symbol, pos.side, pos.qty)

            pnl = self._realized_pnl(pos, exit_price)
            entry_notional = pos.entry_price * pos.qty * self.settings.risk.leverage
            exit_notional = exit_price * pos.qty * self.settings.risk.leverage
            fee = self.risk.fee_cost(entry_notional) + self.risk.fee_cost(exit_notional)
            pnl_after_fee = pnl - fee
            self.cash += pnl_after_fee

            rec = TradeRecord(
                position_id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=pos.qty,
                pnl_usd=pnl_after_fee,
                pnl_pct=(pnl_after_fee / max(1e-9, pos.entry_price * pos.qty)) * 100,
                opened_at=pos.opened_at,
                closed_at=datetime.now(timezone.utc),
                reason=close_reason,
            )
            self.storage.add_trade(rec)
            self.storage.delete_position(pos.id)
            self.trail_distance.pop(pos.id, None)

            logger.info(
                "Position closed: id=%s symbol=%s reason=%s pnl=%.4f",
                pos.id, pos.symbol, close_reason, pnl_after_fee,
            )

            pnl_sign = "🟢" if pnl_after_fee >= 0 else "🔴"
            self._notify(
                f"{pnl_sign} Pozisyon Kapandı\n"
                f"Sembol: {pos.symbol} | Yön: {pos.side.upper()} | Sebep: {close_reason}\n"
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

    # ─── Notifications ────────────────────────────────────────────────────────

    def _notify(self, message: str) -> None:
        if self.telegram:
            try:
                self.telegram.send(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Telegram notify failed: %s", exc)

    # ─── Snapshot ─────────────────────────────────────────────────────────────

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
