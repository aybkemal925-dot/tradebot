from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os

from .logging_utils import get_logger

logger = get_logger(__name__)


def _load_dotenv(path: str = ".env") -> None:
    """Read .env file and inject into os.environ."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class AppConfig:
    name: str
    paper_mode: bool
    starting_balance_usd: float
    refresh_seconds: int
    debug: bool = False


@dataclass(slots=True)
class ExchangeConfig:
    name: str
    symbol: str
    symbols: list[str]
    symbol_count: int
    scan_quote: str
    majors_only: bool
    market_type: str
    leverage: int
    timeframe: str
    limit: int
    api_key: str
    api_secret: str


@dataclass(slots=True)
class RiskConfig:
    risk_per_trade_pct: float
    max_open_positions: int
    max_daily_loss_pct: float
    fee_pct: float
    target_rr: float
    atr_period: int
    atr_stop_mult: float
    leverage: int
    max_drawdown_pct: float = 20.0
    max_consecutive_losses: int = 5
    max_position_duration_hours: float = 48.0
    trailing_callback_atr_mult: float = 1.5
    trailing_activation_atr_mult: float = 2.0
    trailing_callback_min_pct: float = 0.20
    trailing_callback_max_pct: float = 3.00
    trailing_activation_min_pct: float = 0.50
    trailing_activation_max_pct: float = 5.00


@dataclass(slots=True)
class StrategyConfig:
    pivot_lookback: int
    swing_lookback: int
    confirm_bars: int
    use_choch_only: bool
    use_ob_fvg_filter: bool = False
    ema_trend_filter: bool = True
    htf_trend_filter: bool = True
    htf_timeframe: str = "30m"
    rsi_period: int = 14
    rsi_threshold: float = 50.0
    allow_long: bool = True
    allow_short: bool = True
    volume_filter: bool = True
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.2
    rsi_long_max: float = 35.0
    rsi_short_min: float = 65.0
    adx_period: int = 14
    adx_threshold: float = 18.0
    cvd_lookback: int = 30
    score_threshold: float = 75.0
    technical_weight: float = 0.40
    onchain_weight: float = 0.25
    sentiment_weight: float = 0.10
    regime_weight: float = 0.25


@dataclass(slots=True)
class PortfolioConfig:
    max_notional_pct: float = 300.0
    max_margin_usage_pct: float = 80.0
    max_concurrent_per_cluster: int = 1
    volatility_adjustment: bool = True
    target_cash_pct: float = 20.0


@dataclass(slots=True)
class StorageConfig:
    db_path: str


@dataclass(slots=True)
class TelegramConfig:
    enabled: bool
    token: str
    chat_id: str


@dataclass(slots=True)
class Settings:
    app: AppConfig
    exchange: ExchangeConfig
    risk: RiskConfig
    strategy: StrategyConfig
    storage: StorageConfig
    telegram: TelegramConfig
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


def load_settings(path: str | Path = "config.json") -> Settings:
    _load_dotenv()

    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path = Path("config.example.json")
    data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    ex = data["exchange"]
    risk = data["risk"]
    strat = data["strategy"]
    app_data = data["app"]
    tg = data.get("telegram", {})
    port = data.get("portfolio", {})

    env_api_key = os.getenv("AUSTRADE_API_KEY", "")
    env_api_secret = os.getenv("AUSTRADE_API_SECRET", "")

    api_key = env_api_key or ex.get("api_key", "")
    api_secret = env_api_secret or ex.get("api_secret", "")

    symbols = ex.get("symbols") or []
    primary_symbol = ex.get("symbol") or (symbols[0] if symbols else "BTC/USDT")
    leverage = int(ex.get("leverage", risk.get("leverage", 1)))

    if leverage > 50:
        logger.warning("Leverage is too high: %sx", leverage)
    effective_risk = float(risk["risk_per_trade_pct"]) * leverage
    if effective_risk > 50.0:
        logger.error(
            "Risk/leverage combination is dangerous: %.1f%% (risk_pct=%.1f%% * lev=%sx)",
            effective_risk,
            float(risk["risk_per_trade_pct"]),
            leverage,
        )

    settings = Settings(
        app=AppConfig(
            name=app_data["name"],
            paper_mode=bool(app_data["paper_mode"]),
            starting_balance_usd=float(app_data["starting_balance_usd"]),
            refresh_seconds=int(app_data["refresh_seconds"]),
            debug=bool(app_data.get("debug", False)),
        ),
        exchange=ExchangeConfig(
            name=ex["name"],
            symbol=primary_symbol,
            symbols=symbols,
            symbol_count=int(ex.get("symbol_count", 20)),
            scan_quote=str(ex.get("scan_quote", "USDT")).upper(),
            majors_only=bool(ex.get("majors_only", True)),
            market_type=ex.get("market_type", "spot"),
            leverage=leverage,
            timeframe=ex["timeframe"],
            limit=ex["limit"],
            api_key=api_key,
            api_secret=api_secret,
        ),
        risk=RiskConfig(
            risk_per_trade_pct=float(risk["risk_per_trade_pct"]),
            max_open_positions=int(risk["max_open_positions"]),
            max_daily_loss_pct=float(risk.get("max_daily_loss_pct", 10.0)),
            fee_pct=float(risk["fee_pct"]),
            target_rr=float(risk["target_rr"]),
            atr_period=int(risk["atr_period"]),
            atr_stop_mult=float(risk["atr_stop_mult"]),
            leverage=int(risk.get("leverage", leverage)),
            max_drawdown_pct=float(risk.get("max_drawdown_pct", 20.0)),
            max_consecutive_losses=int(risk.get("max_consecutive_losses", 5)),
            max_position_duration_hours=float(risk.get("max_position_duration_hours", 48.0)),
            trailing_callback_atr_mult=float(risk.get("trailing_callback_atr_mult", 1.5)),
            trailing_activation_atr_mult=float(risk.get("trailing_activation_atr_mult", 2.0)),
            trailing_callback_min_pct=float(risk.get("trailing_callback_min_pct", 0.20)),
            trailing_callback_max_pct=float(risk.get("trailing_callback_max_pct", 3.00)),
            trailing_activation_min_pct=float(risk.get("trailing_activation_min_pct", 0.50)),
            trailing_activation_max_pct=float(risk.get("trailing_activation_max_pct", 5.00)),
        ),
        strategy=StrategyConfig(
            pivot_lookback=int(strat["pivot_lookback"]),
            swing_lookback=int(strat.get("swing_lookback", 50)),
            confirm_bars=int(strat["confirm_bars"]),
            use_choch_only=bool(strat["use_choch_only"]),
            use_ob_fvg_filter=bool(strat.get("use_ob_fvg_filter", False)),
            ema_trend_filter=bool(strat.get("ema_trend_filter", True)),
            htf_trend_filter=bool(strat.get("htf_trend_filter", True)),
            htf_timeframe=str(strat.get("htf_timeframe", "30m")) or "30m",
            rsi_period=int(strat.get("rsi_period", 14)),
            rsi_threshold=float(strat.get("rsi_threshold", 50.0)),
            allow_long=bool(strat.get("allow_long", True)),
            allow_short=bool(strat.get("allow_short", True)),
            volume_filter=bool(strat.get("volume_filter", True)),
            volume_ma_period=int(strat.get("volume_ma_period", 20)),
            min_volume_ratio=float(strat.get("min_volume_ratio", 1.2)),
            rsi_long_max=float(strat.get("rsi_long_max", 35.0)),
            rsi_short_min=float(strat.get("rsi_short_min", 65.0)),
            adx_period=int(strat.get("adx_period", 14)),
            adx_threshold=float(strat.get("adx_threshold", 18.0)),
            cvd_lookback=int(strat.get("cvd_lookback", 30)),
            score_threshold=float(strat.get("score_threshold", 75.0)),
            technical_weight=float(strat.get("technical_weight", 0.40)),
            onchain_weight=float(strat.get("onchain_weight", 0.25)),
            sentiment_weight=float(strat.get("sentiment_weight", 0.10)),
            regime_weight=float(strat.get("regime_weight", 0.25)),
        ),
        storage=StorageConfig(**data["storage"]),
        telegram=TelegramConfig(
            enabled=bool(tg.get("enabled", False)),
            token=os.getenv("TELEGRAM_TOKEN", tg.get("token", "")),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", tg.get("chat_id", "")),
        ),
        portfolio=PortfolioConfig(
            max_notional_pct=float(port.get("max_notional_pct", 300.0)),
            max_margin_usage_pct=float(port.get("max_margin_usage_pct", 80.0)),
            max_concurrent_per_cluster=int(port.get("max_concurrent_per_cluster", 1)),
            volatility_adjustment=bool(port.get("volatility_adjustment", True)),
            target_cash_pct=float(port.get("target_cash_pct", 20.0)),
        ),
    )

    logger.info(
        "Settings loaded: paper_mode=%s debug=%s exchange=%s market_type=%s timeframe=%s "
        "symbol_count=%s env_key=%s telegram=%s ob_fvg_filter=%s "
        "ema_trend=%s htf_trend=%s htf_tf=%s rsi=%s>%.1f volume_filter=%s max_drawdown=%.1f%%",
        settings.app.paper_mode,
        settings.app.debug,
        settings.exchange.name,
        settings.exchange.market_type,
        settings.exchange.timeframe,
        settings.exchange.symbol_count,
        bool(env_api_key),
        settings.telegram.enabled,
        settings.strategy.use_ob_fvg_filter,
        settings.strategy.ema_trend_filter,
        settings.strategy.htf_trend_filter,
        settings.strategy.htf_timeframe,
        settings.strategy.rsi_period,
        settings.strategy.rsi_threshold,
        settings.strategy.volume_filter,
        settings.risk.max_drawdown_pct,
    )
    return settings
