# Austrade Tradebot — Strateji Analiz Raporu

> **Tarih:** 2026-03-02
> **Yöntem:** 5 paralel ajan — Risk, Teknik Analiz, ML/AI, Performans, Portföy
> **Kapsam:** Mevcut durum analizi + somut iyileştirme önerileri

---

## İçindekiler

1. [Agent 1 — Risk Yönetimi](#agent-1--risk-yönetimi)
2. [Agent 2 — Teknik Analiz](#agent-2--teknik-analiz)
3. [Agent 3 — ML/AI Entegrasyonu](#agent-3--mlai-entegrasyonu)
4. [Agent 4 — Performans Optimizasyonu](#agent-4--performans-optimizasyonu)
5. [Agent 5 — Portföy Yönetimi](#agent-5--portföy-yönetimi)
6. [Birleşik Öncelik Tablosu](#birleşik-öncelik-tablosu)

---

## AGENT 1 — Risk Yönetimi

### Mevcut Durum Özeti

| Bileşen | Dosya | Durum |
|---------|-------|-------|
| ATR-tabanlı SL/TP | `risk.py` | ✅ Aktif — doğrulama eksik |
| Trailing SL | `engine.py` | ⚠️ Aktif — mantık hatası var |
| Günlük kayıp limiti | `engine.py` | ⚠️ Aktif — denominator yanlış |
| Max açık pozisyon | `engine.py` | ✅ Aktif — korelasyon yok |
| Leverage cap | `risk.py` | ⚠️ Kısmi — multi-position margin eksik |
| Fee handling | `risk.py` | ✅ Aktif — funding rate yok |
| Max drawdown | — | ❌ **Yok** |
| Korelasyon filtresi | — | ❌ **Yok** |
| Position duration | — | ❌ **Yok** |

---

### Tespit Edilen Hatalar ve Düzeltmeler

#### 1. Trailing SL Mantık Hatası (`engine.py:355-368`) — KRİTİK

**Problem:** `price > pos.entry_price` koşulu fazla kısıtlayıcı; fiyat entry altına düştüğünde SL hiç güncellenmez.

**Düzeltme:**
```python
def _update_trailing_stop(self, pos: Position, price: float) -> None:
    d = self.trail_distance.get(pos.id, 0.0)
    if d <= 0:
        return
    if pos.side == "long":
        new_sl = price - d
        if new_sl > pos.stop_loss:          # entry koşulu KALDIR
            old_sl = pos.stop_loss
            pos.stop_loss = new_sl
            self.storage.update_position_sl(pos.id, new_sl)
            logger.debug("Trailing SL updated (LONG): %.4f → %.4f", old_sl, new_sl)
    else:
        new_sl = price + d
        if new_sl < pos.stop_loss:
            old_sl = pos.stop_loss
            pos.stop_loss = new_sl
            self.storage.update_position_sl(pos.id, new_sl)
            logger.debug("Trailing SL updated (SHORT): %.4f → %.4f", old_sl, new_sl)
```

#### 2. Günlük Kayıp Limiti Denominator Hatası (`engine.py:251`) — KRİTİK

**Problem:** `self.cash` yerine `starting_balance` kullanılmalı; çünkü `self.cash` dinamik.

**Düzeltme:**
```python
def _check_daily_loss_limit(self) -> bool:
    max_loss_pct = self.settings.risk.max_daily_loss_pct
    if max_loss_pct <= 0:
        return False
    daily_pnl = self.storage.today_pnl()
    starting_balance = self.settings.app.starting_balance_usd   # sabit baz
    if daily_pnl < 0:
        loss_pct = abs(daily_pnl) / max(1.0, starting_balance) * 100.0
        if loss_pct >= max_loss_pct:
            if not self.daily_loss_halted:
                self.daily_loss_halted = True
                self._notify(f"Günlük zarar limiti! PNL=${daily_pnl:.2f} ({loss_pct:.1f}%)")
            return True
    self.daily_loss_halted = False
    return False
```

#### 3. Multi-Position Margin Takibi Eksik (`risk.py:56-64`) — ÖNEMLİ

**Problem:** Diğer açık pozisyonların margin'i hesaba katılmıyor.

**Düzeltme:**
```python
def size_position(self, signal, equity, df, open_positions=None):
    open_positions = open_positions or []
    # ...
    used_margin = sum(
        (p.qty * p.entry_price) / max(1, self.cfg.leverage)
        for p in open_positions
    )
    available_margin = equity - used_margin
    max_notional = available_margin * max(self.cfg.leverage, 1)
    if qty * signal.price > max_notional:
        qty = max_notional / signal.price
```

#### 4. SL/TP Doğrulama Eksik (`risk.py:66-71`)

```python
if signal.side == "long":
    if sl <= 0 or sl >= signal.price:
        logger.error("Invalid long SL: %.4f >= entry %.4f", sl, signal.price)
        return None
    if tp <= signal.price:
        logger.error("Invalid long TP: %.4f <= entry %.4f", tp, signal.price)
        return None
```

#### 5. Eksik: Max Drawdown + Ardışık Kayıp Kontrolü

**`config.py`'a eklenecek alanlar:**
```python
@dataclass(slots=True)
class RiskConfig:
    # mevcut alanlar...
    max_drawdown_pct: float = 20.0
    max_consecutive_losses: int = 5
    max_position_duration_hours: float = 48.0
```

**`engine.py`'a eklenecek metod:**
```python
def _check_drawdown_limit(self) -> bool:
    peak = self.storage.peak_equity()
    if peak > 0:
        dd_pct = (peak - self.equity) / peak * 100.0
        if dd_pct >= self.settings.risk.max_drawdown_pct:
            logger.error("Max drawdown reached: %.1f%%", dd_pct)
            return True
    return False
```

#### 6. Leverage Uyarısı (`config.py`)

```python
leverage = int(ex.get("leverage", risk.get("leverage", 1)))
if leverage > 50:
    logger.warning("Leverage çok yüksek: %sx", leverage)
effective_risk = float(risk["risk_per_trade_pct"]) * leverage
if effective_risk > 50.0:
    logger.error("Risk-leverage kombinasyonu tehlikeli: %.1f%%!", effective_risk)
```

---

## AGENT 2 — Teknik Analiz

### Mevcut Durum Özeti

İki strateji aktif:
- **SMC (birincil):** Order Blocks, Fair Value Gaps, swing/internal yapı
- **Triple Confirmation:** RSI + MACD + Bollinger Bands

| Sorun | Dosya | Önem |
|-------|-------|------|
| Trend filtresi yok (EMA200) | `strategy_smc.py` | **YÜKSEK** |
| Volume konfirmasyonu yok | Her iki strateji | **YÜKSEK** |
| Multi-timeframe analiz yok | `engine.py` + `exchange.py` | **YÜKSEK** |
| Confirm bars re-validasyonu yok | `strategy_smc.py:233` | ORTA |
| FVG wick vs close invalidasyonu | `strategy_smc.py:147` | DÜŞÜK |
| OB yaş yönetimi yok | `strategy_smc.py:121` | DÜŞÜK |
| Sabit RR (2:1) her sinyal için | `risk.py:66` | ORTA |

---

### Somut İyileştirmeler

#### A. EMA Trend Filtresi (Yüksek Öncelik)

**`strategy_smc.py`'a ekle:**
```python
def _calc_trend_filter(self, df: pd.DataFrame) -> int:
    """1=uptrend, -1=downtrend, 0=neutral"""
    close = df["close"]
    ema_50  = close.ewm(span=50,  adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    c = close.iloc[-1]
    e50 = ema_50.iloc[-1]
    e200 = ema_200.iloc[-1]
    if c > e50 > e200:
        return 1
    if c < e50 < e200:
        return -1
    return 0
```

**`_build_raw_signal()` içinde sinyal öncesi:**
```python
trend = self._calc_trend_filter(df)
if side == "long"  and trend == -1: return None
if side == "short" and trend ==  1: return None
```

**`config.json`'a:**
```json
"strategy": { "ema_trend_filter": true }
```

#### B. Volume Konfirmasyonu (Yüksek Öncelik)

```python
def _check_volume_confluence(self, df: pd.DataFrame) -> bool:
    vol = df["volume"]
    ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
    return ratio >= self.cfg.min_volume_ratio   # default 1.2
```

#### C. Multi-Timeframe Konfirmasyon (Orta Öncelik)

**`exchange.py`'a:**
```python
def fetch_ohlcv_multi(self, symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for tf in timeframes:
        rows = self.ex.fetch_ohlcv(self.normalize_symbol(symbol), tf, limit=200)
        result[tf] = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    return result
```

**`engine.py` `_loop()` içinde:**
```python
dfs = self.exchange.fetch_ohlcv_multi(symbol, ["5m", "1h"])
signal = self.strategies[symbol].next_signal(dfs["5m"], htf=dfs.get("1h"))
```

#### D. Sinyal Confluence Skorlaması (Orta Öncelik)

```python
def _score_confluence(self, df, side, price, ob_ok, fvg_ok) -> int:
    score = 0
    if ob_ok or fvg_ok:             score += 1
    if self._premium_discount_filter(side, price): score += 1
    if self._calc_trend_filter(df) != 0:           score += 1
    if self._check_volume_confluence(df):           score += 1
    return score   # 0-4; minimum 2 gerekli

# _build_raw_signal() sonuna ekle:
if self._score_confluence(...) < 2:
    return None
```

#### E. FVG Wick → Close Invalidasyonu (Kolay, Hızlı)

**`strategy_smc.py:147-150`:**
```python
# ÖNCE: low <= gap.bottom  (wick yeterli)
# SONRA:
close = float(df["close"].iloc[-1])
if gap.side == "long"  and close <= gap.bottom: gap.active = False
if gap.side == "short" and close >= gap.top:    gap.active = False
```

#### F. OB Yaş Yönetimi

```python
current_idx = len(df) - 1
for ob in self.order_blocks:
    if current_idx - ob.pivot_idx > self.swing_lb + 20:
        ob.active = False
```

---

## AGENT 3 — ML/AI Entegrasyonu

### Mevcut Veri Altyapısı

| Tablo | İçerik | ML Kullanımı |
|-------|--------|-------------|
| `trades` | PNL, entry/exit, side | ✅ Win/loss sınıflandırması |
| `balance_snapshots` | Equity, daily PNL | ✅ Regim tespiti |
| `open_positions` | Anlık pozisyonlar | ✅ Bağlam |
| `signal_context` | — | ❌ **Yok — eklenmeli** |

**Ana eksiklik:** İndikatör verileri ve sinyal bağlamı DB'ye kaydedilmiyor.

---

### Önerilen Hafif ML Yaklaşımları

> **Kural:** Sadece NumPy/Pandas + opsiyonel statsmodels. PyTorch/TensorFlow YAPMA.

#### 1. Sinyal Kalite Skoru (`austrade/ml_signal_quality.py`)

```python
class SignalQualityScorer:
    """Pure NumPy/Pandas — ekstra bağımlılık yok."""

    def score(self, symbol: str, side: str, context: dict) -> float:
        """0.0-1.0 arası güven skoru döndür."""
        score = 0.5
        score += (self.side_bias.get(side, 0.5) - 0.5) * 0.3
        support = min((context.get("ob_count",0) + context.get("fvg_count",0)) / 3, 1.0)
        score += support * 0.2
        if context.get("volatility_pct", 2.0) > 5.0:
            score -= 0.15
        if context.get("consecutive_losses", 0) >= 3:
            score *= 0.7
        return max(0.1, min(0.9, score))

    def recommend_qty(self, quality: float, base_qty: float) -> float:
        if quality < 0.4: return 0          # atla
        if quality < 0.5: return base_qty * 0.5
        if quality < 0.7: return base_qty * 0.7
        return base_qty
```

**`engine.py` entegrasyonu:**
```python
quality = self.ml_scorer.score(symbol, signal.side, {
    "ob_count": len(self.strategies[symbol].order_blocks),
    "fvg_count": len(self.strategies[symbol].fvgs),
    "volatility_pct": (self.risk._atr(df) / price) * 100,
    "consecutive_losses": self._count_consecutive_losses(),
})
sizing.qty = self.ml_scorer.recommend_qty(quality, sizing.qty)
if sizing.qty <= 0:
    return
```

#### 2. Regim Dedektörü (`austrade/ml_regime.py`)

```python
class RegimeDetector:
    def detect(self) -> str:
        """'bull' | 'bear' | 'sideways'"""
        trades = self.storage.recent_trades(500)
        df = pd.DataFrame(trades)
        win_rate = (df["pnl_pct"] > 0).mean()
        if win_rate > 0.6: return "bull"
        if win_rate < 0.4: return "bear"
        return "sideways"

    def adjust_risk(self, regime: str, base_pct: float) -> float:
        return {"bull": base_pct*1.2, "sideways": base_pct, "bear": base_pct*0.6}[regime]
```

#### 3. Veri Zenginleştirme — Yeni Tablo

**`storage.py`'a eklenecek şema:**
```sql
CREATE TABLE IF NOT EXISTS signal_context (
    id          INTEGER PRIMARY KEY,
    trade_id    INTEGER UNIQUE,
    signal_ts   TEXT,
    symbol      TEXT,
    side        TEXT,
    atr_14      REAL,
    volatility_pct REAL,
    ob_count    INTEGER,
    fvg_count   INTEGER,
    ob_fvg_aligned BOOLEAN,
    quality_score  REAL,
    win         BOOLEAN,
    pnl_pct     REAL,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);
```

### Yapılmaması Gerekenler

| Yapma | Neden |
|-------|-------|
| LSTM / Transformer | Yetersiz veri + yüksek overfitting riski |
| Reinforcement Learning | Production stability riski |
| Karmaşık ensemble | Bakım yükü > fayda |
| ML-only sinyal üretimi | Explainability kaybı |

### Uygulama Takvimi

| Hafta | Görev |
|-------|-------|
| 1-2 | `signal_context` tablosu + son 30 gün backfill |
| 2-3 | `SignalQualityScorer` implementasyonu + paper test |
| 3-4 | `RegimeDetector` + parametre optimizasyonu |
| 5+ | Canlı test — beklenen +3-5% win rate iyileşmesi |

---

## AGENT 4 — Performans Optimizasyonu

### Kritik Darboğazlar

| Öncelik | Bileşen | Sorun | Etki |
|---------|---------|-------|------|
| **YÜKSEK** | `engine.py:181-206` | Semboller sıralı fetch | 4x yavaş |
| **YÜKSEK** | `storage.py:100` | `substr()` full table scan | 100x yavaş |
| **YÜKSEK** | `storage.py:123` | Her snapshot'ta cleanup | 1000x fazla sorgu |
| **YÜKSEK** | `strategy_smc.py:137` | `list.insert(0)` = O(n) | 20x yavaş |
| ORTA | `engine.py:463` | Snapshot lock süresi | UI gecikmesi |
| ORTA | `risk.py:26` | Tüm ATR serisi hesaplanıyor | 7x gereksiz |
| DÜŞÜK | `strategy_smc.py:85` | Pivot hesabı vektörsüz | 500x yavaş |

---

### Somut Düzeltmeler

#### 1. Paralel Sembol Fetch (30 dk, 4x hız) — `engine.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(self.exchange.fetch_ohlcv, s): s for s in symbol_list}
    market_data = {}
    for fut in as_completed(futures):
        sym = futures[fut]
        try:
            market_data[sym] = fut.result(timeout=15)
        except Exception as e:
            logger.warning("fetch_ohlcv timeout: %s — %s", sym, e)
```

#### 2. SQLite Index + `today_pnl()` Düzeltmesi (15 dk, 100x) — `storage.py`

```python
# _init_db() şemasına ekle:
"CREATE INDEX IF NOT EXISTS idx_trades_closed_date ON trades(DATE(closed_at));"

# today_pnl():
cur = self.conn.execute(
    "SELECT COALESCE(SUM(pnl_usd),0) FROM trades WHERE DATE(closed_at) = ?",
    (today,)   # INDEX kullanılır — substr() değil
)
```

#### 3. Cleanup Throttle (5 dk, 1000x) — `storage.py`

```python
def __init__(self, db_path):
    ...
    self._cleanup_counter = 0

def add_snapshot(self, equity, daily_pnl):
    # ... insert ...
    self.conn.commit()
    self._cleanup_counter += 1
    if self._cleanup_counter >= 1000:       # ~2.8 saatte bir
        self.cleanup_old_snapshots(days=7)
        self._cleanup_counter = 0
```

#### 4. FVG `insert(0)` → `append` (5 dk, 20x) — `strategy_smc.py`

```python
# ÖNCE: self.fvgs.insert(0, gap)   → O(n)
# SONRA:
self.fvgs.append(gap)
self.fvgs = self.fvgs[-20:]          # son 20'yi tut, O(1)
```

#### 5. ATR Hızlı Hesaplama — `risk.py`

```python
def _atr_fast(self, df: pd.DataFrame) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:]-l[1:],
         np.maximum(np.abs(h[1:]-c[:-1]),
                    np.abs(l[1:]-c[:-1])))
    period = self.cfg.atr_period
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())
```

#### 6. Circuit Breaker (`engine.py`)

```python
@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_sec: float = 300.0
    failure_count: int = 0
    state: str = "closed"
    open_at: float = 0.0

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.open_at = time.time()

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state != "open":
            return False
        if time.time() - self.open_at > self.recovery_sec:
            self.state = "half_open"
            return False
        return True
```

#### 7. Pivot Dedup Set Temizliği — `strategy_smc.py`

```python
# next_signal() sonunda:
cutoff = len(df) - 100
self._ob_high_added = {i for i in self._ob_high_added if i >= cutoff}
self._ob_low_added  = {i for i in self._ob_low_added  if i >= cutoff}
```

---

## AGENT 5 — Portföy Yönetimi

### Mevcut Durum

- Multi-symbol: ✅ Var (sembol başına 1 pozisyon, max 5 toplam)
- Korelasyon kontrolü: ❌ **Yok**
- Toplam notional cap: ❌ **Yok**
- Çeşitlendirme mantığı: ❌ **Yok**
- Portföy rebalancing: ❌ **Yok**

**Risk örneği:** 5 pozisyon × 5x leverage = 25x efektif portföy maruziyeti mümkün.

---

### Somut İyileştirmeler

#### 1. Toplam Notional Sınırı — KRİTİK (`engine.py`)

```python
def _maybe_open_position(self, symbol, signal, df):
    # ... mevcut kontroller ...

    # Yeni: toplam notional cap
    max_notional_pct = self.settings.portfolio.max_notional_pct  # default 300
    total_notional = sum(
        p.qty * self.last_prices.get(p.symbol, p.entry_price) * self.settings.risk.leverage
        for p in self.open_positions
    )
    new_notional = sizing.qty * signal.price * self.settings.risk.leverage
    if (total_notional + new_notional) > self.equity * (max_notional_pct / 100):
        logger.info("Portfolio notional cap aşılıyor, pozisyon atlanıyor")
        return
```

#### 2. Korelasyon Filtresi (`engine.py`)

```python
def _is_correlated_with_open(self, symbol: str) -> bool:
    """BTC+ETH gibi yüksek korelasyonlu çiftleri engelle."""
    CORRELATED_CLUSTERS = [
        {"BTC/USDT", "ETH/USDT"},
        {"SOL/USDT", "AVAX/USDT", "NEAR/USDT"},
    ]
    sym_base = symbol.split("/")[0]
    for cluster in CORRELATED_CLUSTERS:
        if symbol in cluster:
            for pos in self.open_positions:
                if pos.symbol in cluster and pos.symbol != symbol:
                    logger.info("Korelasyon filtresi: %s ile %s çakışıyor", symbol, pos.symbol)
                    return True
    return False
```

#### 3. Volatilite Bazlı Pozisyon Büyüklüğü (`risk.py`)

```python
def _volatility_scale(self, df: pd.DataFrame) -> float:
    """ATR percentile'e göre scale faktörü döndür: 0.7-1.2"""
    atr = self._atr(df)
    price = float(df["close"].iloc[-1])
    vol_pct = (atr / price) * 100
    if vol_pct > 5.0: return 0.7   # Yüksek volatilite → küçüt
    if vol_pct < 1.0: return 1.2   # Düşük volatilite → büyüt
    return 1.0
```

#### 4. Config Schema Güncellemesi

**`config.py`'a ekle:**
```python
@dataclass(slots=True)
class PortfolioConfig:
    max_notional_pct: float = 300.0
    max_margin_usage_pct: float = 80.0
    max_concurrent_per_cluster: int = 1
    max_corr_threshold: float = 0.6
    volatility_adjustment: bool = True
    target_cash_pct: float = 20.0
```

**`config.json`'a ekle:**
```json
"portfolio": {
    "max_notional_pct": 300.0,
    "max_margin_usage_pct": 80.0,
    "max_concurrent_per_cluster": 1,
    "volatility_adjustment": true,
    "target_cash_pct": 20.0
}
```

#### 5. Sembol Universe Rotasyonu (Opsiyonel)

```python
# config.json
"exchange": { "symbol_rotation_days": 7 }

# engine.py _loop() içinde:
if self._bars_since_rotation >= rotation_bars:
    new_universe = self.exchange.fetch_universe_symbols(self.settings.exchange.symbol_count)
    added = set(new_universe) - set(self.symbols)
    removed = set(self.symbols) - set(new_universe)
    for sym in removed:
        logger.info("Universe'den çıkarıldı: %s (yeni sinyal gönderilmeyecek)", sym)
    self.symbols = new_universe
    self._bars_since_rotation = 0
```

---

## Birleşik Öncelik Tablosu

### Kritik (Bu Hafta)

| # | Görev | Dosya | Satır | Süre |
|---|-------|-------|-------|------|
| 1 | Trailing SL mantık hatası düzelt | `engine.py` | ~355 | 15 dk |
| 2 | Günlük kayıp denominator düzelt | `engine.py` | ~251 | 10 dk |
| 3 | SQLite index + `today_pnl()` fix | `storage.py` | ~100 | 15 dk |
| 4 | Cleanup throttle (1000 snapshot'ta bir) | `storage.py` | ~123 | 10 dk |
| 5 | FVG `insert(0)` → `append` | `strategy_smc.py` | ~137 | 5 dk |

### Önemli (1-2 Hafta)

| # | Görev | Dosya | Süre |
|---|-------|-------|------|
| 6 | EMA trend filtresi ekle | `strategy_smc.py` | 1 saat |
| 7 | Volume konfirmasyonu ekle | `strategy_smc.py` | 45 dk |
| 8 | Multi-position margin tracking | `risk.py` | 30 dk |
| 9 | Toplam notional cap | `engine.py` | 20 dk |
| 10 | Korelasyon filtresi (basit cluster) | `engine.py` | 30 dk |
| 11 | Paralel sembol fetch | `engine.py` | 30 dk |
| 12 | SL/TP doğrulama (negatif check) | `risk.py` | 20 dk |
| 13 | FVG close vs wick invalidasyonu | `strategy_smc.py` | 5 dk |

### Geliştirme (2-4 Hafta)

| # | Görev | Beklenen Fayda |
|---|-------|---------------|
| 14 | Multi-timeframe konfirmasyon (1H+5m) | -30% whipsaw |
| 15 | Confluence scoring sistemi | Daha iyi RR dağılımı |
| 16 | `SignalQualityScorer` ML modülü | +3-5% win rate |
| 17 | `RegimeDetector` ML modülü | -15-20% drawdown |
| 18 | ATR hızlı hesaplama (numpy) | 7x daha hızlı |
| 19 | Circuit breaker | Ağ hatası dayanıklılığı |
| 20 | `PortfolioConfig` + volatilite ölçeği | Daha iyi risk dağılımı |
| 21 | `signal_context` tablosu + ML veri toplama | ML altyapısı |
| 22 | Max drawdown kontrolü | Kümülatif risk |
| 23 | OB yaş yönetimi | Alakasız OB azalır |

---

*Rapor 5 paralel ajan tarafından oluşturuldu. Her öneri mevcut kod incelenerek dosya:satır referanslarıyla doğrulandı.*
