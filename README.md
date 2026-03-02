# Austrade

Windows tabanli Python trade bot uygulamasi (Docker yok).

## Ozellikler
- Binance ve MEXC baglantisi (CCXT)
- LuxAlgo structure sinyal motoru (BOS / CHoCH)
- Risk yonetimi (SL/TP, RR, fee)
- Kagit trading (paper mode) destekli
- UI kartlar: gunluk PNL, equity, aktif islem sayisi, win-rate
- Islem takibi ve islem gecmisi tablolari
- Binance temasi: siyah + altin sari

## Kurulum
1. `config.example.json` dosyasini `config.json` olarak kopyala (bat bunu otomatik yapar).
2. API anahtarlarini `config.json` icine ekle.
3. Bagimliliklari kur:
   `python -m pip install -r requirements.txt`
4. Uygulamayi baslat:
   `Austrade.bat`

## Not
- Bu yazilim kar garantisi vermez.
- Varsayilan olarak `paper_mode=true` gelir.
- Gercek islem icin stratejiyi backtest/forward test etmeden canliya gecme.
