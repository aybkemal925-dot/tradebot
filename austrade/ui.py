from __future__ import annotations

import asyncio
import threading

import flet as ft

from .backtest import BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade, MultiBacktestResult
from .config import load_settings
from .engine import TradeEngine
from .models import Position

# ── Renk paleti ──────────────────────────────────────────────────────────────
BG    = "#0B0B0C"
CARD  = "#151517"
CARD2 = "#1B1B1F"
BORD  = "#222228"
GOLD  = "#F3BA2F"
TEXT  = "#F5F5F5"
SUB   = "#A9A9AD"
GREEN = "#0ECB81"
RED   = "#F6465D"


# ── Binance tarzı detay kartı (dialog içinde) ─────────────────────────────────

def _detail_card(pos: Position, current: float, leverage: int) -> ft.Container:
    is_long = pos.side == "long"
    upnl    = ((current - pos.entry_price) * pos.qty if is_long
               else (pos.entry_price - current) * pos.qty)
    roi_pct = upnl / max(pos.entry_price * pos.qty, 1e-9) * 100
    margin  = (pos.entry_price * pos.qty) / max(leverage, 1)
    pcolor  = GREEN if upnl >= 0 else RED
    sym     = pos.symbol.split(":")[0]

    def _row(label: str, value: str, val_color: str = TEXT) -> ft.Row:
        return ft.Row([
            ft.Text(label, size=12, color=SUB, expand=True),
            ft.Text(value, size=12, color=val_color, weight=ft.FontWeight.W_600),
        ])

    return ft.Container(
        bgcolor=CARD,
        border_radius=16,
        border=ft.border.all(2, pcolor),
        padding=20,
        width=340,
        content=ft.Column([
            ft.Row([
                ft.Text(sym, size=18, color=GOLD, weight=ft.FontWeight.BOLD),
                ft.Container(
                    padding=ft.padding.symmetric(horizontal=10, vertical=4),
                    border_radius=6,
                    bgcolor=f"{GREEN}22" if is_long else f"{RED}22",
                    border=ft.border.all(1, GREEN if is_long else RED),
                    content=ft.Text(
                        f"{'LONG' if is_long else 'SHORT'}  {leverage}x",
                        size=12, color=GREEN if is_long else RED,
                        weight=ft.FontWeight.BOLD,
                    ),
                ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

            ft.Divider(height=12, color=BORD),

            ft.Row([
                ft.Column([
                    ft.Text("PNL (USDT)", size=11, color=SUB),
                    ft.Text(f"{upnl:+.4f}", size=28, color=pcolor,
                            weight=ft.FontWeight.BOLD),
                ], spacing=2, expand=True),
                ft.Column([
                    ft.Text("ROI", size=11, color=SUB),
                    ft.Text(f"{roi_pct:+.2f}%", size=22, color=pcolor,
                            weight=ft.FontWeight.BOLD),
                ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.END),
            ]),

            ft.Container(height=8),
            _row("Size",         f"{pos.qty:.4f} {sym.split('/')[0]}"),
            _row("Margin (USDT)", f"{margin:.2f}"),
            ft.Divider(height=8, color=BORD),
            _row("Entry Price",  f"{pos.entry_price:.4f}"),
            _row("Mark Price",   f"{current:.4f}" if current else "—"),
            ft.Divider(height=8, color=BORD),
            _row("Stop Loss",    f"{pos.stop_loss:.4f}",   RED),
            _row("Take Profit",  f"{pos.take_profit:.4f}", GREEN),
        ], spacing=6, tight=True),
    )


# ── Yardımcı bileşenler ──────────────────────────────────────────────────────

def stat_card(title: str, value: str, color: str = TEXT) -> ft.Container:
    return ft.Container(
        bgcolor=CARD, border_radius=12,
        padding=ft.padding.symmetric(horizontal=14, vertical=12),
        border=ft.border.all(1, BORD), expand=True,
        content=ft.Column([
            ft.Text(title, size=11, color=SUB, weight=ft.FontWeight.W_500),
            ft.Text(value, size=20, color=color, weight=ft.FontWeight.BOLD),
        ], spacing=4, tight=True),
    )


def _dot(color: str) -> ft.Container:
    return ft.Container(width=9, height=9, border_radius=9, bgcolor=color)


def _tbl_cols(*labels: str) -> list[ft.DataColumn]:
    return [ft.DataColumn(
        ft.Text(l, color=SUB, size=11, weight=ft.FontWeight.W_600)
    ) for l in labels]


def _build_equity_chart(curve: list[dict]) -> ft.LineChart:
    """Equity curve için ft.LineChart oluştur."""
    if not curve:
        return ft.LineChart(
            data_series=[],
            expand=True,
            height=100,
        )

    equities = [float(r["equity"]) for r in curve]
    min_eq = min(equities)
    max_eq = max(equities)

    # Renk: son değer başlangıca göre yeşil mi kırmızı mı?
    line_color = GREEN if equities[-1] >= equities[0] else RED

    points = [
        ft.LineChartDataPoint(x=float(i), y=eq)
        for i, eq in enumerate(equities)
    ]

    return ft.LineChart(
        data_series=[
            ft.LineChartData(
                data_points=points,
                color=line_color,
                stroke_width=2,
                curved=True,
                stroke_cap_round=True,
                below_line_bgcolor=f"{line_color}18",
            )
        ],
        border=ft.border.all(1, BORD),
        horizontal_grid_lines=ft.ChartGridLines(
            interval=max(1.0, (max_eq - min_eq) / 4),
            color=BORD,
            width=1,
        ),
        min_y=min_eq * 0.995,
        max_y=max_eq * 1.005,
        min_x=0,
        max_x=float(max(len(equities) - 1, 1)),
        expand=True,
        height=110,
        tooltip_bgcolor=CARD2,
        left_axis=ft.ChartAxis(show_labels=False),
        bottom_axis=ft.ChartAxis(show_labels=False),
    )


# ── Ana uygulama ─────────────────────────────────────────────────────────────

def run_app() -> None:
    settings = load_settings("config.json")

    def main(page: ft.Page) -> None:
        page.title = settings.app.name
        page.window_width  = 1300
        page.window_height = 860
        page.window_min_width  = 900
        page.window_min_height = 640
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = BG
        page.padding = 0
        page.spacing = 0

        # ── Splash ──────────────────────────────────────────────────────────
        splash_msg = ft.Text("Borsa bağlantısı kuruluyor...", color=SUB,
                             size=13, text_align=ft.TextAlign.CENTER)
        splash = ft.Container(
            expand=True, bgcolor=BG, alignment=ft.alignment.center,
            content=ft.Column([
                ft.Text(settings.app.name, size=52,
                        weight=ft.FontWeight.BOLD, color=GOLD),
                ft.Text("Binance Futures · LuxAlgo Bot", color=SUB, size=14),
                ft.Container(height=32),
                ft.ProgressRing(width=48, height=48, stroke_width=3, color=GOLD),
                ft.Container(height=12),
                splash_msg,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
        )
        page.add(splash)
        page.update()

        engine_ref: list[TradeEngine] = []

        # ── Stat kartları ────────────────────────────────────────────────────
        s_equity    = stat_card("Equity",       f"${settings.app.starting_balance_usd:.2f}")
        s_pnl       = stat_card("Günlük PNL",   "$0.00")
        s_total_pnl = stat_card("Toplam PNL",   "$0.00")
        s_open      = stat_card("Açık İşlem",   "0")
        s_win       = stat_card("Win Rate",     "N/A")
        s_lev       = stat_card("Kaldıraç",     f"{settings.risk.leverage}x", GOLD)
        s_btc       = stat_card("BTC",          "-", GOLD)

        # ── Detay dialog ────────────────────────────────────────────────────
        def _close_dlg(_):
            page.dialog.open = False
            page.update()

        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=BG,
            shape=ft.RoundedRectangleBorder(radius=16),
            content=ft.Container(content=ft.Text("")),
            actions=[
                ft.TextButton("Kapat", on_click=_close_dlg,
                              style=ft.ButtonStyle(color=GOLD)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.dialog = dlg

        def _open_detail(pos: Position, current: float, lev: int):
            def _handler(_):
                dlg.content = _detail_card(pos, current, lev)
                dlg.open = True
                page.update()
            return _handler

        # ── Header ──────────────────────────────────────────────────────────
        def on_start(_):
            if engine_ref: engine_ref[0].start()

        def on_stop(_):
            if engine_ref: engine_ref[0].stop()

        def on_toggle_hist(_):
            history_section.visible = not history_section.visible
            page.update()

        def on_toggle_chart(_):
            chart_section.visible = not chart_section.visible
            page.update()

        def on_toggle_backtest(_):
            is_bt = not backtest_section.visible
            backtest_section.visible = is_bt
            trading_area.visible = not is_bt
            page.update()

        header = ft.Container(
            bgcolor=CARD,
            border=ft.border.only(bottom=ft.border.BorderSide(1, BORD)),
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            content=ft.Row([
                ft.Text(settings.app.name, size=26,
                        weight=ft.FontWeight.BOLD, color=GOLD),
                ft.Text("Binance Futures · LuxAlgo", color=SUB, size=12),
                ft.Container(expand=True),
                ft.ElevatedButton(
                    "▶  Başlat", on_click=on_start,
                    style=ft.ButtonStyle(
                        bgcolor=GOLD, color=BG,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.padding.symmetric(horizontal=18, vertical=10),
                    ),
                ),
                ft.OutlinedButton(
                    "■  Durdur", on_click=on_stop,
                    style=ft.ButtonStyle(
                        side=ft.BorderSide(1, BORD),
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.padding.symmetric(horizontal=18, vertical=10),
                    ),
                ),
                ft.TextButton("Grafik", on_click=on_toggle_chart,
                              style=ft.ButtonStyle(color=SUB)),
                ft.TextButton("Geçmiş", on_click=on_toggle_hist,
                              style=ft.ButtonStyle(color=SUB)),
                ft.TextButton("Backtest", on_click=on_toggle_backtest,
                              style=ft.ButtonStyle(color=GOLD)),
            ], spacing=10),
        )

        # ── Status bar ──────────────────────────────────────────────────────
        run_dot    = _dot(RED)
        mode_text  = ft.Text("PAPER", size=11, color=GOLD, weight=ft.FontWeight.BOLD)
        mode_badge = ft.Container(
            padding=ft.padding.symmetric(horizontal=8, vertical=3),
            border_radius=6, bgcolor="#F3BA2F22",
            border=ft.border.all(1, GOLD), content=mode_text,
        )
        halt_badge = ft.Container(
            visible=False,
            padding=ft.padding.symmetric(horizontal=8, vertical=3),
            border_radius=6, bgcolor=f"{RED}22",
            border=ft.border.all(1, RED),
            content=ft.Text("GÜNLÜK LİMİT", size=11, color=RED,
                            weight=ft.FontWeight.BOLD),
        )
        signal_lbl = ft.Text("Son sinyal: —", size=12, color=SUB)
        error_lbl  = ft.Text("", size=11, color=RED)
        tf_lbl     = ft.Text(f"TF: {settings.exchange.timeframe}", size=12, color=SUB)

        status_bar = ft.Container(
            bgcolor=CARD, border_radius=10,
            border=ft.border.all(1, BORD),
            padding=ft.padding.symmetric(horizontal=14, vertical=8),
            content=ft.Row([
                run_dot,
                ft.Text("Çalışıyor", size=12, color=SUB),
                ft.Container(width=8),
                mode_badge, halt_badge, tf_lbl,
                ft.Container(expand=True),
                signal_lbl,
                ft.Container(width=14),
                error_lbl,
            ], spacing=8),
        )

        # ── Equity curve bölümü ──────────────────────────────────────────────
        chart_container = ft.Container(
            expand=True,
            content=ft.Text("Veri yükleniyor...", color=SUB, size=11),
        )
        chart_section = ft.Container(
            visible=False,
            bgcolor=CARD, border_radius=12,
            border=ft.border.all(1, BORD),
            padding=12, margin=ft.margin.only(top=8),
            content=ft.Column([
                ft.Text("Equity Curve", size=13, color=GOLD,
                        weight=ft.FontWeight.BOLD),
                ft.Container(height=4),
                chart_container,
            ]),
        )

        # ── Açık işlemler tablosu ────────────────────────────────────────────
        open_table = ft.DataTable(
            columns=_tbl_cols(
                "#", "Sembol", "Yön", "Giriş", "Mark", "SL", "TP", "UPNL $", "UPNL %"
            ),
            rows=[],
            heading_row_color=CARD2,
            heading_row_height=36,
            data_row_min_height=44,
            data_row_max_height=48,
            column_spacing=20,
            show_checkbox_column=False,
        )

        # ── Geçmiş tablosu ───────────────────────────────────────────────────
        history_table = ft.DataTable(
            columns=_tbl_cols(
                "#", "Sembol", "Yön", "Giriş", "Çıkış", "PNL $", "Sebep", "Tarih"
            ),
            rows=[],
            heading_row_color=CARD2,
            heading_row_height=34,
            data_row_min_height=34,
            data_row_max_height=38,
            column_spacing=14,
        )
        history_total_lbl = ft.Text("Toplam PNL: $0.00", size=12,
                                    color=SUB, weight=ft.FontWeight.W_500)
        history_section = ft.Container(
            visible=False,
            bgcolor=CARD, border_radius=12,
            border=ft.border.all(1, BORD),
            padding=12, margin=ft.margin.only(top=8),
            content=ft.Column([
                ft.Row([
                    ft.Text("İşlem Geçmişi", size=13, color=GOLD,
                            weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    history_total_lbl,
                ]),
                ft.Container(height=4),
                ft.Column([history_table],
                          scroll=ft.ScrollMode.AUTO, height=260),
            ]),
        )

        # ── Backtest Section ─────────────────────────────────────────────────
        # Konfigürasyon bileşenleri
        bt_symbol_dd = ft.Dropdown(
            label="Sembol",
            value=settings.exchange.symbol,
            options=[ft.dropdown.Option(s) for s in (
                settings.exchange.symbols or [settings.exchange.symbol]
            )],
            width=200,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
        )
        bt_scope_dd = ft.Dropdown(
            label="Kapsam",
            value="single",
            options=[
                ft.dropdown.Option("single", "Tek Sembol"),
                ft.dropdown.Option("majors20", "Major 20"),
            ],
            width=140,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
        )
        bt_strategy_dd = ft.TextField(
            label="Strateji",
            value="LuxAlgo",
            width=200,
            read_only=True,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
            text_style=ft.TextStyle(color=TEXT, size=13),
        )
        bt_months_tf = ft.TextField(
            label="Periyot (ay)",
            value="24",
            width=110,
            keyboard_type=ft.KeyboardType.NUMBER,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
            text_style=ft.TextStyle(color=TEXT, size=13),
        )
        bt_capital_tf = ft.TextField(
            label="Kapital (USDT)",
            value=str(settings.app.starting_balance_usd),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
            text_style=ft.TextStyle(color=TEXT, size=13),
        )
        bt_rsi_tf = ft.TextField(
            label="RSI EÅŸik",
            value="50",
            width=110,
            keyboard_type=ft.KeyboardType.NUMBER,
            bgcolor=CARD2,
            border_color=BORD,
            focused_border_color=GOLD,
            label_style=ft.TextStyle(color=SUB, size=11),
            text_style=ft.TextStyle(color=TEXT, size=13),
        )
        bt_cache_cb = ft.Checkbox(label="Cache kullan", value=True,
                                  check_color=BG, fill_color=GOLD,
                                  label_style=ft.TextStyle(color=SUB, size=12))
        bt_long_cb = ft.Checkbox(label="Long", value=True,
                                 check_color=BG, fill_color=GOLD,
                                 label_style=ft.TextStyle(color=SUB, size=12))
        bt_short_cb = ft.Checkbox(label="Short", value=True,
                                  check_color=BG, fill_color=GOLD,
                                  label_style=ft.TextStyle(color=SUB, size=12))
        bt_run_btn = ft.ElevatedButton(
            "▶  Backtest Çalıştır",
            style=ft.ButtonStyle(
                bgcolor=GOLD, color=BG,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.padding.symmetric(horizontal=18, vertical=10),
            ),
        )
        bt_progress_bar = ft.ProgressBar(visible=False, color=GOLD,
                                         bgcolor=CARD2, width=300)
        bt_status_lbl = ft.Text("", color=SUB, size=12)

        # Metrik kartları (backtest sonuçları)
        bt_c_pnl      = stat_card("Net PNL $",      "—")
        bt_c_pnl_pct  = stat_card("Net PNL %",      "—")
        bt_c_winrate  = stat_card("Win Rate",        "—")
        bt_c_maxdd    = stat_card("Max DD %",        "—")
        bt_c_sharpe   = stat_card("Sharpe",          "—")
        bt_c_pf       = stat_card("Profit Factor",   "—")
        bt_c_trades   = stat_card("Toplam Trade",    "—")
        bt_c_avg_win  = stat_card("Avg Kazanç $",    "—")
        bt_c_avg_loss = stat_card("Avg Kayıp $",     "—")
        bt_c_wl       = stat_card("Win / Loss",      "—")
        bt_c_final    = stat_card("Final Equity $",  "—")

        # Equity curve alanı
        bt_chart_container = ft.Container(
            expand=True,
            content=ft.Text("Backtest çalıştırılmadı...", color=SUB, size=12,
                            text_align=ft.TextAlign.CENTER),
        )
        bt_chart_section = ft.Container(
            bgcolor=CARD, border_radius=12,
            border=ft.border.all(1, BORD),
            padding=12, margin=ft.margin.only(top=8),
            content=ft.Column([
                ft.Text("Equity Curve (Backtest)", size=13, color=GOLD,
                        weight=ft.FontWeight.BOLD),
                ft.Container(height=4),
                bt_chart_container,
            ]),
        )

        # Trade tablosu
        bt_trade_table = ft.DataTable(
            columns=_tbl_cols(
                "#", "Yön", "Giriş", "Çıkış", "PNL $", "PNL %",
                "Sebep", "Giriş Zaman", "Çıkış Zaman"
            ),
            rows=[],
            heading_row_color=CARD2,
            heading_row_height=34,
            data_row_min_height=32,
            data_row_max_height=36,
            column_spacing=12,
        )
        bt_trade_section = ft.Container(
            bgcolor=CARD, border_radius=12,
            border=ft.border.all(1, BORD),
            padding=12, margin=ft.margin.only(top=8),
            content=ft.Column([
                ft.Text("Trade Listesi", size=13, color=GOLD,
                        weight=ft.FontWeight.BOLD),
                ft.Container(height=4),
                ft.Column([bt_trade_table],
                          scroll=ft.ScrollMode.AUTO, height=280),
            ]),
        )

        # Tüm backtest section
        bt_coin_table = ft.DataTable(
            columns=_tbl_cols("Sembol", "Trade", "WR %", "PNL $", "PF"),
            rows=[],
            heading_row_color=CARD2,
            heading_row_height=34,
            data_row_min_height=32,
            data_row_max_height=36,
            column_spacing=18,
        )
        bt_coin_section = ft.Container(
            bgcolor=CARD, border_radius=12,
            border=ft.border.all(1, BORD),
            padding=12, margin=ft.margin.only(top=8),
            content=ft.Column([
                ft.Text("Coin Sonuclari", size=13, color=GOLD,
                        weight=ft.FontWeight.BOLD),
                ft.Container(height=4),
                ft.Column([bt_coin_table],
                          scroll=ft.ScrollMode.AUTO, height=260),
            ]),
        )
        backtest_section = ft.Container(
            visible=False,
            expand=True,
            padding=ft.padding.only(left=12, right=12, bottom=12, top=6),
            content=ft.Column([
                # Config paneli
                ft.Container(
                    bgcolor=CARD, border_radius=12,
                    border=ft.border.all(1, BORD),
                    padding=16,
                    content=ft.Column([
                        ft.Text("Backtest Ayarları", size=13, color=GOLD,
                                weight=ft.FontWeight.BOLD),
                        ft.Container(height=8),
                        ft.Row([
                            bt_scope_dd, bt_symbol_dd, bt_strategy_dd,
                            bt_months_tf, bt_capital_tf, bt_rsi_tf, bt_cache_cb,
                            bt_long_cb, bt_short_cb,
                        ], spacing=12, wrap=True),
                        ft.Container(height=10),
                        ft.Row([
                            bt_run_btn,
                            ft.Container(width=12),
                            bt_progress_bar,
                            ft.Container(width=8),
                            bt_status_lbl,
                        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ], spacing=0),
                ),
                ft.Container(height=8),
                # Metrik kartları — satır 1
                ft.Row([
                    bt_c_pnl, bt_c_pnl_pct, bt_c_winrate,
                    bt_c_maxdd, bt_c_sharpe, bt_c_pf, bt_c_trades,
                ], spacing=8),
                # Metrik kartları — satır 2
                ft.Row([
                    bt_c_avg_win, bt_c_avg_loss, bt_c_wl, bt_c_final,
                ], spacing=8),
                bt_chart_section,
                bt_coin_section,
                bt_trade_section,
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=4),
        )

        # ── Trading area (mevcut içerik) ─────────────────────────────────────
        trading_area = ft.Container(
            expand=True,
            padding=ft.padding.only(left=12, right=12, bottom=12, top=6),
            content=ft.Column([
                ft.Text("Açık İşlemler", size=13, color=GOLD,
                        weight=ft.FontWeight.BOLD),
                ft.Container(height=4),
                ft.Container(
                    bgcolor=CARD, border_radius=12,
                    border=ft.border.all(1, BORD),
                    padding=8,
                    content=ft.Column(
                        [open_table],
                        scroll=ft.ScrollMode.AUTO,
                    ),
                ),
                chart_section,
                history_section,
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=4),
        )

        # ── Ana layout ───────────────────────────────────────────────────────
        main_content = ft.Column([
            header,
            ft.Container(
                padding=ft.padding.symmetric(horizontal=12, vertical=6),
                content=ft.Row(
                    [s_equity, s_pnl, s_total_pnl, s_open, s_win, s_lev, s_btc],
                    spacing=8,
                ),
            ),
            ft.Container(
                padding=ft.padding.symmetric(horizontal=12, vertical=2),
                content=status_bar,
            ),
            trading_area,
            backtest_section,
        ], expand=True, spacing=0, visible=False)

        page.add(main_content)

        # ── Yardımcılar ──────────────────────────────────────────────────────
        def _set_card(card: ft.Container, val: str, color: str = TEXT) -> None:
            card.content.controls[1].value = val
            card.content.controls[1].color = color

        # ── Backtest çalıştırma ───────────────────────────────────────────────
        def _update_backtest_ui(result: BacktestResult) -> None:
            pnl_c = GREEN if result.net_pnl_usd >= 0 else RED
            wr_c  = GREEN if result.win_rate >= 0.5 else RED
            pf_c  = GREEN if result.profit_factor >= 1.0 else RED

            _set_card(bt_c_pnl,     f"${result.net_pnl_usd:+.2f}", pnl_c)
            _set_card(bt_c_pnl_pct, f"{result.net_pnl_pct:+.2f}%", pnl_c)
            _set_card(bt_c_winrate, f"{result.win_rate*100:.1f}%",  wr_c)
            _set_card(bt_c_maxdd,   f"{result.max_drawdown_pct:.1f}%", RED)
            _set_card(bt_c_sharpe,  f"{result.sharpe_ratio:.2f}")
            pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
            _set_card(bt_c_pf,      pf_str, pf_c)
            _set_card(bt_c_trades,  str(result.total_trades))
            _set_card(bt_c_avg_win,  f"${result.avg_win_usd:.2f}",  GREEN)
            _set_card(bt_c_avg_loss, f"${result.avg_loss_usd:.2f}", RED)
            _set_card(bt_c_wl,      f"{result.win_count}W / {result.loss_count}L")
            _set_card(bt_c_final,   f"${result.final_equity:.2f}",  pnl_c)

            # Equity curve
            if result.equity_curve:
                eq_data = [{"equity": e} for e in result.equity_curve]
                bt_chart_container.content = _build_equity_chart(eq_data)

            # Trade tablosu (son 500)
            bt_trade_table.rows = []
            for t in result.trades[-500:]:
                pc = GREEN if t.pnl_usd >= 0 else RED
                bt_trade_table.rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(t.position_id), size=11)),
                    ft.DataCell(ft.Container(
                        padding=ft.padding.symmetric(horizontal=5, vertical=1),
                        border_radius=4,
                        bgcolor=f"{GREEN}22" if t.side == "long" else f"{RED}22",
                        content=ft.Text(
                            t.side.upper(), size=10, weight=ft.FontWeight.BOLD,
                            color=GREEN if t.side == "long" else RED,
                        ),
                    )),
                    ft.DataCell(ft.Text(f"{t.entry_price:.4f}", size=11)),
                    ft.DataCell(ft.Text(f"{t.exit_price:.4f}",  size=11)),
                    ft.DataCell(ft.Text(f"{t.pnl_usd:+.2f}",   size=11, color=pc)),
                    ft.DataCell(ft.Text(f"{t.pnl_pct:+.1f}%",  size=11, color=pc)),
                    ft.DataCell(ft.Text(t.reason, size=11, color=SUB)),
                    ft.DataCell(ft.Text(
                        t.entry_time.strftime("%m-%d %H:%M"), size=10, color=SUB)),
                    ft.DataCell(ft.Text(
                        t.exit_time.strftime("%m-%d %H:%M"),  size=10, color=SUB)),
                ]))

            page.update()

        def _update_backtest_ui(result_obj: BacktestResult | MultiBacktestResult) -> None:
            summary = result_obj.summary if isinstance(result_obj, MultiBacktestResult) else result_obj
            coin_results = result_obj.results if isinstance(result_obj, MultiBacktestResult) else [result_obj]

            pnl_c = GREEN if summary.net_pnl_usd >= 0 else RED
            wr_c = GREEN if summary.win_rate >= 0.5 else RED
            pf_c = GREEN if summary.profit_factor >= 1.0 else RED

            _set_card(bt_c_pnl, f"${summary.net_pnl_usd:+.2f}", pnl_c)
            _set_card(bt_c_pnl_pct, f"{summary.net_pnl_pct:+.2f}%", pnl_c)
            _set_card(bt_c_winrate, f"{summary.win_rate*100:.1f}%", wr_c)
            _set_card(bt_c_maxdd, f"{summary.max_drawdown_pct:.1f}%", RED)
            _set_card(bt_c_sharpe, f"{summary.sharpe_ratio:.2f}")
            pf_str = f"{summary.profit_factor:.2f}" if summary.profit_factor != float("inf") else "inf"
            _set_card(bt_c_pf, pf_str, pf_c)
            _set_card(bt_c_trades, str(summary.total_trades))
            _set_card(bt_c_avg_win, f"${summary.avg_win_usd:.2f}", GREEN)
            _set_card(bt_c_avg_loss, f"${summary.avg_loss_usd:.2f}", RED)
            _set_card(bt_c_wl, f"{summary.win_count}W / {summary.loss_count}L")
            _set_card(bt_c_final, f"${summary.final_equity:.2f}", pnl_c)

            if summary.equity_curve:
                bt_chart_container.content = _build_equity_chart(
                    [{"equity": e} for e in summary.equity_curve]
                )

            bt_coin_table.rows = []
            for coin in sorted(coin_results, key=lambda r: r.net_pnl_usd, reverse=True):
                pf_val = "inf" if coin.profit_factor == float("inf") else f"{coin.profit_factor:.2f}"
                pnl_color = GREEN if coin.net_pnl_usd >= 0 else RED
                bt_coin_table.rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(coin.symbol.replace(":USDT", ""), size=11, color=GOLD)),
                    ft.DataCell(ft.Text(str(coin.total_trades), size=11)),
                    ft.DataCell(ft.Text(
                        f"{coin.win_rate*100:.1f}%",
                        size=11,
                        color=GREEN if coin.win_rate >= 0.5 else RED,
                    )),
                    ft.DataCell(ft.Text(f"{coin.net_pnl_usd:+.2f}", size=11, color=pnl_color)),
                    ft.DataCell(ft.Text(pf_val, size=11, color=pnl_color)),
                ]))

            bt_trade_table.rows = []
            for t in summary.trades[-500:]:
                pnl_color = GREEN if t.pnl_usd >= 0 else RED
                bt_trade_table.rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(
                        f"{t.position_id} {t.symbol.replace(':USDT', '')}",
                        size=11,
                        color=GOLD,
                    )),
                    ft.DataCell(ft.Text(
                        t.side.upper(),
                        size=10,
                        color=GREEN if t.side == "long" else RED,
                    )),
                    ft.DataCell(ft.Text(f"{t.entry_price:.4f}", size=11)),
                    ft.DataCell(ft.Text(f"{t.exit_price:.4f}", size=11)),
                    ft.DataCell(ft.Text(f"{t.pnl_usd:+.2f}", size=11, color=pnl_color)),
                    ft.DataCell(ft.Text(f"{t.pnl_pct:+.1f}%", size=11, color=pnl_color)),
                    ft.DataCell(ft.Text(t.reason, size=11, color=SUB)),
                    ft.DataCell(ft.Text(t.entry_time.strftime("%m-%d %H:%M"), size=10, color=SUB)),
                    ft.DataCell(ft.Text(t.exit_time.strftime("%m-%d %H:%M"), size=10, color=SUB)),
                ]))

            page.update()

        def on_run_backtest(_):
            if not engine_ref:
                bt_status_lbl.value = "Engine başlatılmadı!"
                page.update()
                return
            bt_run_btn.disabled = True
            bt_progress_bar.visible = True
            bt_trade_table.rows = []
            bt_status_lbl.value = "Başlatılıyor..."
            page.update()

            def _run():
                try:
                    bt_cfg = BacktestConfig(
                        symbol=bt_symbol_dd.value or settings.exchange.symbol,
                        timeframe=settings.exchange.timeframe,
                        months=int(bt_months_tf.value or "24"),
                        initial_equity=float(bt_capital_tf.value or "1000"),
                        use_cache=bt_cache_cb.value,
                    )

                    def _progress(msg: str, pct: float) -> None:
                        bt_status_lbl.value = msg
                        bt_progress_bar.value = pct if pct > 0 else None
                        page.update()

                    engine = engine_ref[0]
                    bt_engine = BacktestEngine(engine.settings, engine.exchange)
                    df = bt_engine.fetch_data(bt_cfg, _progress)
                    htf_df = bt_engine.fetch_htf_data(bt_cfg, _progress)

                    if df.empty:
                        bt_status_lbl.value = "Veri alınamadı — sembolü veya bağlantıyı kontrol edin."
                        page.update()
                        return

                    result = bt_engine.run(bt_cfg, df, _progress, htf_df=htf_df)
                    _update_backtest_ui(result)

                    bt_status_lbl.value = (
                        f"Tamamlandı — {result.total_trades} trade  |  "
                        f"{result.start_date.strftime('%Y-%m-%d')} → "
                        f"{result.end_date.strftime('%Y-%m-%d')}  |  "
                        f"{result.total_bars:,} bar"
                    )
                    page.update()

                except Exception as exc:  # noqa: BLE001
                    bt_status_lbl.value = f"Hata: {exc}"
                    page.update()
                finally:
                    bt_run_btn.disabled = False
                    bt_progress_bar.visible = False
                    page.update()

            threading.Thread(target=_run, daemon=True).start()

        def on_run_backtest(_):
            if not engine_ref:
                bt_status_lbl.value = "Engine baslatilmadi!"
                page.update()
                return
            if not bt_long_cb.value and not bt_short_cb.value:
                bt_status_lbl.value = "En az bir yon secmelisin."
                page.update()
                return

            bt_run_btn.disabled = True
            bt_progress_bar.visible = True
            bt_trade_table.rows = []
            bt_coin_table.rows = []
            bt_status_lbl.value = "Baslatiliyor..."
            page.update()

            def _run():
                try:
                    bt_cfg = BacktestConfig(
                        symbol=bt_symbol_dd.value or settings.exchange.symbol,
                        timeframe=settings.exchange.timeframe,
                        months=int(bt_months_tf.value or "24"),
                        initial_equity=float(bt_capital_tf.value or "1000"),
                        use_cache=bool(bt_cache_cb.value),
                        rsi_threshold=float(bt_rsi_tf.value or "50"),
                        allow_long=bool(bt_long_cb.value),
                        allow_short=bool(bt_short_cb.value),
                    )

                    def _progress(msg: str, pct: float) -> None:
                        bt_status_lbl.value = msg
                        bt_progress_bar.value = pct if pct > 0 else None
                        page.update()

                    engine = engine_ref[0]
                    bt_engine = BacktestEngine(engine.settings, engine.exchange)

                    if bt_scope_dd.value == "majors20":
                        symbols = engine.exchange.fetch_universe_symbols(20)
                        result_obj = bt_engine.run_multi(symbols, bt_cfg, _progress)
                        summary = result_obj.summary
                    else:
                        df = bt_engine.fetch_data(bt_cfg, _progress)
                        htf_df = bt_engine.fetch_htf_data(bt_cfg, _progress)
                        if df.empty:
                            bt_status_lbl.value = "Veri alinamadi."
                            page.update()
                            return
                        result_obj = bt_engine.run(bt_cfg, df, _progress, htf_df=htf_df)
                        summary = result_obj

                    _update_backtest_ui(result_obj)
                    bt_status_lbl.value = (
                        f"Tamamlandi - {summary.total_trades} trade | "
                        f"{summary.start_date.strftime('%Y-%m-%d')} -> "
                        f"{summary.end_date.strftime('%Y-%m-%d')} | "
                        f"{summary.total_bars:,} bar"
                    )
                    page.update()
                except Exception as exc:  # noqa: BLE001
                    bt_status_lbl.value = f"Hata: {exc}"
                    page.update()
                finally:
                    bt_run_btn.disabled = False
                    bt_progress_bar.visible = False
                    page.update()

            threading.Thread(target=_run, daemon=True).start()

        bt_run_btn.on_click = on_run_backtest

        # ── Refresh ──────────────────────────────────────────────────────────
        def refresh_view() -> None:
            if not engine_ref:
                return
            snap = engine_ref[0].snapshot()
            lev  = snap.leverage

            _set_card(s_equity, f"${snap.equity:.2f}")
            _set_card(s_pnl,    f"${snap.daily_pnl:.2f}",
                      GREEN if snap.daily_pnl >= 0 else RED)
            _set_card(s_total_pnl, f"${snap.total_pnl:.2f}",
                      GREEN if snap.total_pnl >= 0 else RED)
            _set_card(s_open,   str(len(snap.open_positions)))

            if snap.recent_trades:
                wins = sum(1 for t in snap.recent_trades if t["pnl_usd"] > 0)
                wr   = wins / len(snap.recent_trades) * 100
                _set_card(s_win, f"{wr:.1f}%", GREEN if wr >= 50 else RED)

            for sym, pr in snap.last_prices.items():
                if sym.startswith("BTC/") and pr > 0:
                    _set_card(s_btc, f"${pr:,.0f}", GOLD)
                    break

            run_dot.bgcolor    = GREEN if snap.running else RED
            signal_lbl.value   = f"Son sinyal: {snap.last_signal}"
            error_lbl.value    = f"⚠ {snap.last_error}" if snap.last_error else ""
            halt_badge.visible = snap.daily_loss_halted
            live = not snap.paper_mode
            mode_text.value    = "🔴 LIVE" if live else "PAPER"
            mode_text.color    = RED if live else GOLD
            mode_badge.bgcolor = f"{RED}22" if live else "#F3BA2F22"
            mode_badge.border  = ft.border.all(1, RED if live else GOLD)

            # Equity curve güncelle
            if chart_section.visible and snap.equity_curve:
                chart_container.content = _build_equity_chart(snap.equity_curve)

            # Açık işlemler
            rows = []
            for p in snap.open_positions:
                cur  = snap.last_prices.get(p.symbol, p.entry_price)
                upnl = ((cur - p.entry_price) * p.qty if p.side == "long"
                        else (p.entry_price - cur) * p.qty)
                upct = upnl / max(p.entry_price * p.qty, 1e-9) * 100
                pc   = GREEN if upnl >= 0 else RED
                sym_s = p.symbol.split(":")[0].replace("/USDT", "")

                rows.append(ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(p.id), size=12, color=SUB)),
                        ft.DataCell(ft.Text(sym_s, size=13, color=GOLD,
                                           weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Container(
                            padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            border_radius=4,
                            bgcolor=f"{GREEN}22" if p.side == "long" else f"{RED}22",
                            content=ft.Text(
                                "LONG" if p.side == "long" else "SHORT",
                                size=11, weight=ft.FontWeight.BOLD,
                                color=GREEN if p.side == "long" else RED,
                            ),
                        )),
                        ft.DataCell(ft.Text(f"{p.entry_price:.4f}", size=12)),
                        ft.DataCell(ft.Text(f"{cur:.4f}" if cur else "—", size=12)),
                        ft.DataCell(ft.Text(f"{p.stop_loss:.4f}", size=12, color=RED)),
                        ft.DataCell(ft.Text(f"{p.take_profit:.4f}", size=12, color=GREEN)),
                        ft.DataCell(ft.Text(f"{upnl:+.2f}", size=12, color=pc,
                                           weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(f"{upct:+.1f}%", size=12, color=pc)),
                    ],
                    on_select_changed=_open_detail(p, cur, lev),
                ))
            open_table.rows = rows

            # Geçmiş
            history_table.rows = [
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(str(t["position_id"]), size=11)),
                    ft.DataCell(ft.Text(
                        str(t["symbol"]).replace("/USDT:USDT","").replace("/USDT",""),
                        size=11, color=GOLD)),
                    ft.DataCell(ft.Text(str(t["side"]).upper(), size=11,
                                       color=GREEN if t["side"]=="long" else RED)),
                    ft.DataCell(ft.Text(f"{t['entry_price']:.4f}", size=11)),
                    ft.DataCell(ft.Text(f"{t['exit_price']:.4f}",  size=11)),
                    ft.DataCell(ft.Text(f"{t['pnl_usd']:+.2f}",    size=11,
                                       color=GREEN if t["pnl_usd"]>=0 else RED)),
                    ft.DataCell(ft.Text(str(t["reason"]), size=11)),
                    ft.DataCell(ft.Text(
                        str(t["closed_at"])[0:16].replace("T"," "), size=11)),
                ])
                for t in snap.recent_trades[:40]
            ]

            # Toplam PNL etiketi
            tp = snap.total_pnl
            history_total_lbl.value = f"Toplam PNL: ${tp:+.2f}"
            history_total_lbl.color = GREEN if tp >= 0 else RED

            page.update()

        # ── Engine init ──────────────────────────────────────────────────────
        def _do_init() -> None:
            try:
                splash_msg.value = "Piyasa verileri alınıyor..."
                page.update()
                e = TradeEngine(settings)
                engine_ref.append(e)
                bt_symbol_dd.options = [ft.dropdown.Option(s) for s in e.symbols]
                if e.symbols:
                    bt_symbol_dd.value = e.symbols[0]
            except Exception as ex:
                splash_msg.value = f"Bağlantı hatası: {ex}"
                page.update()
                return

            splash.visible       = False
            main_content.visible = True
            page.update()
            page.run_task(_periodic_refresh, refresh_view)

        threading.Thread(target=_do_init, daemon=True).start()

    async def _periodic_refresh(update_fn):
        while True:
            update_fn()
            await asyncio.sleep(2)

    ft.app(target=main, assets_dir="assets")
