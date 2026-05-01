"""
QRS Research — Strategy Comparison & Backtesting
FerryRichMan Limited
────────────────────────────────────────────────────────────────────────────
Compares seven strategies / benchmarks:
  1. QRS Standard          – relative momentum (SPY / VEU / BIL)
  2. QRS + AM Guard        – + absolute-momentum safety net (12M < 0 → BIL)
  3. QRS + Trend (TF)      – + 10-month SMA trend filter on SPY
  4. QRS + TF + AM         – trend filter AND AM Guard combined
  5. SPY Buy-and-Hold      – S&P 500 benchmark
  6. TLT Buy-and-Hold      – 20+ Year Treasury benchmark
  7. GLD Buy-and-Hold      – Gold benchmark

Trend-Filter rule (趨勢濾網):
  End-of-month SPY close  > 10-month SMA(SPY) → uptrend  → use QRS signal
  End-of-month SPY close ≤ 10-month SMA(SPY) → downtrend → override to BIL
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from app import (
    load_prices, compute_signals, _hk_date_key,
    PERF_YEAR_START, TICKERS, inject_css,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QRS Research · FerryRichMan",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "QRS Standard":     "#818cf8",
    "QRS + AM Guard":   "#34d399",
    "QRS + Trend (TF)": "#f472b6",
    "QRS + TF + AM":    "#fb923c",
    "SPY B&H":          "#94a3b8",
    "TLT B&H":          "#38bdf8",
    "GLD B&H":          "#fbbf24",
}

# ── Benchmark loader ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86_400, show_spinner=False)
def _load_bm(date_key: str = "") -> dict:
    """Monthly adj-close for TLT and GLD (benchmark buy-and-hold only)."""
    raw = yf.download(
        ["TLT", "GLD"], start="2002-01-01",
        auto_adjust=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw
    monthly = closes.resample("ME").last()
    return {t: monthly[t].dropna() for t in ["TLT", "GLD"] if t in monthly.columns}


# ── Trend-filter helper ───────────────────────────────────────────────────────
def add_trend_filter(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    趨勢濾網 (Trend Filter):
      SPY close > rolling(window)-month SMA  → uptrend  → keep QRS signal
      SPY close ≤ rolling(window)-month SMA  → downtrend → go to BIL

    Adds: signal_tf, signal_tf_am, signal_return_tf, signal_return_tf_am
    """
    df = prices.copy()
    spy_sma = df["SPY"].rolling(window, min_periods=window).mean()
    df["trend_up"] = df["SPY"] > spy_sma

    # TF signal (based on QRS standard)
    df["signal_tf"] = df["signal"].copy()
    df.loc[~df["trend_up"].fillna(False), "signal_tf"] = "BIL"

    # TF + AM Guard
    df["signal_tf_am"] = df["signal_am"].copy()
    df.loc[~df["trend_up"].fillna(False), "signal_tf_am"] = "BIL"

    # Monthly returns for TF signals
    for ret_col, sig_col in [
        ("signal_return_tf",    "signal_tf"),
        ("signal_return_tf_am", "signal_tf_am"),
    ]:
        prev = df[sig_col].shift(1)
        df[ret_col] = np.nan
        for t in TICKERS:
            mask        = prev == t
            holding_ret = df[f"{t}_open"].shift(-1) / df[f"{t}_open"] - 1
            df.loc[mask, ret_col] = holding_ret[mask]

    return df


# ── Performance helpers ───────────────────────────────────────────────────────
def _cum(rets: pd.Series) -> pd.Series:
    r = rets.dropna()
    return (1 + r).cumprod()


def _kpis(rets: pd.Series, name: str) -> dict:
    r = rets.dropna()
    if len(r) < 6:
        return {"策略": name, "年化回報": None, "年化波幅": None,
                "夏普比率": None, "最大回撤": None, "勝率": None}
    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum     = (1 + r).cumprod()
    dd      = (cum / cum.cummax() - 1).min()
    win     = (r > 0).mean()
    return {
        "策略":    name,
        "年化回報": ann_ret,
        "年化波幅": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": dd,
        "勝率":    win,
    }


# ── Load & process data ───────────────────────────────────────────────────────
with st.spinner("正在載取市場數據…"):
    prices  = load_prices(date_key=_hk_date_key())
    prices  = compute_signals(prices)
    prices  = add_trend_filter(prices, window=10)
    bm_data = _load_bm(date_key=_hk_date_key())

pf = prices[prices.index.year >= PERF_YEAR_START].copy()

# SPY B&H return from open prices (same execution as strategies)
pf["spy_bh"] = pf["SPY_open"].shift(-1) / pf["SPY_open"] - 1


def _bm_rets(series: pd.Series) -> pd.Series:
    r = series.pct_change().dropna()
    return r[r.index.year >= PERF_YEAR_START]


strats = {
    "QRS Standard":     pf["signal_return"].dropna(),
    "QRS + AM Guard":   pf["signal_return_am"].dropna(),
    "QRS + Trend (TF)": pf["signal_return_tf"].dropna(),
    "QRS + TF + AM":    pf["signal_return_tf_am"].dropna(),
    "SPY B&H":          pf["spy_bh"].dropna(),
}
if "TLT" in bm_data:
    strats["TLT B&H"] = _bm_rets(bm_data["TLT"])
if "GLD" in bm_data:
    strats["GLD B&H"] = _bm_rets(bm_data["GLD"])


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:22px;font-weight:800;color:#e2e8f0;'
    'letter-spacing:-0.5px;padding:20px 0 4px;">🔬 QRS Research — 策略回測比較</div>'
    '<div style="font-size:13px;color:#64748b;margin-bottom:18px;">'
    'QRS Standard &nbsp;·&nbsp; AM Guard &nbsp;·&nbsp; '
    '趨勢濾網 10M-SMA &nbsp;·&nbsp; SPY / TLT / GLD 買入持有</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chart, tab_kpi, tab_annual, tab_info = st.tabs(
    ["📈 累積回報", "📊 績效指標", "📅 年度回報", "ℹ️ 策略說明"]
)

# ─────────────────────────── Tab 1: cumulative chart ─────────────────────────
with tab_chart:
    fig = go.Figure()
    for name, rets in strats.items():
        cum   = _cum(rets)
        color = COLORS.get(name, "#64748b")
        is_bh = "B&H" in name
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=name,
            line=dict(color=color, width=1.5 if is_bh else 2.2,
                      dash="dot" if is_bh else "solid"),
            hovertemplate=(
                f"<b>{name}</b><br>%{{x|%Y-%m}}<br>"
                "累計：%{y:.2f}x<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"策略累積回報（{PERF_YEAR_START} 年至今，起始 = 1.0）",
            font=dict(color="#e2e8f0", size=14),
        ),
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#111827",
        font=dict(color="#94a3b8", size=12),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155",
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155",
                   tickformat=".1f", title="倍數 (x)"),
        margin=dict(l=8, r=8, t=44, b=8),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mini annualised-return summary row
    cols = st.columns(len(strats))
    for col, (name, rets) in zip(cols, strats.items()):
        r   = rets.dropna()
        ann = (1 + r.mean()) ** 12 - 1 if len(r) >= 6 else float("nan")
        clr = COLORS.get(name, "#64748b")
        col.markdown(
            f'<div style="text-align:center;padding:8px 2px;">'
            f'<div style="font-size:9px;color:{clr};font-weight:700;">{name}</div>'
            f'<div style="font-size:14px;font-weight:800;color:#e2e8f0;">'
            f'{ann*100:+.1f}%</div>'
            f'<div style="font-size:9px;color:#475569;">年化</div></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────── Tab 2: KPI table ────────────────────────────────
with tab_kpi:
    rows = [_kpis(rets, name) for name, rets in strats.items()]
    df_k = pd.DataFrame(rows)

    def _p(v, plus=False):
        if pd.isna(v) or v is None:
            return "—"
        return f"{v*100:+.1f}%" if plus else f"{v*100:.1f}%"

    df_k["年化回報"] = df_k["年化回報"].apply(lambda x: _p(x, plus=True))
    df_k["年化波幅"] = df_k["年化波幅"].apply(_p)
    df_k["夏普比率"] = df_k["夏普比率"].apply(
        lambda x: f"{x:.2f}" if (x is not None and pd.notna(x)) else "—"
    )
    df_k["最大回撤"] = df_k["最大回撤"].apply(lambda x: _p(x, plus=True))
    df_k["勝率"]    = df_k["勝率"].apply(
        lambda x: f"{x*100:.0f}%" if (x is not None and pd.notna(x)) else "—"
    )

    st.dataframe(
        df_k, use_container_width=True, hide_index=True,
        column_config={
            "策略":    st.column_config.TextColumn("策略",    width=160),
            "年化回報": st.column_config.TextColumn("年化回報", width=90),
            "年化波幅": st.column_config.TextColumn("年化波幅", width=90),
            "夏普比率": st.column_config.TextColumn("夏普比率", width=80),
            "最大回撤": st.column_config.TextColumn("最大回撤", width=90),
            "勝率":    st.column_config.TextColumn("勝率",    width=70),
        },
    )
    st.markdown(
        f'<div style="font-size:11px;color:#475569;margin-top:6px;">'
        f'回測期間：{PERF_YEAR_START} 年至今。'
        f'年化波幅與夏普比率基於月度回報計算。夏普比率使用 0% 無風險利率（保守估算）。</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────── Tab 3: annual returns ───────────────────────────
with tab_annual:
    annual = {}
    for name, rets in strats.items():
        r = rets.dropna()
        annual[name] = r.groupby(r.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )

    all_years = sorted({yr for a in annual.values() for yr in a.index})

    fig2 = go.Figure()
    for name, ann in annual.items():
        color  = COLORS.get(name, "#64748b")
        y_vals = [ann.get(yr) for yr in all_years]
        fig2.add_trace(go.Bar(
            name=name,
            x=[str(y) for y in all_years],
            y=[v * 100 if v is not None else None for v in y_vals],
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:+.1f}}%<extra></extra>",
        ))

    fig2.update_layout(
        title=dict(text="年度回報比較 (%)", font=dict(color="#e2e8f0", size=14)),
        barmode="group",
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#111827",
        font=dict(color="#94a3b8", size=12),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155",
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
        margin=dict(l=8, r=8, t=44, b=8),
        height=440,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────── Tab 4: strategy info ────────────────────────────
with tab_info:
    info_items = [
        ("🔵", "QRS Standard",
         "相對動力系統：計算 SPY / VEU / BIL 的加權動力分數"
         "（3M×0.8 + 6M×0.6 + 9M×0.4 + 12M×0.2），"
         "每月末選出分數最高的 ETF，次月首個交易日開市執行。"),
        ("🟢", "QRS + AM Guard（絕對動力濾網）",
         "在相對動力選出 ETF 後，檢查其 12 個月絕對回報。"
         "若所選 ETF 自身 12M 回報 < 0（自身處於下跌趨勢），"
         "強制轉入 BIL（現金）。BIL 本身不受覆蓋。"),
        ("🩷", "QRS + Trend (TF)（趨勢濾網）",
         "若 SPY 月底收市價 > SPY 近 10 個月簡單移動平均線 (SMA10)"
         " → 市場上升趨勢，正常使用 QRS 訊號；"
         "若 SPY ≤ SMA10 → 市場下降趨勢，強制轉入 BIL。"),
        ("🟠", "QRS + TF + AM（雙重濾網）",
         "同時使用趨勢濾網（TF）和絕對動力濾網（AM Guard）。"
         "市場下跌趨勢時轉 BIL，上升趨勢時再用 AM Guard 確認所選 ETF 自身動力。"),
        ("⚫", "SPY B&H",
         "標普 500 ETF 買入持有（月度複利再投資）。股票市場基準。"),
        ("🔵", "TLT B&H",
         "iShares 20+ 年期美國長期國債 ETF 買入持有。"
         "作為債券市場基準，與 QRS 策略比較利率週期對報酬的影響。"),
        ("🟡", "GLD B&H",
         "SPDR 黃金 ETF 買入持有。"
         "作為黃金資產基準，反映避險情緒與通脹對資產的影響。"),
    ]

    for icon, title, desc in info_items:
        st.markdown(
            f'<div style="font-size:13px;color:#94a3b8;line-height:1.9;margin-bottom:14px;">'
            f'<span style="color:#e2e8f0;font-weight:700;">{icon} {title}</span><br>'
            f'{desc}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="border-top:1px solid #1e293b;margin-top:8px;padding-top:10px;'
        f'font-size:11px;color:#475569;">'
        f'回測期間：{PERF_YEAR_START} 年至今。'
        f'月度回報基於調整後收市價計算（SPY 策略使用首個交易日開市價執行）。'
        f'績效僅供研究參考，不構成任何投資建議。'
        f'FerryRichMan Limited · QRS Standard Signal System</div>',
        unsafe_allow_html=True,
    )
