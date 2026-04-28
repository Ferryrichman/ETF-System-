"""
QRS Research — Dual Momentum Comparison
FerryRichMan Limited
────────────────────────────────────────
This page is for research only and does NOT affect the live signal system.
It compares the current QRS strategy vs a Dual Momentum variant.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Re-use data loading from main app
from app import load_prices, compute_signals, _hk_date_key, TICKERS, WEIGHTS, PERF_YEAR_START

st.set_page_config(
    page_title="QRS Research · FerryRichMan",
    page_icon="🔬",
    layout="centered",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0f1e !important; }
[data-testid="stHeader"]           { background: transparent !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 820px; }
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
[data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 14px; }
h1,h2,h3 { color: #e2e8f0 !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 20px;border-bottom:1px solid #1e293b;margin-bottom:24px;">
  <div style="font-size:11px;color:#6366f1;font-weight:800;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
    FerryRichMan Limited · Research
  </div>
  <div style="font-size:28px;font-weight:900;color:#f1f5f9;letter-spacing:-1px;">
    策略比較分析
  </div>
  <div style="font-size:13px;color:#475569;margin-top:6px;">
    QRS Standard &nbsp;vs&nbsp; QRS + Dual Momentum Filter &nbsp;vs&nbsp; SPY 買入持有
  </div>
  <div style="font-size:11px;color:#334155;margin-top:8px;padding:8px 12px;
    background:#111827;border-radius:8px;border:1px solid #1e293b;">
    ⚠️ 此頁僅供研究參考，不影響主系統訊號
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  LOAD & COMPUTE
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_research_data(date_key: str = ""):
    prices = load_prices(date_key=date_key)
    prices = compute_signals(prices)
    return prices

with st.spinner("載入數據中..."):
    prices = load_research_data(date_key=_hk_date_key())

# ══════════════════════════════════════════════════════════
#  DUAL MOMENTUM SIGNAL
# ══════════════════════════════════════════════════════════
# Rule: if the relative winner's 12M momentum < BIL's 12M momentum → hold BIL
prices2 = prices.copy()
valid = prices2["signal"].notna()

winner_12m = pd.Series(np.nan, index=prices2.index)
for t in TICKERS:
    mask = prices2["signal"] == t
    winner_12m[mask] = prices2.loc[mask, f"{t}_12m"]

bil_12m = prices2["BIL_12m"]
override = valid & (winner_12m < bil_12m)
prices2["signal_dm"] = prices2["signal"].copy()
prices2.loc[override, "signal_dm"] = "BIL"

# Dual Momentum returns
prev_dm = prices2["signal_dm"].shift(1)
prices2["signal_return_dm"] = np.nan
for t in TICKERS:
    mask = prev_dm == t
    holding = prices2[f"{t}_open"].shift(-1) / prices2[f"{t}_open"] - 1
    prices2.loc[mask, "signal_return_dm"] = holding[mask]

# SPY B&H returns (aligned)
prices2["spy_ret"] = prices2["SPY_open"].shift(-1) / prices2["SPY_open"] - 1

n_overrides = int(override.sum())
override_months = prices2.index[override].strftime("%Y-%m").tolist()

# ══════════════════════════════════════════════════════════
#  STATS FUNCTION
# ══════════════════════════════════════════════════════════
def calc(ret_series, label, start_yr=PERF_YEAR_START):
    r = ret_series.dropna()
    r = r[r.index.year >= start_yr]
    if len(r) < 12:
        return None
    cum    = (1 + r).cumprod()
    total  = float(cum.iloc[-1] - 1)
    n_yrs  = len(r) / 12
    cagr   = float((1 + total) ** (1/n_yrs) - 1)
    rm     = cum.cummax()
    dd     = (cum - rm) / rm
    mdd    = float(dd.min())
    mdd_e  = dd.idxmin()
    mdd_s  = cum[:mdd_e].idxmax()
    # longest months underwater
    max_uw, cur = 0, 0
    for v in (dd < 0):
        cur = cur + 1 if v else 0
        max_uw = max(max_uw, cur)
    sharpe  = float(r.mean() * np.sqrt(12) / r.std()) if r.std() > 0 else None
    calmar  = cagr / abs(mdd) if mdd != 0 else None
    win_rt  = float((r > 0).mean())
    return {
        "label":      label,
        "cagr":       cagr,
        "mdd":        mdd,
        "mdd_start":  mdd_s.strftime("%Y-%m"),
        "mdd_end":    mdd_e.strftime("%Y-%m"),
        "max_uw_mo":  max_uw,
        "sharpe":     sharpe,
        "calmar":     calmar,
        "win_rate":   win_rt,
        "total_ret":  total,
        "n_years":    n_yrs,
        "final_10k":  float(cum.iloc[-1] * 10000),
        "cum":        cum * 10000,
    }

sA   = calc(prices2["signal_return"],    "QRS Standard")
sB   = calc(prices2["signal_return_dm"], "QRS + Dual Momentum")
sSPY = calc(prices2["spy_ret"],          "SPY 買入持有")

# ══════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════
def kpi_card(val, label, color="#94a3b8", sub=None):
    sub_html = f'<div style="font-size:10px;color:#334155;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:#111827;border:1px solid #1e293b;border-radius:12px;'
        f'padding:14px 10px;text-align:center;">'
        f'<div style="font-size:22px;font-weight:900;color:{color};letter-spacing:-0.5px;">{val}</div>'
        f'<div style="font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;margin-top:4px;">{label}</div>'
        f'{sub_html}</div>'
    )

colors = {"QRS Standard": "#818cf8", "QRS + Dual Momentum": "#34d399", "SPY 買入持有": "#94a3b8"}

for s in [sA, sB, sSPY]:
    if s is None:
        continue
    clr = colors[s["label"]]
    st.markdown(
        f'<div style="font-size:13px;font-weight:800;color:{clr};letter-spacing:1px;'
        f'text-transform:uppercase;margin:20px 0 10px;">{s["label"]}</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f'{s["cagr"]*100:.1f}%',    "CAGR",         "#4ade80" if s["cagr"] >= 0 else "#f87171"),
        (c2, f'{s["mdd"]*100:.1f}%',     "最大回撤",     "#f87171"),
        (c3, f'{s["max_uw_mo"]}個月',    "最長水下",     "#fb923c"),
        (c4, f'{s["sharpe"]:.2f}' if s["sharpe"] else "—", "Sharpe",  "#818cf8"),
        (c5, f'${s["final_10k"]:,.0f}',  "$10k 增長至",  clr),
    ]
    for col, val, lbl, c in cards:
        with col:
            st.markdown(kpi_card(val, lbl, c), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  DETAILED TABLE
# ══════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:28px;font-size:13px;font-weight:800;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">📊 詳細對比</div>',
            unsafe_allow_html=True)

rows_data = []
metrics = [
    ("CAGR",          lambda s: f'{s["cagr"]*100:.2f}%'),
    ("MDD",           lambda s: f'{s["mdd"]*100:.2f}%'),
    ("MDD 期間",      lambda s: f'{s["mdd_start"]} ~ {s["mdd_end"]}'),
    ("最長水下月數",  lambda s: f'{s["max_uw_mo"]} 個月'),
    ("Sharpe Ratio",  lambda s: f'{s["sharpe"]:.3f}' if s["sharpe"] else "—"),
    ("Calmar Ratio",  lambda s: f'{s["calmar"]:.3f}' if s["calmar"] else "—"),
    ("月勝率",        lambda s: f'{s["win_rate"]*100:.1f}%'),
    ("總回報",        lambda s: f'{s["total_ret"]*100:.1f}%'),
    ("$10k 增長至",   lambda s: f'${s["final_10k"]:,.0f}'),
    ("回測年數",      lambda s: f'{s["n_years"]:.1f} 年'),
]

for name, fn in metrics:
    row = {"指標": name}
    for s in [sA, sB, sSPY]:
        if s:
            row[s["label"]] = fn(s)
    rows_data.append(row)

df_table = pd.DataFrame(rows_data).set_index("指標")
st.dataframe(df_table, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  GROWTH CHART (all 3)
# ══════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:28px;font-size:13px;font-weight:800;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">📈 增長曲線比較</div>',
            unsafe_allow_html=True)

fig = go.Figure()
line_styles = [
    (sA,   "#818cf8", "solid",  2.5),
    (sB,   "#34d399", "solid",  2.0),
    (sSPY, "#475569", "dot",    1.5),
]
for s, color, dash, width in line_styles:
    if s:
        fig.add_trace(go.Scatter(
            x=s["cum"].index, y=s["cum"].values,
            name=s["label"], mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate="%{x|%Y-%m}  <b>$%{y:,.0f}</b><extra>" + s["label"] + "</extra>",
        ))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
    yaxis=dict(showgrid=True, gridcolor="#1e293b",
               tickfont=dict(color="#64748b", size=10),
               tickprefix="$", tickformat=",.0f"),
    hovermode="x unified",
    legend=dict(orientation="h", x=0, y=1.1,
                font=dict(color="#64748b", size=11),
                bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════
#  OVERRIDE MONTHS
# ══════════════════════════════════════════════════════════
st.markdown(
    f'<div style="margin-top:24px;font-size:13px;font-weight:800;color:#94a3b8;'
    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">'
    f'🔄 Dual Momentum 覆蓋月份（共 {n_overrides} 個月）</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#475569;margin-bottom:10px;">'
    '以下月份：QRS 相對動能選出非 BIL，但絕對動能過濾後改為持有 BIL</div>',
    unsafe_allow_html=True,
)

ov_rows = []
for dt in prices2.index[override]:
    orig = prices2.loc[dt, "signal"]
    w12  = prices2.loc[dt, f"{orig}_12m"] if orig in TICKERS else np.nan
    b12  = prices2.loc[dt, "BIL_12m"]
    ov_rows.append({
        "月份": dt.strftime("%Y-%m"),
        "原訊號": orig,
        f"原ETF 12M": f"{w12*100:.2f}%" if pd.notna(w12) else "—",
        "BIL 12M": f"{b12*100:.2f}%" if pd.notna(b12) else "—",
        "改為": "BIL",
    })

if ov_rows:
    st.dataframe(pd.DataFrame(ov_rows), use_container_width=True, hide_index=True)
else:
    st.info("沒有被覆蓋的月份")

st.markdown(
    '<div style="margin-top:24px;font-size:11px;color:#334155;text-align:center;">'
    'FerryRichMan Limited · 研究分析頁面 · 不構成任何投資建議</div>',
    unsafe_allow_html=True,
)
