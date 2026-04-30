"""
QRS Research — Strategy Comparison
FerryRichMan Limited
────────────────────────────────────────
Compares QRS Standard vs Enhanced Universe (+ TLT/GLD)
vs Enhanced + Trend Filter vs SPY Buy-and-Hold.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

from app import load_prices, compute_signals, _hk_date_key, WEIGHTS, PERF_YEAR_START

st.set_page_config(
    page_title="QRS Research · FerryRichMan",
    page_icon="🔬",
    layout="centered",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0f1e !important; }
[data-testid="stHeader"]           { background: transparent !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 860px; }
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
[data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 14px; }
h1,h2,h3 { color: #e2e8f0 !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 20px;border-bottom:1px solid #1e293b;margin-bottom:24px;">
  <div style="font-size:11px;color:#6366f1;font-weight:800;letter-spacing:3px;
    text-transform:uppercase;margin-bottom:8px;">FerryRichMan Limited · Research</div>
  <div style="font-size:28px;font-weight:900;color:#f1f5f9;letter-spacing:-1px;">
    策略比較分析
  </div>
  <div style="font-size:13px;color:#475569;margin-top:6px;">
    QRS Standard &nbsp;vs&nbsp; QRS Enhanced（+TLT/GLD）&nbsp;vs&nbsp;
    Enhanced + 趨勢濾網 &nbsp;vs&nbsp; SPY 買入持有
  </div>
  <div style="font-size:11px;color:#334155;margin-top:8px;padding:8px 12px;
    background:#111827;border-radius:8px;border:1px solid #1e293b;">
    ⚠️ 此頁僅供研究參考，不影響主系統訊號
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  ENHANCED UNIVERSE DATA LOADER
#  SPY, VEU, BIL, TLT, GLD  — monthly adj close + first open
# ══════════════════════════════════════════════════════════
ENHANCED_TICKERS = ["SPY", "VEU", "BIL", "TLT", "GLD"]


def _fetch_monthly(tickers, years=22):
    """Download daily adj OHLC for all tickers, resample to monthly."""
    end   = datetime.today()
    start = end - relativedelta(years=years)

    close_d, open_d = {}, {}
    for t in tickers:
        for attempt in range(4):
            try:
                raw = yf.download(
                    t,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True,
                )
                if not raw.empty:
                    # Handle MultiIndex (yfinance ≥ 1.x)
                    if isinstance(raw.columns, pd.MultiIndex):
                        c = raw["Close"]
                        o = raw["Open"]
                        c = c[t] if t in c.columns else c.iloc[:, 0]
                        o = o[t] if t in o.columns else o.iloc[:, 0]
                    else:
                        c = raw.get("Close", pd.Series(dtype=float))
                        o = raw.get("Open",  pd.Series(dtype=float))
                    c = c.squeeze(); o = o.squeeze()
                    close_d[t] = c.resample("ME").last()
                    open_d[t]  = o.resample("ME").first()
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)

    if not close_d:
        return None, None
    df_close = pd.DataFrame(close_d).dropna()
    df_open  = pd.DataFrame(open_d).reindex(df_close.index)
    return df_close, df_open


@st.cache_data(show_spinner=False)
def load_enhanced(date_key: str = ""):
    """Load and compute signals for the enhanced 5-asset universe."""
    close, opn = _fetch_monthly(ENHANCED_TICKERS)
    if close is None:
        return None

    df = close.copy()

    # ── Momentum scores (same weights as QRS Standard) ──
    for m in [3, 6, 9, 12]:
        target_dates = df.index - pd.DateOffset(months=m)
        for t in ENHANCED_TICKERS:
            full_idx = df.index.union(target_dates).sort_values()
            filled   = df[t].reindex(full_idx).ffill()
            prior    = filled.reindex(target_dates).values
            df[f"{t}_{m}m"] = df[t].values / prior - 1

    for t in ENHANCED_TICKERS:
        df[f"score_{t}"] = (
            df[f"{t}_3m"]  * WEIGHTS["3m"]
            + df[f"{t}_6m"]  * WEIGHTS["6m"]
            + df[f"{t}_9m"]  * WEIGHTS["9m"]
            + df[f"{t}_12m"] * WEIGHTS["12m"]
        )

    score_df = df[[f"score_{t}" for t in ENHANCED_TICKERS]].copy()
    score_df.columns = ENHANCED_TICKERS
    has_valid = score_df.notna().any(axis=1)
    filled_sc = score_df.fillna(-np.inf)
    df["signal_enh"] = pd.Series(dtype=object, index=df.index)
    df.loc[has_valid, "signal_enh"] = filled_sc.loc[has_valid].idxmax(axis=1).values

    # ── Trend filter: SPY < 12-month moving average → force BIL ──
    df["spy_ma12"] = df["SPY"].rolling(12, min_periods=12).mean()
    spy_below_ma   = df["SPY"] < df["spy_ma12"]

    df["signal_tf"] = df["signal_enh"].copy()
    df.loc[spy_below_ma & df["signal_enh"].notna(), "signal_tf"] = "BIL"

    # ── Monthly returns (first open of M+1 / first open of M - 1) ──
    for col, sig_col in [("ret_enh", "signal_enh"), ("ret_tf", "signal_tf")]:
        df[col] = np.nan
        prev = df[sig_col].shift(1)
        for t in ENHANCED_TICKERS:
            mask = prev == t
            if t in opn.columns:
                holding = opn[t].reindex(df.index).shift(-1) / opn[t].reindex(df.index) - 1
                df.loc[mask, col] = holding[mask]

    # ── SPY buy-and-hold returns ──
    df["spy_ret"] = opn["SPY"].reindex(df.index).shift(-1) / opn["SPY"].reindex(df.index) - 1

    # ── SPY 12M column for reference ──
    df["SPY_12m"] = df["SPY"] / df["SPY"].shift(12) - 1

    return df


# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════
with st.spinner("載入 5 資產數據中（SPY / VEU / BIL / TLT / GLD）…"):
    base   = load_prices(date_key=_hk_date_key())
    base   = compute_signals(base)
    enh_df = load_enhanced(date_key=_hk_date_key())

if enh_df is None:
    st.error("無法載入擴充資產數據，請稍後再試。")
    st.stop()

# SPY B&H from base (already computed)
base["spy_ret"] = base["SPY_open"].shift(-1) / base["SPY_open"] - 1


# ══════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════
def calc(ret_series, label, start_yr=PERF_YEAR_START):
    r = ret_series.dropna()
    r = r[r.index.year >= start_yr]
    if len(r) < 12:
        return None
    cum   = (1 + r).cumprod()
    total = float(cum.iloc[-1] - 1)
    n_yrs = len(r) / 12
    cagr  = float((1 + total) ** (1 / n_yrs) - 1)
    rm    = cum.cummax()
    dd    = (cum - rm) / rm
    mdd   = float(dd.min())
    mdd_e = dd.idxmin()
    mdd_s = cum[:mdd_e].idxmax()
    max_uw, cur = 0, 0
    for v in (dd < 0):
        cur = cur + 1 if v else 0
        max_uw = max(max_uw, cur)
    sharpe = float(r.mean() * np.sqrt(12) / r.std()) if r.std() > 0 else None
    calmar = cagr / abs(mdd) if mdd != 0 else None
    return {
        "label":     label,
        "cagr":      cagr,
        "mdd":       mdd,
        "mdd_start": mdd_s.strftime("%Y-%m"),
        "mdd_end":   mdd_e.strftime("%Y-%m"),
        "max_uw_mo": max_uw,
        "sharpe":    sharpe,
        "calmar":    calmar,
        "win_rate":  float((r > 0).mean()),
        "total_ret": total,
        "n_years":   n_yrs,
        "final_10k": float(cum.iloc[-1] * 10000),
        "cum":       cum * 10000,
    }


sA   = calc(base["signal_return"],  "QRS Standard")
sB   = calc(enh_df["ret_enh"],      "QRS Enhanced（+TLT/GLD）")
sC   = calc(enh_df["ret_tf"],       "Enhanced + 趨勢濾網")
sSPY = calc(base["spy_ret"],        "SPY 買入持有")

strategies = [s for s in [sA, sB, sC, sSPY] if s is not None]

COLORS = {
    "QRS Standard":              "#818cf8",
    "QRS Enhanced（+TLT/GLD）":  "#34d399",
    "Enhanced + 趨勢濾網":       "#fb923c",
    "SPY 買入持有":              "#64748b",
}


# ══════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════
def kpi_card(val, label, color="#94a3b8"):
    return (
        f'<div style="background:#111827;border:1px solid #1e293b;border-radius:12px;'
        f'padding:14px 8px;text-align:center;">'
        f'<div style="font-size:21px;font-weight:900;color:{color};letter-spacing:-0.5px;">{val}</div>'
        f'<div style="font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;margin-top:4px;">{label}</div></div>'
    )


for s in strategies:
    clr = COLORS.get(s["label"], "#94a3b8")
    st.markdown(
        f'<div style="font-size:13px;font-weight:800;color:{clr};letter-spacing:1px;'
        f'text-transform:uppercase;margin:24px 0 10px;">{s["label"]}</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, c in [
        (c1, f'{s["cagr"]*100:.1f}%',   "CAGR",       "#4ade80" if s["cagr"] >= 0 else "#f87171"),
        (c2, f'{s["mdd"]*100:.1f}%',    "最大回撤",   "#f87171"),
        (c3, f'{s["max_uw_mo"]}個月',   "最長水下",   "#fb923c"),
        (c4, f'{s["sharpe"]:.2f}' if s["sharpe"] else "—", "Sharpe", "#818cf8"),
        (c5, f'${s["final_10k"]:,.0f}', "$10k →",     clr),
    ]:
        with col:
            st.markdown(kpi_card(val, lbl, c), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  DETAILED TABLE
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">📊 詳細對比</div>',
    unsafe_allow_html=True,
)

metrics = [
    ("CAGR",         lambda s: f'{s["cagr"]*100:.2f}%'),
    ("MDD",          lambda s: f'{s["mdd"]*100:.2f}%'),
    ("MDD 期間",     lambda s: f'{s["mdd_start"]} ~ {s["mdd_end"]}'),
    ("最長水下月數", lambda s: f'{s["max_uw_mo"]} 個月'),
    ("Sharpe Ratio", lambda s: f'{s["sharpe"]:.3f}' if s["sharpe"] else "—"),
    ("Calmar Ratio", lambda s: f'{s["calmar"]:.3f}' if s["calmar"] else "—"),
    ("月勝率",       lambda s: f'{s["win_rate"]*100:.1f}%'),
    ("$10k 增長至",  lambda s: f'${s["final_10k"]:,.0f}'),
    ("回測年數",     lambda s: f'{s["n_years"]:.1f} 年'),
]

rows = []
for name, fn in metrics:
    row = {"指標": name}
    for s in strategies:
        row[s["label"]] = fn(s)
    rows.append(row)

st.dataframe(
    pd.DataFrame(rows).set_index("指標"),
    use_container_width=True,
)


# ══════════════════════════════════════════════════════════
#  GROWTH CURVE
# ══════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════
#  GROWTH CURVE
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">\U0001f4c8 \u589e\u9577\u66f2\u7dda</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()
styles = [
    ("QRS Standard",             "#818cf8", "solid", 2.5),
    ("QRS Enhanced\uff08+TLT/GLD\uff09", "#34d399", "solid", 2.0),
    ("Enhanced + \u8da8\u52e2\u6fc3\u7db2",      "#fb923c", "solid", 2.0),
    ("SPY \u8cb7\u5165\u6301\u6709",             "#64748b", "dot",   1.5),
]
for label, color, dash, width in styles:
    s = next((x for x in strategies if x["label"] == label), None)
    if s:
        fig.add_trace(go.Scatter(
            x=s["cum"].index, y=s["cum"].values,
            name=label, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate="%{x|%Y-%m}  <b>$%{y:,.0f}</b><extra>" + label + "</extra>",
        ))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=320, margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
    yaxis=dict(showgrid=True, gridcolor="#1e293b",
               tickfont=dict(color="#64748b", size=10),
               tickprefix="$", tickformat=",.0f"),
    hovermode="x unified",
    legend=dict(orientation="h", x=0, y=1.12,
                font=dict(color="#94a3b8", size=11),
                bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════
#  TREND FILTER TRIGGERED MONTHS
# ══════════════════════════════════════════════════════════
tf_mask = (
    enh_df["signal_enh"].notna()
    & (enh_df["SPY"] < enh_df["spy_ma12"])
    & enh_df["spy_ma12"].notna()
)
n_tf = int(tf_mask.sum())

st.markdown(
    f'<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">'
    f'\U0001f504 \u8da8\u52e2\u6fc3\u7db2\u89f8\u767c\u6708\u4efd\uff08\u5171 {n_tf} \u500b\u6708\uff09</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#475569;margin-bottom:10px;">'
    'SPY \u6708\u5e95\u6536\u76e4 &lt; 12 \u500b\u6708\u79fb\u52d5\u5e73\u5747\u7dda\u6642\uff0c\u5f37\u5236\u8f49\u6301 BIL</div>',
    unsafe_allow_html=True,
)

if n_tf > 0:
    tf_rows = []
    for dt in enh_df.index[tf_mask]:
        orig = enh_df.loc[dt, "signal_enh"]
        spy  = enh_df.loc[dt, "SPY"]
        ma12 = enh_df.loc[dt, "spy_ma12"]
        tf_rows.append({
            "\u6708\u4efd":       dt.strftime("%Y-%m"),
            "\u52d5\u91cf\u9078\u51fa":   orig,
            "SPY \u6536\u76e4":   f"{spy:.2f}",
            "SPY 12M MA": f"{ma12:.2f}",
            "\u5f37\u5236\u6539\u70ba":   "BIL",
        })
    st.dataframe(pd.DataFrame(tf_rows), use_container_width=True, hide_index=True)
else:
    st.info("\u76ee\u524d\u671f\u9593\u5167\u8da8\u52e2\u6fc3\u7db2\u672a\u89f8\u767c")


# ══════════════════════════════════════════════════════════
#  HOLDINGS BREAKDOWN
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">'
    '\U0001f4c2 \u5404 ETF \u6301\u5009\u6bd4\u4f8b\uff08QRS Enhanced\uff09</div>',
    unsafe_allow_html=True,
)

valid_signals = enh_df["signal_enh"].dropna()
if len(valid_signals) > 0:
    vc = valid_signals.value_counts()
    total_months = len(valid_signals)
    cols = st.columns(len(vc))
    ticker_colors = {
        "SPY": "#818cf8", "VEU": "#34d399", "BIL": "#fbbf24",
        "TLT": "#60a5fa", "GLD": "#f59e0b",
    }
    for i, (tk, cnt) in enumerate(vc.items()):
        with cols[i]:
            pct = cnt / total_months * 100
            st.markdown(
                kpi_card(f"{pct:.0f}%", tk, ticker_colors.get(tk, "#94a3b8")),
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div style="font-size:11px;color:#334155;margin-top:8px;">'
        f'\u7d71\u8a08\u671f\u9593\uff1a{valid_signals.index[0].strftime("%Y-%m")} \u2013 '
        f'{valid_signals.index[-1].strftime("%Y-%m")}\uff0c\u5171 {total_months} \u500b\u6708</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="margin-top:36px;font-size:11px;color:#334155;text-align:center;">'
    'FerryRichMan Limited \u00b7 \u7814\u7a76\u5206\u6790\u9801\u9762 \u00b7 \u4e0d\u69cb\u6210\u4efb\u4f55\u6295\u8cc7\u5efa\u8b70</div>',
    unsafe_allow_html=True,
)

