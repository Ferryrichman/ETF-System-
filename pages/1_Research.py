"""
QRS Research — Absolute Momentum Guard Comparison
FerryRichMan Limited
────────────────────────────────────────────────────────────────────────────
Compares QRS Standard vs QRS + Absolute Momentum Guard vs SPY Buy-and-Hold.

AM Guard rule:
  After relative momentum selects the winner, check that ETF's own 12M return.
  If 12M < 0 (the asset is in absolute downtrend) -> override to BIL (cash).
  Cash (BIL) is never overridden.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app import load_prices, compute_signals, _hk_date_key, PERF_YEAR_START

st.set_page_config(
    page_title="QRS Research · FerryRichMan",
    page_icon="\U0001f52c",
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
    策略對比分析
  </div>
  <div style="font-size:13px;color:#475569;margin-top:6px;">
    QRS Standard &nbsp;vs&nbsp; QRS + 絕對動力護盾（AM Guard）&nbsp;vs&nbsp; SPY 買入持有
  </div>
  <div style="font-size:11px;color:#334155;margin-top:8px;padding:8px 12px;
    background:#111827;border-radius:8px;border:1px solid #1e293b;">
    AM Guard：動力選出 ETF 後，若該 ETF 的 12M 絕對回報 &lt; 0（自身負勢）→ 強制轉居 BIL（現金）
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  LOAD DATA  (reuses app cache — no extra download)
# ══════════════════════════════════════════════════════════
with st.spinner("載入數據中..."):
    base = load_prices(date_key=_hk_date_key())
    base = compute_signals(base)

base["spy_ret"] = base["SPY_open"].shift(-1) / base["SPY_open"] - 1


# ══════════════════════════════════════════════════════════
#  STATS CALCULATOR
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
        "final_10k": float(cum.iloc[-1] * 10000),
        "n_years":   n_yrs,
        "cum":       cum * 10000,
    }


sA   = calc(base["signal_return"],    "QRS Standard")
sB   = calc(base["signal_return_am"], "QRS + AM Guard")
sSPY = calc(base["spy_ret"],          "SPY 買入持有")

strategies = [s for s in [sA, sB, sSPY] if s is not None]

COLORS = {
    "QRS Standard":         "#818cf8",
    "QRS + AM Guard":       "#34d399",
    "SPY 買入持有": "#64748b",
}


# ══════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════
def kpi_card(val, label, color="#94a3b8"):
    return (
        f'<div style="background:#111827;border:1px solid #1e293b;border-radius:12px;'
        f'padding:14px 8px;text-align:center;">'
        f'<div style="font-size:20px;font-weight:900;color:{color};letter-spacing:-0.5px;">{val}</div>'
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
        (c1, f'{s["cagr"]*100:.1f}%',   "CAGR",    "#4ade80" if s["cagr"] >= 0 else "#f87171"),
        (c2, f'{s["mdd"]*100:.1f}%',    "MDD",     "#f87171"),
        (c3, f'{s["max_uw_mo"]}個月', "最長回撤", "#fb923c"),
        (c4, f'{s["sharpe"]:.2f}' if s["sharpe"] else "—", "Sharpe", "#818cf8"),
        (c5, f'${s["final_10k"]:,.0f}', "$10k →", clr),
    ]:
        with col:
            st.markdown(kpi_card(val, lbl, c), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  DETAILED TABLE
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
    '\U0001f4ca 詳細指標對比</div>',
    unsafe_allow_html=True,
)

metrics = [
    ("CAGR",                  lambda s: f'{s["cagr"]*100:.2f}%'),
    ("MDD",                   lambda s: f'{s["mdd"]*100:.2f}%'),
    ("MDD 期間",      lambda s: f'{s["mdd_start"]} ~ {s["mdd_end"]}'),
    ("最長回撤月數", lambda s: f'{s["max_uw_mo"]} 個月'),
    ("Sharpe Ratio",          lambda s: f'{s["sharpe"]:.3f}' if s["sharpe"] else "—"),
    ("Calmar Ratio",          lambda s: f'{s["calmar"]:.3f}' if s["calmar"] else "—"),
    ("勝率",          lambda s: f'{s["win_rate"]*100:.1f}%'),
    ("$10k 終値",     lambda s: f'${s["final_10k"]:,.0f}'),
    ("回測年數", lambda s: f'{s["n_years"]:.1f} 年'),
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
#  GROWTH CHART
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
    '\U0001f4c8 增長曲線（$10,000 起始）</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()
line_styles = {
    "QRS Standard":         ("#818cf8", "solid", 2.5),
    "QRS + AM Guard":       ("#34d399", "solid", 2.5),
    "SPY 買入持有": ("#64748b", "dot",   1.5),
}
for s in strategies:
    clr, dash, w = line_styles.get(s["label"], ("#94a3b8", "solid", 1.5))
    fig.add_trace(go.Scatter(
        x=s["cum"].index, y=s["cum"].values,
        name=s["label"], mode="lines",
        line=dict(color=clr, width=w, dash=dash),
        hovertemplate="%{x|%Y-%m}  <b>$%{y:,.0f}</b><extra>" + s["label"] + "</extra>",
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
#  AM GUARD OVERRIDE TABLE
# ══════════════════════════════════════════════════════════
am_diff = base[
    base["signal"].notna() & base["signal_am"].notna() &
    (base["signal"] != base["signal_am"])
].copy()
n_overrides = len(am_diff)

st.markdown(
    f'<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">'
    f'\U0001f6e1 AM Guard 觸發月份（共 {n_overrides} 個月）</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#475569;margin-bottom:10px;">'
    '原訊號 ETF 的 12M 絕對回報 &lt; 0，被覆蓋轉为 BIL</div>',
    unsafe_allow_html=True,
)

if n_overrides > 0:
    override_rows = []
    for dt in am_diff.index:
        orig = str(base.loc[dt, "signal"])
        col_12m = f"{orig}_12m"
        ret_12m = base.loc[dt, col_12m] if col_12m in base.columns else float("nan")
        override_rows.append({
            "月份": dt.strftime("%Y-%m"),
            "原訊號": orig,
            f"{orig} 12M 回報": f"{ret_12m*100:.1f}%" if pd.notna(ret_12m) else "—",
            "覆蓋訊號": "BIL",
        })
    st.dataframe(pd.DataFrame(override_rows), use_container_width=True, hide_index=True)
else:
    st.info("目前期間內 AM Guard 未觸發")


# ══════════════════════════════════════════════════════════
#  ANNUAL RETURNS
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div style="margin-top:32px;font-size:13px;font-weight:800;color:#94a3b8;'
    'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">'
    '\U0001f4c5 年度回報對比</div>',
    unsafe_allow_html=True,
)


def annual_rets(col):
    r = base[col].dropna()
    r = r[r.index.year >= PERF_YEAR_START]
    return {yr: float((1 + grp).prod() - 1)
            for yr, grp in r.groupby(r.index.year)}


ann_a   = annual_rets("signal_return")
ann_b   = annual_rets("signal_return_am")
ann_spy = annual_rets("spy_ret")

all_yrs = sorted(set(ann_a) | set(ann_b) | set(ann_spy))
ann_rows = []
for y in all_yrs:
    a   = ann_a.get(y)
    b   = ann_b.get(y)
    spy = ann_spy.get(y)
    ann_rows.append({
        "年份": str(y),
        "QRS Standard":   f'{a*100:+.1f}%'   if a   is not None else "—",
        "QRS + AM Guard": f'{b*100:+.1f}%'   if b   is not None else "—",
        "SPY B&H":        f'{spy*100:+.1f}%' if spy is not None else "—",
    })

st.dataframe(
    pd.DataFrame(ann_rows).set_index("年份"),
    use_container_width=True,
    height=440,
)

st.markdown(
    '<div style="margin-top:36px;font-size:11px;color:#334155;text-align:center;">'
    'FerryRichMan Limited · 研究分析頁面 · 不構成任何投資建議</div>',
    unsafe_allow_html=True,
)
