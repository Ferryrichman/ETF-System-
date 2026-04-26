"""
QRS Standard Signal System
FerryRichMan Limited
──────────────────────────────────────────────────────────
Streamlit App — Monthly Momentum ETF Signal Dashboard
Deploy free at: https://streamlit.io/cloud
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QRS Signal · FerryRichMan Limited",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
TICKERS = ["SPY", "VEU", "BIL"]
WEIGHTS = {"3m": 0.8, "6m": 0.6, "9m": 0.4, "12m": 0.2}
ETF_INFO = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "class_cn": "🇺🇸 美國 S&P 500 股票市場",
        "color": "#818cf8",
        "bg": "#1e1b4b",
    },
    "VEU": {
        "name": "Vanguard FTSE All World ex-US ETF",
        "class_cn": "🌍 全球（除美）股票市場",
        "color": "#34d399",
        "bg": "#064e3b",
    },
    "BIL": {
        "name": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        "class_cn": "🛡️ 短期美國國庫券（現金避險）",
        "color": "#fbbf24",
        "bg": "#451a03",
    },
}
MONTH_CN = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]

# ══════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        """
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0a0f1e !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none; }
.block-container { padding: 0 1.5rem 5rem !important; max-width: 780px; }
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

/* ── Text ── */
[data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 13px; }
h1,h2,h3 { color: #e2e8f0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #334155;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 600;
    border-radius: 9px;
    font-size: 12px;
    padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: #334155 !important;
    color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 14px; }

/* ── TextArea (WhatsApp msg) ── */
div[data-testid="stTextArea"] textarea {
    background: #0a0f1e !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
    line-height: 1.8 !important;
    resize: none !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
div[data-testid="stTextArea"] label { display: none; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Metric delta hack ── */
[data-testid="stMetricDelta"] { font-size: 13px !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════
def _get_close_series(t: str, start, end) -> pd.Series:
    """
    Robustly fetch monthly-resampled Adj Close for one ticker.
    Handles yfinance API changes (single-ticker vs multi-level columns)
    and retries on transient Yahoo Finance errors.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    for attempt in range(4):
        try:
            # ── Method 1: Ticker.history (most reliable on cloud) ──
            tk  = yf.Ticker(t)
            raw = tk.history(start=start_str, end=end_str, auto_adjust=True)
            if not raw.empty and "Close" in raw.columns:
                return raw["Close"].resample("ME").last()
        except Exception:
            pass

        try:
            # ── Method 2: yf.download single ticker ──
            raw = yf.download(t, start=start_str, end=end_str,
                              progress=False, auto_adjust=True)
            if not raw.empty:
                # New yfinance returns MultiIndex (metric, ticker)
                if isinstance(raw.columns, pd.MultiIndex):
                    col = raw["Close"]
                    series = col[t] if t in col.columns else col.iloc[:, 0]
                elif "Close" in raw.columns:
                    series = raw["Close"]
                elif "Adj Close" in raw.columns:
                    series = raw["Adj Close"]
                else:
                    series = raw.iloc[:, 0]
                if isinstance(series, pd.DataFrame):
                    series = series.squeeze()
                return series.resample("ME").last()
        except Exception:
            pass

        time.sleep(2 ** attempt)   # exponential back-off: 1s, 2s, 4s, 8s

    return pd.Series(dtype=float)  # empty — caller handles


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    """Download monthly end-of-month Adj-Close for SPY / VEU / BIL."""
    end   = datetime.today()
    start = end - relativedelta(years=22)
    frames = {}
    for t in TICKERS:
        s = _get_close_series(t, start, end)
        if s.empty:
            st.error(
                f"⚠️ 無法從 Yahoo Finance 下載 **{t}** 數據。\n\n"
                "可能原因：Yahoo Finance 暫時限制，請等候 1-2 分鐘後重新整理頁面（F5）。"
            )
            st.stop()
        frames[t] = s
    df = pd.DataFrame(frames).dropna()
    if df.empty:
        st.error("數據合併失敗，請重新整理頁面。")
        st.stop()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted momentum scores and monthly signals."""
    prices = df.copy()

    # Lookback returns
    for m in [3, 6, 9, 12]:
        for t in TICKERS:
            prices[f"{t}_{m}m"] = prices[t] / prices[t].shift(m) - 1

    # Weighted score  (3M×0.8 + 6M×0.6 + 9M×0.4 + 12M×0.2)
    for t in TICKERS:
        prices[f"score_{t}"] = (
            prices[f"{t}_3m"] * WEIGHTS["3m"]
            + prices[f"{t}_6m"] * WEIGHTS["6m"]
            + prices[f"{t}_9m"] * WEIGHTS["9m"]
            + prices[f"{t}_12m"] * WEIGHTS["12m"]
        )

    # Signal = ETF with highest weighted score
    score_df = prices[[f"score_{t}" for t in TICKERS]].copy()
    score_df.columns = TICKERS
    prices["signal"] = score_df.idxmax(axis=1)

    # Monthly strategy return:
    # Signal at row i → hold that ETF during month i+1
    prev_sig = prices["signal"].shift(1)
    prices["signal_return"] = np.nan
    for t in TICKERS:
        mask = prev_sig == t
        prices.loc[mask, "signal_return"] = prices[t].pct_change()[mask]

    return prices


def get_current_info(prices: pd.DataFrame) -> dict:
    """Extract current month signal, previous signal, last month return, scores."""
    now = datetime.today()
    completed = prices[prices.index.month != now.month]
    if len(completed) < 2:
        return {}

    cur_row  = completed.index[-1]   # end of last complete month
    prev_row = completed.index[-2]

    cur_sig  = prices.loc[cur_row,  "signal"]
    prev_sig = prices.loc[prev_row, "signal"]
    last_ret = prices.loc[cur_row,  "signal_return"]

    scores = {t: float(prices.loc[cur_row, f"score_{t}"]) for t in TICKERS}

    return {
        "current":      cur_sig,
        "prev":         prev_sig,
        "changed":      cur_sig != prev_sig,
        "last_ret":     float(last_ret) if not np.isnan(last_ret) else None,
        "scores":       scores,
        "data_date":    cur_row,
        "now":          now,
    }


def calc_stats(prices: pd.DataFrame) -> dict:
    monthly  = prices["signal_return"].dropna()
    cum      = (1 + monthly).cumprod()
    total    = float(cum.iloc[-1] - 1)
    n_yrs    = len(monthly) / 12
    cagr     = float((1 + total) ** (1 / n_yrs) - 1)
    drawdown = (cum - cum.cummax()) / cum.cummax()
    mdd      = float(drawdown.min())
    changes  = int((prices["signal"] != prices["signal"].shift(1)).sum())

    return {
        "cagr":       cagr,
        "mdd":        mdd,
        "total_ret":  total,
        "n_years":    n_yrs,
        "init_10k":   float(cum.iloc[-1] * 10000),
        "trades_yr":  changes / n_yrs,
        "monthly":    monthly,
        "cumulative": cum * 10000,
    }


def calc_annual(prices: pd.DataFrame) -> dict:
    monthly = prices["signal_return"].dropna()
    return {
        yr: float((1 + grp).prod() - 1)
        for yr, grp in monthly.groupby(monthly.index.year)
    }


# ══════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════

def render_header():
    st.markdown(
        """
<div style="
    text-align:center;
    padding: 32px 0 28px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
">
    <div style="
        display:inline-block;
        background: linear-gradient(135deg,#6366f1,#8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    ">FerryRichMan Limited</div>
    <div style="
        font-size: 30px;
        font-weight: 900;
        color: #f1f5f9;
        letter-spacing: -1.5px;
        line-height: 1.15;
        margin-bottom: 6px;
    ">QRS Standard<br><span style="color:#6366f1;">Signal System</span></div>
    <div style="
        font-size: 12px;
        color: #475569;
        letter-spacing: 0.5px;
    ">月度動力 ETF 訊號 &nbsp;·&nbsp; Momentum ETF Monthly Signal</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(info: dict):
    if not info:
        st.error("無法獲取訊號資料，請重新整理頁面。")
        return

    sig      = info["current"]
    prev     = info["prev"]
    changed  = info["changed"]
    last_ret = info["last_ret"]
    now      = info["now"]
    data_dt  = info["data_date"]

    clr  = ETF_INFO[sig]["color"]
    bg   = ETF_INFO[sig]["bg"]
    name = ETF_INFO[sig]["name"]
    cls  = ETF_INFO[sig]["class_cn"]

    month_str = f"{now.year}年{MONTH_CN[now.month-1]}月"
    data_str  = data_dt.strftime("%Y-%m-%d")

    # Change badge
    if changed:
        change_html = (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:#451a03;color:#fbbf24;padding:5px 12px;border-radius:100px;'
            f'font-size:11px;font-weight:700;">🔄 由 {prev} 轉換至 {sig}</span>'
        )
    else:
        change_html = (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:#052e16;color:#4ade80;padding:5px 12px;border-radius:100px;'
            f'font-size:11px;font-weight:700;">✅ 維持 {sig} — 無需換倉</span>'
        )

    # Return badge
    if last_ret is not None:
        pct    = f"{last_ret*100:+.2f}%"
        r_bg   = "#052e16" if last_ret >= 0 else "#450a0a"
        r_clr  = "#4ade80" if last_ret >= 0 else "#f87171"
        arrow  = "↑" if last_ret >= 0 else "↓"
        ret_html = (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:{r_bg};color:{r_clr};padding:5px 12px;border-radius:100px;'
            f'font-size:11px;font-weight:700;">{arrow} 上月回報 {pct}</span>'
        )
    else:
        ret_html = ""

    st.markdown(
        f"""
<div style="
    background: linear-gradient(135deg, #0f1a35 0%, {bg} 100%);
    border: 1.5px solid {clr}33;
    border-radius: 20px;
    padding: 28px 28px 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
">
    <!-- glow -->
    <div style="
        position:absolute; top:-60px; right:-60px;
        width:200px; height:200px;
        background: radial-gradient(circle, {clr}18 0%, transparent 70%);
        pointer-events:none;
    "></div>

    <!-- header row -->
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:18px; flex-wrap:wrap; gap:8px;">
        <div>
            <div style="font-size:10px; color:#475569; font-weight:800; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;">
                {month_str} 訊號
            </div>
            <div style="font-size:10px; color:#334155;">
                數據截至 {data_str}
            </div>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
            {change_html}
            {ret_html}
        </div>
    </div>

    <!-- main signal -->
    <div style="display:flex; align-items:center; gap:24px;">
        <div style="
            font-size: 80px;
            font-weight: 900;
            color: {clr};
            letter-spacing: -4px;
            line-height: 1;
            text-shadow: 0 0 40px {clr}44;
        ">{sig}</div>
        <div>
            <div style="font-size:15px; color:#cbd5e1; font-weight:700; margin-bottom:5px;">{name}</div>
            <div style="font-size:12px; color:#64748b;">{cls}</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_scores_chart(scores: dict):
    tickers = list(scores.keys())
    vals    = [scores[t] for t in tickers]
    colors  = [ETF_INFO[t]["color"] for t in tickers]

    fig = go.Figure(go.Bar(
        x=vals, y=tickers,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12, family="'Courier New', monospace"),
        width=0.5,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=140,
        margin=dict(l=10, r=60, t=0, b=0),
        xaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            zeroline=True, zerolinecolor="#334155", zerolinewidth=1.5,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#e2e8f0", size=13, family="'Courier New', monospace"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_whatsapp_section(info: dict, stats: dict):
    sig      = info["current"]
    prev     = info["prev"]
    changed  = info["changed"]
    last_ret = info["last_ret"]
    now      = info["now"]

    month_en = now.strftime("%B %Y")
    ret_str  = f"{last_ret*100:+.2f}%" if last_ret is not None else "N/A"
    change_line = (
        f"🔄 ETF 轉換：{prev} → {sig}"
        if changed
        else f"✅ 維持上月持倉：{sig} 不變，無需換倉"
    )

    msg = (
        f"📊【QRS Standard Signal】{month_en}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ 本月持倉：{sig}\n"
        f"   {ETF_INFO[sig]['name']}\n"
        f"\n"
        f"{change_line}\n"
        f"📈 上月策略回報：{ret_str}\n"
        f"\n"
        f"📅 執行時間：本月第一個交易日\n"
        f"⏰ 美股開市後任何時間均可執行\n"
        f"\n"
        f"回測CAGR：{stats['cagr']*100:.1f}%  |  MDD：{stats['mdd']*100:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"@QRS Standard · FerryRichMan Limited\n"
        f"（不構成任何投資建議）"
    )

    # JS-safe escaping
    js_msg = (
        msg
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", "\\n")
    )
    html_msg = (
        msg
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    components.html(
        f"""
<style>
  body {{ margin:0; font-family:-apple-system,sans-serif; background:transparent; }}
  .lbl {{
    font-size:10px; color:#64748b; font-weight:800;
    text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;
  }}
  .msg {{
    background:#060c1a;
    border:1.5px dashed #1e293b;
    border-radius:12px;
    padding:16px 18px;
    font-size:12px;
    color:#94a3b8;
    line-height:1.8;
    font-family:'Courier New',monospace;
    margin-bottom:12px;
    white-space:pre-wrap;
  }}
  .btn {{
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 28px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.2px;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
  }}
  .btn:hover {{ transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }}
  .btn.ok {{ background: linear-gradient(135deg,#059669,#047857); box-shadow: 0 4px 15px rgba(5,150,105,0.3); }}
</style>
<div class="lbl">📱 WhatsApp 訊息</div>
<div class="msg">{html_msg}</div>
<button class="btn" id="b" onclick="cp()">📋 一鍵複製</button>
<script>
function cp() {{
  const t = `{js_msg}`;
  const b = document.getElementById('b');
  const done = () => {{
    b.className='btn ok';
    b.innerHTML='✅ 已複製！可直接貼到 WhatsApp';
    setTimeout(()=>{{b.className='btn';b.innerHTML='📋 一鍵複製';}},3000);
  }};
  if(navigator.clipboard) {{
    navigator.clipboard.writeText(t).then(done,fall);
  }} else {{ fall(); }}
  function fall() {{
    const x=document.createElement('textarea');
    x.value=t; x.style.position='fixed'; x.style.opacity='0';
    document.body.appendChild(x); x.select();
    try{{document.execCommand('copy');done();}}catch(e){{}}
    document.body.removeChild(x);
  }}
}}
</script>
        """,
        height=310,
        scrolling=False,
    )


def render_kpis(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    kpi_style = """
        background:#111827;
        border:1px solid #1e293b;
        border-radius:14px;
        padding:16px 12px;
        text-align:center;
    """
    with c1:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:22px;font-weight:900;color:#4ade80;letter-spacing:-0.5px;">'
            f'{stats["cagr"]*100:.1f}%</div>'
            f'<div style="font-size:9.5px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">CAGR</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:22px;font-weight:900;color:#f87171;letter-spacing:-0.5px;">'
            f'{stats["mdd"]*100:.1f}%</div>'
            f'<div style="font-size:9.5px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">最大回撤</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:22px;font-weight:900;color:#818cf8;letter-spacing:-0.5px;">'
            f'${stats["init_10k"]:,.0f}</div>'
            f'<div style="font-size:9.5px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">$10k 增長至</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:22px;font-weight:900;color:#fbbf24;letter-spacing:-0.5px;">'
            f'{stats["n_years"]:.1f}</div>'
            f'<div style="font-size:9.5px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">回測年數</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_annual_chart(annual: dict):
    years = sorted(annual.keys())
    vals  = [annual[y] * 100 for y in years]
    clrs  = ["#4ade80" if v >= 0 else "#f87171" for v in vals]

    fig = go.Figure(go.Bar(
        x=[str(y) for y in years],
        y=vals,
        marker_color=clrs,
        marker_line_width=0,
        text=[f"{v:+.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#64748b", size=9),
        width=0.7,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#64748b", size=10),
            tickangle=-45,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            zeroline=True, zerolinecolor="#334155",
            tickfont=dict(color="#64748b", size=9),
            ticksuffix="%",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_cumulative_chart(stats: dict):
    cum = stats["cumulative"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum.values,
        mode="lines",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.07)",
        hovertemplate="%{x|%Y-%m}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            tickfont=dict(color="#64748b", size=10),
            tickprefix="$", tickformat=",.0f",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_allocation_heatmap(prices: pd.DataFrame):
    """Show year × month allocation grid (SPY / VEU / BIL)."""
    sig = prices["signal"].dropna()
    rows = {}
    for dt, s in sig.items():
        rows.setdefault(dt.year, {})[dt.month] = s

    color_map = {"SPY": "#818cf8", "VEU": "#34d399", "BIL": "#fbbf24", "": "#1e293b"}
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years  = sorted(rows.keys())

    cells = []
    for y in years:
        row_cells = []
        for m in range(1, 13):
            s = rows.get(y, {}).get(m, "")
            row_cells.append(
                f'<td style="background:{color_map.get(s,"#1e293b")};'
                f'color:#0a0f1e;font-weight:700;font-size:9px;'
                f'padding:5px 3px;text-align:center;border-radius:4px;'
                f'min-width:32px;">{s}</td>'
            )
        cells.append(
            f"<tr><td style='font-size:10px;color:#475569;padding-right:10px;white-space:nowrap;'>{y}</td>"
            + "".join(row_cells)
            + "</tr>"
        )

    header_row = (
        "<tr><th></th>"
        + "".join(
            f'<th style="font-size:9px;color:#475569;font-weight:700;padding:3px 3px 8px;text-align:center;">{m}</th>'
            for m in months
        )
        + "</tr>"
    )

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{color_map[t]};display:inline-block;"></span>'
        f'<span style="font-size:10px;color:#64748b;">{t}</span></span>'
        for t in ["SPY", "VEU", "BIL"]
    )

    html = f"""
<div style="overflow-x:auto;">
  <div style="margin-bottom:10px;">{legend}</div>
  <table style="border-collapse:separate;border-spacing:3px;width:100%;">
    {header_row}
    {"".join(cells)}
  </table>
</div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_disclaimer():
    st.markdown(
        """
<div style="
    background: #0d1424;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 28px;
">
    <div style="font-size:10px;color:#6366f1;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">
        ⚠️ 重要聲明 / Important Disclaimer
    </div>
    <p style="font-size:11.5px;color:#475569;line-height:1.7;margin-bottom:8px;">
        本頁面所顯示之 ETF 訊號及相關資訊，由 <strong style="color:#64748b;">FerryRichMan Limited</strong>
        根據量化動力策略模型計算而來，僅供參考之用，<strong style="color:#64748b;">不構成任何投資建議、要約、邀請或招攬</strong>。
        過往績效並不代表未來表現。所有投資均涉及風險，投資者可能損失部分或全部本金。
        在作出任何投資決定前，請諮詢持牌財務顧問。
    </p>
    <p style="font-size:11px;color:#374151;line-height:1.7;">
        The ETF signals displayed on this page are generated by FerryRichMan Limited based on a quantitative momentum
        model, for reference only and <strong style="color:#4b5563;">do not constitute investment advice, an offer, or solicitation</strong>.
        Past performance is not indicative of future results. All investments involve risk.
        Please consult a licensed financial advisor before making any investment decisions.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    year = datetime.today().year
    st.markdown(
        f"""
<div style="text-align:center; padding:32px 0 12px;">
    <div style="
        display:inline-block;
        background: linear-gradient(135deg,#6366f1,#8b5cf6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-size:14px;
        font-weight:900;
        letter-spacing:-0.5px;
        margin-bottom:6px;
    ">FerryRichMan Limited</div>
    <div style="font-size:10px;color:#1e293b;">
        © {year} FerryRichMan Limited · All Rights Reserved<br>
        QRS Standard Signal System · Powered by Python &amp; Streamlit
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  DIVIDER HELPER
# ══════════════════════════════════════════════════════════
def section_header(icon: str, title: str):
    st.markdown(
        f'<div style="font-size:10px;color:#64748b;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 10px;">'
        f'{icon}&nbsp; {title}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    inject_css()
    render_header()

    # ── Load Data ──
    with st.spinner("正在獲取最新市場數據..."):
        prices = load_prices()
        prices = compute_signals(prices)

    info   = get_current_info(prices)
    stats  = calc_stats(prices)
    annual = calc_annual(prices)

    # ── Signal ──
    render_signal_card(info)

    # ── Scores ──
    section_header("📊", "本月 ETF 動力分數比較")
    st.markdown(
        '<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:16px 16px 8px;">',
        unsafe_allow_html=True,
    )
    if info:
        render_scores_chart(info["scores"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── WhatsApp ──
    section_header("📱", "WhatsApp 訊息 — 一鍵複製後轉發")
    if info:
        render_whatsapp_section(info, stats)

    # ── Performance Tabs ──
    section_header("📈", "歷史績效")
    render_kpis(stats)

    tab1, tab2, tab3 = st.tabs(["  年度回報  ", "  增長曲線  ", "  持倉記錄  "])
    with tab1:
        render_annual_chart(annual)
    with tab2:
        render_cumulative_chart(stats)
    with tab3:
        render_allocation_heatmap(prices)

    # ── Disclaimer + Footer ──
    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()
