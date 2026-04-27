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
[data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; font-size: 15px; }
h1,h2,h3 { color: #e2e8f0 !important; font-size: 22px !important; }

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
    font-size: 14px;
    padding: 8px 22px;
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
def _get_daily_ohlc(t: str, start, end) -> pd.DataFrame:
    """
    Robustly fetch DAILY Open + Close prices for one ticker.
    Returns DataFrame with columns ['Open', 'Close'].
    Tries Ticker.history first, falls back to yf.download.
    Retries with exponential back-off on transient Yahoo Finance errors.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    def _extract(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Pull Adjusted Open + Raw Close from a yfinance auto_adjust=False DataFrame.

        Price usage:
          Close  → RAW (unadjusted) close  → momentum lookback (stable, no dividend drift)
          Open   → ADJUSTED open           → buy/sell execution price (includes dividend value)

        Adjusted Open = raw_open × (adj_close / raw_close)
        If Adj Close is unavailable, falls back to raw open (ratio = 1.0).
        """
        def _squeeze(s):
            return s.squeeze() if isinstance(s, pd.DataFrame) else s

        if isinstance(raw.columns, pd.MultiIndex):
            def _col(name):
                if name in raw.columns.get_level_values(0):
                    lvl = raw[name]
                    s = lvl[t] if t in lvl.columns else lvl.iloc[:, 0]
                else:
                    s = pd.Series(dtype=float, index=raw.index)
                return _squeeze(s)
            raw_o = _col("Open")
            raw_c = _col("Close")
            adj_c = _col("Adj Close")
        else:
            raw_o = _squeeze(raw.get("Open",      pd.Series(dtype=float, index=raw.index)))
            raw_c = _squeeze(raw.get("Close",     pd.Series(dtype=float, index=raw.index)))
            adj_c = _squeeze(raw.get("Adj Close", pd.Series(dtype=float, index=raw.index)))

        # Adjustment ratio: adj_close / raw_close
        # Falls back to 1.0 when Adj Close is missing or raw_close is zero
        if not isinstance(adj_c, pd.Series) or adj_c.dropna().empty:
            adj_ratio = pd.Series(1.0, index=raw.index)
        else:
            adj_ratio = (adj_c / raw_c.replace(0, np.nan))
            adj_ratio = adj_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        result = pd.DataFrame(index=raw.index)
        result["Open"]  = _squeeze(raw_o * adj_ratio)  # adjusted open  (for returns)
        result["Close"] = raw_c                         # raw close      (for momentum)
        return result.sort_index()

    # IMPORTANT: Only use yf.download(auto_adjust=False), NOT Ticker.history().
    # Ticker.history() returns Adj Close in its "Close" column regardless of the
    # auto_adjust flag in many yfinance versions — this silently corrupts momentum.
    # yf.download(auto_adjust=False) reliably returns RAW close + Adj Close separately.
    for attempt in range(4):
        try:
            raw = yf.download(
                t, start=start_str, end=end_str,
                progress=False, auto_adjust=False, actions=False,
            )
            if not raw.empty:
                return _extract(raw)
        except Exception:
            pass

        time.sleep(2 ** attempt)

    return pd.DataFrame(columns=["Open", "Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    """
    Download daily OHLC and compute two monthly series per ETF:
      {t}       – last trading day CLOSE  → momentum signal lookback
      {t}_open  – first trading day OPEN  → actual buy / sell price

    Correct return formula (user's actual trading):
      Monthly return = first_open(M+1) ÷ first_open(M) − 1
    """
    end   = datetime.today()
    start = end - relativedelta(years=22)

    close_frames = {}   # month-end close  (signal)
    open_frames  = {}   # month-start open (trade price)

    for t in TICKERS:
        daily = _get_daily_ohlc(t, start, end)
        if daily.empty or daily["Close"].dropna().empty:
            st.error(
                f"⚠️ 無法從 Yahoo Finance 下載 **{t}** 數據。\n\n"
                "Yahoo Finance 暫時限制，請等候 1-2 分鐘後重新整理頁面（F5）。"
            )
            st.stop()

        # Last trading day CLOSE of each month → momentum signal
        close_frames[t] = daily["Close"].resample("ME").last()
        # First trading day OPEN of each month → buy/sell execution price
        open_frames[t]  = daily["Open"].resample("ME").first()

    # Main DataFrame indexed by month-end dates
    df = pd.DataFrame(close_frames).dropna()
    if df.empty:
        st.error("數據合併失敗，請重新整理頁面。")
        st.stop()

    # Attach first-trading-day open prices (same month-end index)
    for t in TICKERS:
        df[f"{t}_open"] = open_frames[t].reindex(df.index)

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
    # Use fillna(-inf) so NaN scores never win; skip rows where ALL are NaN
    score_df = prices[[f"score_{t}" for t in TICKERS]].copy()
    score_df.columns = TICKERS
    has_any_valid = score_df.notna().any(axis=1)
    score_filled  = score_df.fillna(-np.inf)
    # Use object dtype so string ticker names can be assigned
    prices["signal"] = pd.Series(dtype=object, index=prices.index)
    prices.loc[has_any_valid, "signal"] = (
        score_filled.loc[has_any_valid].idxmax(axis=1).values
    )

    # Monthly strategy return (correct formula):
    # Signal determined at end of month M → trade on first day of month M+1
    #
    # Buy:  first trading day OPEN of month M   → prices[f"{t}_open"] at row M
    # Sell: first trading day OPEN of month M+1 → prices[f"{t}_open"] at row M+1
    # Return = open(M+1) / open(M) - 1
    #
    # In the DataFrame (indexed by month-end):
    #   prices[f"{t}_open"]          at row M   = buy  price
    #   prices[f"{t}_open"].shift(-1) at row M  = sell price (next month's open)
    #
    # prev_sig at row M = signal(M-1) → tells us which ETF we bought on day 1 of M
    prev_sig = prices["signal"].shift(1)
    prices["signal_return"] = np.nan
    for t in TICKERS:
        mask = prev_sig == t
        holding_ret = prices[f"{t}_open"].shift(-1) / prices[f"{t}_open"] - 1
        prices.loc[mask, "signal_return"] = holding_ret[mask]

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

    # MDD with start/end dates
    running_max = cum.cummax()
    drawdown    = (cum - running_max) / running_max
    mdd         = float(drawdown.min())
    mdd_end     = drawdown.idxmin()
    mdd_start   = cum[:mdd_end].idxmax()

    # Trailing 3-year and 5-year annualised returns
    def trailing_cagr(years: int):
        cutoff = monthly.index[-1] - relativedelta(years=years)
        sub = monthly[monthly.index > cutoff]
        if len(sub) < 6:
            return None
        r = float((1 + sub).prod() - 1)
        n = len(sub) / 12
        return float((1 + r) ** (1 / n) - 1) if n > 0 else None

    changes = int((prices["signal"] != prices["signal"].shift(1)).sum())

    return {
        "cagr":            cagr,
        "mdd":             mdd,
        "mdd_start":       mdd_start,
        "mdd_end":         mdd_end,
        "backtest_start":  monthly.index[0],
        "backtest_end":    monthly.index[-1],
        "total_ret":       total,
        "n_years":         n_yrs,
        "init_10k":        float(cum.iloc[-1] * 10000),
        "trades_yr":       changes / n_yrs,
        "monthly":         monthly,
        "cumulative":      cum * 10000,
        "cagr_3yr":        trailing_cagr(3),
        "cagr_5yr":        trailing_cagr(5),
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
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    ">FerryRichMan Limited</div>
    <div style="
        font-size: 42px;
        font-weight: 900;
        color: #f1f5f9;
        letter-spacing: -2px;
        line-height: 1.12;
        margin-bottom: 10px;
    ">QRS Standard<br><span style="color:#6366f1;">Signal System</span></div>
    <div style="
        font-size: 15px;
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
            f'<span class="badge" style="background:#451a03;color:#fbbf24;">'
            f'🔄 由 {prev} 轉換至 {sig}</span>'
        )
    else:
        change_html = (
            f'<span class="badge" style="background:#052e16;color:#4ade80;">'
            f'✅ 維持 {sig} — 無需換倉</span>'
        )

    # Return badge
    if last_ret is not None:
        pct   = f"{last_ret*100:+.2f}%"
        r_bg  = "#052e16" if last_ret >= 0 else "#450a0a"
        r_clr = "#4ade80" if last_ret >= 0 else "#f87171"
        arrow = "↑" if last_ret >= 0 else "↓"
        ret_html = (
            f'<span class="badge" style="background:{r_bg};color:{r_clr};">'
            f'{arrow} 上月回報 {pct}</span>'
        )
    else:
        ret_html = ""

    # Use components.html so HTML always renders correctly regardless of Streamlit version
    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }}
  body {{ background:transparent; padding:0; margin:0; }}
  .card {{
    background: linear-gradient(135deg, #0f1a35 0%, {bg} 100%);
    border: 1.5px solid {clr}55;
    border-radius: 20px;
    padding: 26px 28px 22px;
    position: relative;
    overflow: hidden;
  }}
  .glow {{
    position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background: radial-gradient(circle, {clr}22 0%, transparent 70%);
    pointer-events:none;
  }}
  .top-row {{
    display:flex; justify-content:space-between;
    align-items:flex-start; flex-wrap:wrap; gap:10px;
    margin-bottom:18px;
  }}
  .label {{ font-size:12px; color:#64748b; font-weight:800;
             text-transform:uppercase; letter-spacing:2px; margin-bottom:5px; }}
  .date  {{ font-size:12px; color:#475569; }}
  .badges {{ display:flex; flex-direction:column; align-items:flex-end; gap:7px; }}
  .badge {{
    display:inline-flex; align-items:center; gap:5px;
    padding:7px 16px; border-radius:100px;
    font-size:13px; font-weight:700;
  }}
  .main-row {{ display:flex; align-items:center; gap:24px; }}
  .ticker {{
    font-size:96px; font-weight:900; color:{clr};
    letter-spacing:-5px; line-height:1;
    text-shadow: 0 0 50px {clr}55;
  }}
  .etf-name {{ font-size:16px; color:#cbd5e1; font-weight:700; margin-bottom:6px; }}
  .etf-cls  {{ font-size:14px; color:#64748b; }}
</style>
</head>
<body>
<div class="card">
  <div class="glow"></div>
  <div class="top-row">
    <div>
      <div class="label">{month_str} 訊號</div>
      <div class="date">數據截至 {data_str}</div>
    </div>
    <div class="badges">
      {change_html}
      {ret_html}
    </div>
  </div>
  <div class="main-row">
    <div class="ticker">{sig}</div>
    <div>
      <div class="etf-name">{name}</div>
      <div class="etf-cls">{cls}</div>
    </div>
  </div>
</div>
</body>
</html>
        """,
        height=230,
        scrolling=False,
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

    # Month display: e.g. "4月2026"
    month_str = f"{now.month}月{now.year}"

    # Return string — show sign only when negative so it reads naturally
    if last_ret is not None:
        ret_str = f"{last_ret*100:.2f}%" if last_ret >= 0 else f"{last_ret*100:.2f}%"
    else:
        ret_str = "N/A"

    change_line = (
        f"🔄 ETF 轉換：{prev} → {sig}"
        if changed
        else f"✅ 維持上月持倉：{sig} 不變，無需換倉"
    )

    # ── Build backtest period string ──
    bt_start = stats["backtest_start"].year if stats.get("backtest_start") is not None else "—"
    bt_end   = stats["backtest_end"].year   if stats.get("backtest_end")   is not None else "—"
    bt_period = f"（{bt_start}-{bt_end}）"

    # ── Build CAGR / MDD line ──
    mdd_start_str = stats["mdd_start"].strftime("%Y-%m") if stats.get("mdd_start") is not None else "—"
    mdd_end_str   = stats["mdd_end"].strftime("%Y-%m")   if stats.get("mdd_end")   is not None else "—"
    mdd_period    = f"（{mdd_start_str}~{mdd_end_str}）"

    c3 = stats.get("cagr_3yr")
    c5 = stats.get("cagr_5yr")
    yr3_line = f"3年年化回報：{c3*100:.1f}%" if c3 is not None else "3年年化回報：數據不足"
    yr5_line = f"5年年化回報：{c5*100:.1f}%" if c5 is not None else "5年年化回報：數據不足"

    msg = (
        f"📊【FRM Standard Signal】{month_str}\n"
        f"✅ 本月持倉：{sig}   {ETF_INFO[sig]['name']}\n"
        f"\n"
        f"{change_line}\n"
        f"📈 上月策略回報：{ret_str}\n"
        f"\n"
        f"📅 執行時間：本月第一個交易日\n"
        f"⏰ 美股開市後任何時間均可執行\n"
        f"\n"
        f"回測CAGR：{stats['cagr']*100:.1f}% {bt_period}  |  MDD：{stats['mdd']*100:.1f}% {mdd_period}\n"
        f"{yr3_line}\n"
        f"{yr5_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"@FRM Standard · FerryRichMan Limited\n"
        f"（不構成任何投資建議）"
    )

    # HTML-encode for display div and hidden textarea
    # Using hidden textarea avoids ALL JS string-escaping issues
    html_enc = (
        msg
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html_display = html_enc.replace("\n", "<br>")

    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    margin:0; padding:10px 0 4px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:transparent;
  }}
  .lbl {{
    font-size:11px; color:#64748b; font-weight:800;
    text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;
  }}
  .msg {{
    background:#060c1a;
    border:1.5px dashed #334155;
    border-radius:12px;
    padding:16px 18px;
    font-size:13px;
    color:#94a3b8;
    line-height:1.85;
    font-family:'Courier New',monospace;
    margin-bottom:14px;
    word-break:break-word;
  }}
  .btn {{
    display:block; width:100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border: none; border-radius: 12px;
    padding: 15px 0; font-size: 15px; font-weight: 700;
    cursor: pointer; letter-spacing: 0.3px;
    box-shadow: 0 4px 18px rgba(99,102,241,0.4);
    transition: opacity 0.15s, transform 0.15s;
  }}
  .btn:hover {{ opacity:0.88; transform:translateY(-1px); }}
  .btn:active {{ transform:translateY(0); }}
  .btn.ok {{
    background: linear-gradient(135deg,#059669,#047857);
    box-shadow: 0 4px 18px rgba(5,150,105,0.4);
  }}
</style>
</head>
<body>
<!-- hidden textarea: text lives in DOM, zero JS-escaping issues -->
<textarea id="rawmsg" readonly
  style="position:absolute;left:-9999px;top:0;width:1px;height:1px;opacity:0;pointer-events:none;"
>{html_enc}</textarea>

<div class="lbl">📱 WhatsApp 訊息</div>
<div class="msg">{html_display}</div>
<button class="btn" id="b" onclick="cp()">📋 一鍵複製</button>

<script>
function cp() {{
  var ta  = document.getElementById('rawmsg');
  var b   = document.getElementById('b');
  var txt = ta.value;
  function done() {{
    b.className = 'btn ok';
    b.innerHTML = '✅ 已複製！可直接貼到 WhatsApp';
    setTimeout(function(){{ b.className='btn'; b.innerHTML='📋 一鍵複製'; }}, 3000);
  }}
  function fall() {{
    ta.style.cssText = 'position:static;width:100%;height:60px;opacity:1;pointer-events:auto;margin-bottom:8px;';
    ta.select(); ta.setSelectionRange(0, 99999);
    try {{ document.execCommand('copy'); done(); }} catch(e) {{}}
    ta.style.cssText = 'position:absolute;left:-9999px;top:0;width:1px;height:1px;opacity:0;pointer-events:none;';
  }}
  if (navigator.clipboard && window.isSecureContext) {{
    navigator.clipboard.writeText(txt).then(done, fall);
  }} else {{ fall(); }}
}}
</script>
</body>
</html>
        """,
        height=490,
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
            f'<div style="font-size:28px;font-weight:900;color:#4ade80;letter-spacing:-0.5px;">'
            f'{stats["cagr"]*100:.1f}%</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">CAGR</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:#f87171;letter-spacing:-0.5px;">'
            f'{stats["mdd"]*100:.1f}%</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">最大回撤</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:#818cf8;letter-spacing:-0.5px;">'
            f'${stats["init_10k"]:,.0f}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">$10k 增長至</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:#fbbf24;letter-spacing:-0.5px;">'
            f'{stats["n_years"]:.1f}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">回測年數</div>'
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
    """Show year × month grid: ETF held + return earned that month."""

    # ETF held in month M  = signal determined at end of month M-1
    # Return earned in M   = signal_return at index M
    held   = prices["signal"].shift(1)        # what we actually held this month
    ret_s  = prices["signal_return"]          # return earned this month

    # Build lookup dicts keyed by (year, month)
    etf_map: dict = {}
    ret_map: dict = {}
    for dt in prices.index:
        y, m = dt.year, dt.month
        e = held.loc[dt]
        r = ret_s.loc[dt]
        if pd.notna(e):
            etf_map[(y, m)] = str(e)
        if pd.notna(r):
            ret_map[(y, m)] = float(r)

    # Annual returns per year
    annual_ret: dict = {}
    for (y, m), r in ret_map.items():
        annual_ret[y] = annual_ret.get(y, 0.0)
        annual_ret[y] = (1 + annual_ret[y]) * (1 + r) - 1

    color_map = {"SPY": "#818cf8", "VEU": "#34d399", "BIL": "#fbbf24"}
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    years = sorted({k[0] for k in etf_map})

    header_row = (
        "<tr><th></th>"
        + "".join(
            f'<th style="font-size:9px;color:#475569;font-weight:700;'
            f'padding:3px 3px 8px;text-align:center;">{ml}</th>'
            for ml in months_lbl
        )
        + "</tr>"
    )

    rows_html = []
    for y in years:
        cells_html = []
        for m in range(1, 13):
            etf = etf_map.get((y, m), "")
            ret = ret_map.get((y, m), None)
            bg  = color_map.get(etf, "#1a2035")

            # Monthly return text
            if ret is not None:
                pct      = f"{ret*100:+.1f}%"
                ret_clr  = "#052e16" if ret >= 0 else "#450a0a"
                txt_clr  = "#4ade80" if ret >= 0 else "#f87171"
                ret_block = (
                    f'<div style="font-size:8px;color:{txt_clr};'
                    f'background:{ret_clr};border-radius:3px;'
                    f'padding:1px 3px;margin-top:3px;line-height:1.3;">'
                    f'{pct}</div>'
                )
            else:
                ret_block = ""

            cells_html.append(
                f'<td style="background:{bg};color:#0a0f1e;font-weight:800;'
                f'font-size:9px;padding:5px 3px 4px;text-align:center;'
                f'border-radius:5px;min-width:36px;vertical-align:top;">'
                f'{etf}{ret_block}</td>'
            )
        # Year label with annual return below
        yr_ret = annual_ret.get(y)
        if yr_ret is not None:
            yr_clr  = "#4ade80" if yr_ret >= 0 else "#f87171"
            yr_sign = "+" if yr_ret >= 0 else ""
            yr_ret_html = (
                f'<div style="font-size:10px;color:{yr_clr};font-weight:700;'
                f'margin-top:3px;white-space:nowrap;">{yr_sign}{yr_ret*100:.1f}%</div>'
            )
        else:
            yr_ret_html = ""

        rows_html.append(
            f"<tr>"
            f"<td style='font-size:11px;color:#94a3b8;font-weight:700;padding-right:10px;"
            f"white-space:nowrap;vertical-align:middle;text-align:right;'>"
            f"{y}{yr_ret_html}</td>"
            + "".join(cells_html)
            + "</tr>"
        )

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
        f'<span style="width:10px;height:10px;border-radius:2px;'
        f'background:{color_map[t]};display:inline-block;"></span>'
        f'<span style="font-size:10px;color:#64748b;">{t}</span></span>'
        for t in ["SPY", "VEU", "BIL"]
    )

    html = f"""
<div style="overflow-x:auto;">
  <div style="margin-bottom:10px;">{legend}
    <span style="font-size:10px;color:#475569;margin-left:8px;">
      （每格顯示：持倉ETF + 當月回報）
    </span>
  </div>
  <table style="border-collapse:separate;border-spacing:3px;width:100%;">
    {header_row}
    {"".join(rows_html)}
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
        f'<div style="font-size:14px;color:#94a3b8;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin:24px 0 12px;">'
        f'{icon}&nbsp; {title}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  MOMENTUM CALCULATION TABLE
# ══════════════════════════════════════════════════════════
def render_momentum_table(prices: pd.DataFrame):
    """
    Show a detailed month-by-month momentum calculation table,
    mirroring the Excel 'QRS Basic Signal' CSV structure.

    Columns:
      月份 | 持倉 | 月回報 | 訊號 | SPY/VEU/BIL 收市 |
      3M/6M/9M/12M returns per ETF | Scores per ETF
    """
    st.markdown(
        '<div style="font-size:12px;color:#64748b;line-height:1.8;margin-bottom:16px;">'
        '動力計算：<b style="color:#94a3b8;">原始收市價</b>（Raw Close，同 Excel）｜'
        '月回報：<b style="color:#94a3b8;">調整開市價</b>（Adj Open = Raw Open × Adj Close / Raw Close）<br>'
        '「訊號」= 當月月底計算的下月持倉；「持倉」= 本月實際持有（上月訊號）。'
        '最新月份在最上方。'
        '</div>',
        unsafe_allow_html=True,
    )

    held_s = prices["signal"].shift(1)   # what we actually held this month

    rows = []
    for dt in reversed(prices.index):
        r = prices.loc[dt]
        row: dict = {}

        row["月份"] = dt.strftime("%Y-%m")

        held = held_s.loc[dt]
        row["持倉"] = str(held) if pd.notna(held) else "—"

        ret = r["signal_return"]
        row["月回報"] = f"{ret*100:+.2f}%" if pd.notna(ret) else "—"

        sig = r["signal"]
        row["訊號"] = str(sig) if pd.notna(sig) else "—"

        # Raw close prices (used for momentum)
        for t in TICKERS:
            v = r[t]
            row[f"{t} 收市"] = round(float(v), 2) if pd.notna(v) else None

        # Lookback returns: display order matches Excel (12M → 9M → 6M → 3M)
        for m in [12, 9, 6, 3]:
            for t in TICKERS:
                v = r.get(f"{t}_{m}m", np.nan)
                row[f"{t} {m}M"] = f"{v*100:.2f}%" if pd.notna(v) else "—"

        # Weighted scores
        winner = str(sig) if pd.notna(sig) else None
        for t in TICKERS:
            v = r.get(f"score_{t}", np.nan)
            mark = " ✓" if (t == winner and pd.notna(v)) else ""
            row[f"{t} 分"] = (f"{v:.4f}{mark}" if pd.notna(v) else "—")

        rows.append(row)

    df_disp = pd.DataFrame(rows)

    # ── Colour-highlight held/signal columns via cell_style ──
    # Streamlit dataframe doesn't support per-cell colour easily;
    # we use column_config for nicer headers and a fixed-height scrollable table.
    col_cfg: dict = {
        "月份":    st.column_config.TextColumn("月份",    width=80),
        "持倉":    st.column_config.TextColumn("持倉",    width=60),
        "月回報":  st.column_config.TextColumn("月回報",  width=80),
        "訊號":    st.column_config.TextColumn("訊號",    width=60),
    }
    for t in TICKERS:
        col_cfg[f"{t} 收市"] = st.column_config.NumberColumn(
            f"{t} 收市", format="%.2f", width=72
        )
    for m in [12, 9, 6, 3]:
        for t in TICKERS:
            col_cfg[f"{t} {m}M"] = st.column_config.TextColumn(
                f"{t} {m}M", width=72
            )
    for t in TICKERS:
        col_cfg[f"{t} 分"] = st.column_config.TextColumn(f"{t} 分", width=90)

    st.dataframe(
        df_disp,
        use_container_width=True,
        height=640,
        hide_index=True,
        column_config=col_cfg,
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

    # ── Momentum Calculation Detail ──
    section_header("🔢", "動力計算明細")
    render_momentum_table(prices)

    # ── Disclaimer + Footer ──
    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()
