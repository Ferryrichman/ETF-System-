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
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings("ignore")

# Hong Kong timezone (UTC+8) — no external dependency
_HK_TZ = timezone(timedelta(hours=8))

def _hk_date_key() -> str:
    """Return today's date string in HK time (resets at 00:00 HKT = 16:00 UTC prev day).
    Used as cache key so data refreshes once per day before HK market open."""
    return datetime.now(_HK_TZ).strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
TICKERS = ["SPY", "VEU", "BIL"]
WEIGHTS = {"3m": 0.8, "6m": 0.0, "9m": 0.0, "12m": 0.2}
PERF_YEAR_START = 2009   # KPIs & annual chart start year (growth curve unchanged)
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
    Fetch DAILY Adjusted Open + Adjusted Close for one ticker.
    Returns DataFrame with columns ['Open', 'Close'].

    Price methodology — matches the Excel 'QRS Basic Signal' exactly:
      Close  → Adj Close  → momentum lookback (consistent with Excel 'Price data (Adj Close)')
      Open   → Adj Open   → buy/sell execution price (dividend-inclusive return)

    Uses auto_adjust=True so Yahoo Finance returns all prices already adjusted.
    Retries up to 4× with exponential back-off on transient errors.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    def _extract(raw: pd.DataFrame) -> pd.DataFrame:
        """Pull adj Open + adj Close from a yf.download(auto_adjust=True) DataFrame."""
        def _squeeze(s):
            return s.squeeze() if isinstance(s, pd.DataFrame) else s

        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance ≥ 1.x returns MultiIndex even for a single ticker
            def _col(name):
                if name in raw.columns.get_level_values(0):
                    lvl = raw[name]
                    s = lvl[t] if t in lvl.columns else lvl.iloc[:, 0]
                else:
                    s = pd.Series(dtype=float, index=raw.index)
                return _squeeze(s)
            adj_o = _col("Open")
            adj_c = _col("Close")
        else:
            # yfinance 0.2.x returns flat columns for single ticker
            adj_o = _squeeze(raw.get("Open",  pd.Series(dtype=float, index=raw.index)))
            adj_c = _squeeze(raw.get("Close", pd.Series(dtype=float, index=raw.index)))

        result = pd.DataFrame(index=raw.index)
        result["Open"]  = adj_o   # adj open  → monthly return calculation
        result["Close"] = adj_c   # adj close → momentum score calculation
        return result.sort_index()

    for attempt in range(4):
        try:
            raw = yf.download(
                t, start=start_str, end=end_str,
                progress=False, auto_adjust=True,
            )
            if not raw.empty:
                return _extract(raw)
        except Exception:
            pass

        time.sleep(2 ** attempt)

    return pd.DataFrame(columns=["Open", "Close"])


@st.cache_data(show_spinner=False)
def load_prices(date_key: str = "") -> pd.DataFrame:
    """
    Download daily adj OHLC and compute two monthly series per ETF:
      {t}       – last trading day adj CLOSE  → momentum signal (matches Excel Adj Close)
      {t}_open  – first trading day adj OPEN  → actual buy / sell price

    Monthly return = adj_open(M+1) ÷ adj_open(M) − 1

    Cache key = HK date string → refreshes once per day at 00:00 HKT.
    To force a hard reset, bump _CACHE_VERSION below.
    """
    _CACHE_VERSION = "adj_v8"   # ← bump this string to force a fresh download (ignores date_key)
    end   = datetime.now(_HK_TZ) + timedelta(days=2)
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


@st.cache_data(show_spinner=False)
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted momentum scores and monthly signals."""
    prices = df.copy()

    # ── Date-based lookback returns ──────────────────────────────────────────
    # IMPORTANT: We use pd.DateOffset (calendar months), NOT shift(m) (row count).
    # shift(m) breaks when dropna() removes rows for missing ETF data, causing
    # the 12-row lookback to land on the wrong calendar month (e.g. Dec 2018
    # instead of Feb 2019), which silently corrupts all momentum scores.
    #
    # Method: for each month-end date, compute the target date (exactly m months
    # earlier), build a forward-filled lookup series to handle any gaps, then
    # divide current price by the looked-up prior price.
    for m in [3, 6, 9, 12]:
        target_dates = prices.index - pd.DateOffset(months=m)
        for t in TICKERS:
            # Union the index with target dates, ffill to cover any missing months
            full_idx = prices.index.union(target_dates).sort_values()
            filled   = prices[t].reindex(full_idx).ffill()
            prior    = filled.reindex(target_dates).values
            prices[f"{t}_{m}m"] = prices[t].values / prior - 1

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

    # ── Monthly strategy return ───────────────────────────────────────────────
    # Signal at end of month M → hold ETF during month M+1.
    # Buy  open: first trading day of M+1  = prices["{t}_open"] at row M+1
    # Sell open: first trading day of M+2  = prices["{t}_open"] at row M+2
    # Return earned in month M+1 = open(row M+2) / open(row M+1) - 1
    #
    # shift(-1) is correct here because we only need ADJACENT rows
    # (consecutive months), not a long lookback across possible gaps.
    #
    # prev_sig[row M+1] = signal[row M] → what ETF we held in M+1
    # holding_ret[row M+1] = open(M+2)/open(M+1) - 1
    prev_sig = prices["signal"].shift(1)
    prices["signal_return"] = np.nan
    for t in TICKERS:
        mask        = prev_sig == t
        holding_ret = prices[f"{t}_open"].shift(-1) / prices[f"{t}_open"] - 1
        prices.loc[mask, "signal_return"] = holding_ret[mask]

    # ── Absolute Momentum Guard (AM Guard) ───────────────────────────────────
    # After relative momentum picks the winner, check its 12M absolute return.
    # If the winner's own 12M return < 0 (i.e. the asset itself is in a downtrend)
    # → override to BIL regardless.  BIL is never overridden (cash is always safe).
    prices["signal_am"] = prices["signal"].copy()
    for t in TICKERS:
        if t == "BIL":
            continue
        neg_abs = prices[f"{t}_12m"] < 0
        override = (prices["signal_am"] == t) & neg_abs
        prices.loc[override, "signal_am"] = "BIL"

    prev_am = prices["signal_am"].shift(1)
    prices["signal_return_am"] = np.nan
    for t in TICKERS:
        mask        = prev_am == t
        holding_ret = prices[f"{t}_open"].shift(-1) / prices[f"{t}_open"] - 1
        prices.loc[mask, "signal_return_am"] = holding_ret[mask]

    return prices


def get_current_info(prices: pd.DataFrame) -> dict:
    """Extract current month signal, previous signal, last month return, scores."""
    now = datetime.now(_HK_TZ)          # ← use HK time, not server UTC
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
    monthly_all = prices["signal_return"].dropna()

    # Full-history strategy cumulative series — used for the growth curve
    cum_all = (1 + monthly_all).cumprod() * 10000

    # SPY buy-and-hold cumulative series (same period, same $10k start)
    # Uses first-trading-day open prices, identical methodology to strategy returns
    spy_ret_all = (prices["SPY_open"].shift(-1) / prices["SPY_open"] - 1)
    spy_ret_all = spy_ret_all.reindex(monthly_all.index)   # align to strategy months
    spy_cum_all = (1 + spy_ret_all.fillna(0)).cumprod() * 10000

    # KPI series filtered from PERF_YEAR_START (e.g. 2009) onward
    monthly = monthly_all[monthly_all.index.year >= PERF_YEAR_START]
    cum      = (1 + monthly).cumprod()
    total    = float(cum.iloc[-1] - 1)
    n_yrs    = len(monthly) / 12
    cagr     = float((1 + total) ** (1 / n_yrs) - 1)

    # MDD with start/end dates (over KPI period)
    running_max = cum.cummax()
    drawdown    = (cum - running_max) / running_max
    mdd         = float(drawdown.min())
    mdd_end     = drawdown.idxmin()
    mdd_start   = cum[:mdd_end].idxmax()

    # Sharpe ratio (annualised, risk-free = 0%)
    # Formula: mean_monthly * sqrt(12) / std_monthly
    m_std = float(monthly.std())
    sharpe = float(monthly.mean() * np.sqrt(12) / m_std) if m_std > 0 else None

    # Full-history drawdown series (for underwater chart, % values)
    cum_all_norm    = (1 + monthly_all).cumprod()
    dd_series_all   = (cum_all_norm - cum_all_norm.cummax()) / cum_all_norm.cummax() * 100

    # Longest consecutive months underwater (KPI period)
    max_dd_months = 0
    cur_run = 0
    for v in (drawdown < 0):
        if v:
            cur_run += 1
            max_dd_months = max(max_dd_months, cur_run)
        else:
            cur_run = 0

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
        "cumulative":      cum_all,        # strategy full-history for growth curve
        "spy_cumulative":  spy_cum_all,    # SPY buy-and-hold for comparison
        "drawdown_series": dd_series_all,  # full-history % drawdown for underwater chart
        "max_dd_months":   max_dd_months,  # longest consecutive months underwater (KPI period)
        "sharpe":          sharpe,
        "cagr_3yr":        trailing_cagr(3),
        "cagr_5yr":        trailing_cagr(5),
    }


def calc_annual(prices: pd.DataFrame) -> dict:
    monthly = prices["signal_return"].dropna()
    monthly = monthly[monthly.index.year >= PERF_YEAR_START]
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
        f"回測CAGR：{stats['cagr']*100:.1f}% {bt_period}  |  MDD：{stats['mdd']*100:.1f}%\n"
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
    kpi_style = """
        background:#111827;
        border:1px solid #1e293b;
        border-radius:14px;
        padding:16px 12px;
        text-align:center;
    """

    # ── Row 1: Core KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:26px;font-weight:900;color:#4ade80;letter-spacing:-0.5px;">'
            f'{stats["cagr"]*100:.1f}%</div>'
            f'<div style="font-size:11px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">CAGR</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:26px;font-weight:900;color:#f87171;letter-spacing:-0.5px;">'
            f'{stats["mdd"]*100:.1f}%</div>'
            f'<div style="font-size:11px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">最大回撤</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    mdd_mo = stats.get("max_dd_months", 0)
    with c3:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:26px;font-weight:900;color:#fb923c;letter-spacing:-0.5px;">'
            f'{mdd_mo}</div>'
            f'<div style="font-size:11px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">最長回撤月數</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:26px;font-weight:900;color:#818cf8;letter-spacing:-0.5px;">'
            f'${stats["init_10k"]:,.0f}</div>'
            f'<div style="font-size:11px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">$10k 增長至</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:26px;font-weight:900;color:#fbbf24;letter-spacing:-0.5px;">'
            f'{stats["n_years"]:.1f}</div>'
            f'<div style="font-size:11px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">回測年數</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Row 2: Risk-adjusted + trailing returns ──
    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)

    sharpe = stats.get("sharpe")
    sharpe_val = f"{sharpe:.2f}" if sharpe is not None else "—"
    sharpe_clr = "#4ade80" if (sharpe is not None and sharpe >= 1.0) else \
                 "#fbbf24" if (sharpe is not None and sharpe >= 0.5) else "#f87171"
    with r1:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:{sharpe_clr};letter-spacing:-0.5px;">'
            f'{sharpe_val}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">Sharpe Ratio</div>'
            f'<div style="font-size:10px;color:#334155;margin-top:3px;">年化 · RF=0%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    c3yr = stats.get("cagr_3yr")
    c3yr_val = f"{c3yr*100:.1f}%" if c3yr is not None else "數據不足"
    c3yr_clr = "#4ade80" if (c3yr is not None and c3yr >= 0) else "#f87171"
    with r2:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:{c3yr_clr};letter-spacing:-0.5px;">'
            f'{c3yr_val}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">3年年化回報</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    c5yr = stats.get("cagr_5yr")
    c5yr_val = f"{c5yr*100:.1f}%" if c5yr is not None else "數據不足"
    c5yr_clr = "#4ade80" if (c5yr is not None and c5yr >= 0) else "#f87171"
    with r3:
        st.markdown(
            f'<div style="{kpi_style}">'
            f'<div style="font-size:28px;font-weight:900;color:{c5yr_clr};letter-spacing:-0.5px;">'
            f'{c5yr_val}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px;">5年年化回報</div>'
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
    cum      = stats["cumulative"]
    spy_cum  = stats.get("spy_cumulative")
    mdd_s    = stats.get("mdd_start")
    mdd_e    = stats.get("mdd_end")

    fig = go.Figure()

    # Shade the MDD period (drawn first so it sits behind lines)
    if mdd_s is not None and mdd_e is not None:
        fig.add_vrect(
            x0=mdd_s, x1=mdd_e,
            fillcolor="rgba(248,113,113,0.08)",
            layer="below",
            line_width=0,
            annotation_text="MDD",
            annotation_position="top left",
            annotation_font=dict(color="#f87171", size=10),
        )

    # SPY buy-and-hold baseline
    if spy_cum is not None:
        fig.add_trace(go.Scatter(
            x=spy_cum.index,
            y=spy_cum.values,
            name="SPY 買入持有",
            mode="lines",
            line=dict(color="#475569", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m}  SPY B&H: <b>$%{y:,.0f}</b><extra></extra>",
        ))

    # QRS strategy
    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum.values,
        name="QRS 策略",
        mode="lines",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.07)",
        hovertemplate="%{x|%Y-%m}  QRS: <b>$%{y:,.0f}</b><extra></extra>",
    ))

    # MDD start/end markers
    if mdd_s is not None and mdd_e is not None and mdd_s in cum.index and mdd_e in cum.index:
        fig.add_trace(go.Scatter(
            x=[mdd_s, mdd_e],
            y=[cum[mdd_s], cum[mdd_e]],
            mode="markers",
            name="MDD 區間",
            marker=dict(color="#f87171", size=8, symbol="circle"),
            hovertemplate="%{x|%Y-%m}  <b>$%{y:,.0f}</b><extra>MDD</extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            tickfont=dict(color="#64748b", size=10),
            tickprefix="$", tickformat=",.0f",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(color="#64748b", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Underwater (drawdown) chart directly below ──
    render_drawdown_chart(stats)


def render_drawdown_chart(stats: dict):
    """Underwater chart: drawdown % over full history, red fill below zero."""
    dd = stats.get("drawdown_series")
    if dd is None or dd.empty:
        return

    mdd_val = stats.get("mdd", 0) * 100   # convert to % for reference line

    fig = go.Figure()

    # Red fill area
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#f87171", width=1.2),
        fillcolor="rgba(248,113,113,0.15)",
        name="回撤深度",
        hovertemplate="%{x|%Y-%m}  <b>%{y:.1f}%</b><extra></extra>",
    ))

    # Horizontal MDD reference line
    fig.add_hline(
        y=mdd_val,
        line=dict(color="#f87171", width=1, dash="dot"),
        annotation_text=f"MDD {mdd_val:.1f}%",
        annotation_position="bottom right",
        annotation_font=dict(color="#f87171", size=10),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=150,
        margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
        yaxis=dict(
            showgrid=True, gridcolor="#1e293b",
            zeroline=True, zerolinecolor="#334155",
            tickfont=dict(color="#64748b", size=9),
            ticksuffix="%",
            autorange="reversed",   # negative values grow downward
        ),
        hovermode="x unified",
        showlegend=False,
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
        Important Disclaimer
    </div>
    <p style="font-size:11.5px;color:#475569;line-height:1.7;margin-bottom:8px;">
        ETF signals displayed are generated by FerryRichMan Limited based on a quantitative momentum model,
        for reference only and <strong style="color:#64748b;">do not constitute investment advice</strong>.
        Past performance is not indicative of future results.
    </p>
    <p style="font-size:11px;color:#374151;line-height:1.7;">
        All investments involve risk. Please consult a licensed financial advisor before making any investment decisions.
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
        &copy; {year} FerryRichMan Limited &middot; All Rights Reserved<br>
        QRS Standard Signal System &middot; Powered by Python &amp; Streamlit
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
#  DIVIDER HELPER
# ======================================================
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
def render_momentum_table(prices):
    st.markdown(
        '<div style="font-size:12px;color:#64748b;line-height:1.8;margin-bottom:16px;">'
        '\u52d5\u529b\u8a08\u7b97 &amp; \u6708\u56de\u5831\uff1a\u5747\u4f7f\u7528 <b style="color:#94a3b8;">\u8abf\u6574\u50f9\uff08Adj Close / Adj Open\uff09</b>\uff0c'
        '\u8207 Excel\u300ePrice data (Adj Close)\u300f\u65b9\u6cd5\u4e00\u81f4\u3002<br>'
        '\u300c\u8a0a\u865f\u300d= \u7576\u6708\u6708\u5e95\u8a08\u7b97\u7684\u4e0b\u6708\u6301\u5009\uff1b\u300c\u6301\u5009\u300d= \u672c\u6708\u5be6\u969b\u6301\u6709\uff08\u4e0a\u6708\u8a0a\u865f\uff09\u3002'
        '\u6700\u65b0\u6708\u4efd\u5728\u6700\u4e0a\u65b9\u3002'
        '</div>',
        unsafe_allow_html=True,
    )

    held_s = prices["signal"].shift(1)

    rows = []
    for dt in reversed(prices.index):
        r = prices.loc[dt]
        row = {}
        row["\u6708\u4efd"] = dt.strftime("%Y-%m")
        held = held_s.loc[dt]
        row["\u6301\u5009"] = str(held) if pd.notna(held) else "\u2014"
        ret = r["signal_return"]
        row["\u6708\u56de\u5831"] = f"{ret*100:+.2f}%" if pd.notna(ret) else "\u2014"
        sig = r["signal"]
        row["\u8a0a\u865f"] = str(sig) if pd.notna(sig) else "\u2014"
        for t in TICKERS:
            v = r[t]
            row[f"{t} \u6536\u5e02"] = round(float(v), 2) if pd.notna(v) else None
        for m in [12, 9, 6, 3]:
            for t in TICKERS:
                v = r.get(f"{t}_{m}m", float("nan"))
                row[f"{t} {m}M"] = f"{v*100:.2f}%" if pd.notna(v) else "\u2014"
        winner = str(sig) if pd.notna(sig) else None
        for t in TICKERS:
            v = r.get(f"score_{t}", float("nan"))
            mark = " \u2713" if (t == winner and pd.notna(v)) else ""
            row[f"{t} \u5206"] = (f"{v:.4f}{mark}" if pd.notna(v) else "\u2014")
        rows.append(row)

    df_disp = pd.DataFrame(rows)
    col_cfg = {
        "\u6708\u4efd": st.column_config.TextColumn("\u6708\u4efd", width=80),
        "\u6301\u5009": st.column_config.TextColumn("\u6301\u5009", width=60),
        "\u6708\u56de\u5831": st.column_config.TextColumn("\u6708\u56de\u5831", width=80),
        "\u8a0a\u865f": st.column_config.TextColumn("\u8a0a\u865f", width=60),
    }
    for t in TICKERS:
        col_cfg[f"{t} \u6536\u5e02"] = st.column_config.NumberColumn(f"{t} \u6536\u5e02", format="%.2f", width=72)
    for m in [12, 9, 6, 3]:
        for t in TICKERS:
            col_cfg[f"{t} {m}M"] = st.column_config.TextColumn(f"{t} {m}M", width=72)
    for t in TICKERS:
        col_cfg[f"{t} \u5206"] = st.column_config.TextColumn(f"{t} \u5206", width=90)
    st.dataframe(df_disp, use_container_width=True, height=640, hide_index=True, column_config=col_cfg)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="QRS Signal · FerryRichMan Limited",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header()

    with st.spinner("\u6b63\u5728\u7372\u53d6\u6700\u65b0\u5e02\u5834\u6578\u64da..."):
        prices = load_prices(date_key=_hk_date_key())
        prices = compute_signals(prices)

    info   = get_current_info(prices)
    stats  = calc_stats(prices)
    annual = calc_annual(prices)

    render_signal_card(info)

    section_header("\U0001f4ca", "\u672c\u6708 ETF \u52d5\u529b\u5206\u6578\u6bd4\u8f03")
    st.markdown(
        '<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:16px 16px 8px;">',
        unsafe_allow_html=True,
    )
    if info:
        render_scores_chart(info["scores"])
    st.markdown("</div>", unsafe_allow_html=True)

    section_header("📱", "WhatsApp 訊息 — 一鍵複製後轉發")
    if info:
        render_whatsapp_section(info, stats)

    section_header("📈", "歷史績效")
    render_kpis(stats)

    tab1, tab2, tab3 = st.tabs(["  年度回報  ", "  增長曲線  ", "  持倉記錄  "])
    with tab1:
        render_annual_chart(annual)
    with tab2:
        render_cumulative_chart(stats)
    with tab3:
        render_allocation_heatmap(prices)

    section_header("🔢", "動力計算明細")
    render_momentum_table(prices)

    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()
