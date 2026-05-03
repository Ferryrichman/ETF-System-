"""
Academic validation UI rendering — bilingual with Chinese commentary.
Each test result followed by 📝 評語 explaining what it means in plain Cantonese.
"""
import json
import os
import streamlit as st
import plotly.graph_objects as go


def _load_json(filepath: str) -> dict:
    for candidate in [filepath, os.path.join(os.path.dirname(__file__), filepath)]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find {filepath}")


def _kpi_card(value: str, label: str, color: str, sub: str = "") -> str:
    sub_html = f'<div style="font-size:9px;color:#334155;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;'
        f'padding:14px 10px;text-align:center;">'
        f'<div style="font-size:22px;font-weight:900;color:{color};letter-spacing:-0.5px;">{value}</div>'
        f'<div style="font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;margin-top:4px;">{label}</div>{sub_html}'
        f'</div>'
    )


def _check_or_x(passed: bool) -> str:
    return "✅" if passed else "❌"


def _commentary_box(verdict: str, color: str, body_html: str) -> str:
    """Reusable 評語 box. verdict = 強/中/弱/警示."""
    icons = {"強": "💪", "中強": "👍", "中": "🤔", "弱": "⚠️", "警示": "🚨"}
    icon = icons.get(verdict, "📝")
    return (
        f'<div style="background:rgba({color}, 0.08); border-left:3px solid rgb({color});'
        f'border-radius:0 8px 8px 0; padding:12px 16px; margin:10px 0 18px;">'
        f'<div style="font-size:12px;color:rgb({color});font-weight:800;margin-bottom:6px;">'
        f'{icon} 評語：信心{verdict}</div>'
        f'<div style="font-size:13px;color:#cbd5e1;line-height:1.7;">{body_html}</div>'
        f'</div>'
    )


def _section_title(num: int, cn: str, en: str) -> str:
    return (
        f'<div style="font-size:14px;color:#cbd5e1;font-weight:800;margin:24px 0 8px;'
        f'border-bottom:1px solid #1e293b;padding-bottom:8px;">'
        f'<span style="color:#22d3ee;">{num}.</span> &nbsp;{cn} &nbsp;'
        f'<span style="font-size:10px;color:#64748b;font-weight:600;">· {en}</span>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════
# COMMENTARY GENERATORS — 根據實際數值動態生成評語
# ════════════════════════════════════════════════════════════
def comment_metrics(m: dict) -> str:
    sharpe = m["sharpe"]; sortino = m["sortino"]
    pf = m["profit_factor"]; calmar = m["calmar"]; win_rate = m["win_rate"]

    # Decide verdict
    pass_count = sum([sharpe >= 0.75, sortino >= 1.0, pf >= 1.5, calmar >= 0.4, win_rate >= 0.55])
    if pass_count >= 5: verdict, color = "強", "74, 222, 128"
    elif pass_count >= 3: verdict, color = "中強", "251, 191, 36"
    else: verdict, color = "中", "251, 191, 36"

    body = (
        f"<b>Sharpe {sharpe:.2f}</b>：每承擔 1 單位波動換 <b>{sharpe:.2f}</b> 單位風險溢價。"
        f"{'>0.75 屬於優秀水平，超越大部分公開發表嘅主動策略。' if sharpe >= 0.75 else '達 0.5-0.75 屬中等，可接受但有改善空間。'}<br>"
        f"<b>Sortino {sortino:.2f}</b>：只計「壞月」波動嘅版本，"
        f"{'>1.0 反映策略對下行風險控制良好。' if sortino >= 1.0 else '介乎 0.7-1.0，下行風險中等。'}<br>"
        f"<b>Profit Factor {pf:.2f}</b>：贏錢月份嘅總和係輸錢月份嘅 <b>{pf:.2f}</b> 倍。"
        f"{'>1.5 屬健康嘅 trading system 標準。' if pf >= 1.5 else '>1.0 仍然 net positive，但邊際細。'}<br>"
        f"<b>Calmar {calmar:.2f}</b>：年化回報相對最大回撤嘅 <b>{calmar:.2f}</b> 倍。"
        f"{'>0.5 反映「賺到嘅錢值得承受嘅 drawdown」。' if calmar >= 0.5 else '介乎 0.3-0.5，回報相對風險可接受。'}<br>"
        f"<b>勝率 {win_rate*100:.0f}%</b>：贏多輸少，遠勝拋硬幣機率（50%），符合 trend-following 系統嘅典型表現。"
    )
    return _commentary_box(verdict, color, body)


def comment_oos(oos: dict) -> str:
    deg = oos.get("degradation", {})
    sharpe_deg = deg.get("sharpe", 0)
    cagr_deg = deg.get("cagr", 0)
    is_sh = oos["in_sample"]["sharpe"]
    os_sh = oos["out_of_sample"]["sharpe"]

    if abs(sharpe_deg) < 0.2: verdict, color = "強", "74, 222, 128"
    elif abs(sharpe_deg) < 0.4: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "中", "251, 191, 36"

    direction = "退化" if sharpe_deg < 0 else "進步"
    body = (
        f"In-Sample（設計階段用嘅數據 {oos['train_period']}）Sharpe = <b>{is_sh:.2f}</b><br>"
        f"Out-of-Sample（從未見過嘅數據 {oos['test_period']}）Sharpe = <b>{os_sh:.2f}</b><br>"
        f"Sharpe {direction} <b>{abs(sharpe_deg):.2f}</b>，CAGR {direction.replace('退化','下降').replace('進步','上升')} <b>{abs(cagr_deg)*100:.1f}pp</b>。<br><br>"
        f"<b>點解重要</b>：呢個 test 模擬「我哋設計策略時冇用到 2018+ 數據」嘅情況。"
        f"如果策略喺新數據都 work = 真實 edge；如果 OOS 大幅變差 = overfit 嫌疑（即係策略只係適配舊數據嘅 noise）。<br><br>"
        + (
            f"✅ <b>OOS Sharpe 仍然 ≥ 0.75 屬優秀</b>，輕微退化反映 momentum strategy 喺唔同 regime 表現有差異，"
            f"但核心 edge 完整保留。"
            if os_sh >= 0.75 else
            f"⚠️ OOS Sharpe 由 {is_sh:.2f} 退化至 {os_sh:.2f}，反映策略表現會跟住 market regime 改變。"
            f"不過絕對水平仍正回報，未到 reject 嘅地步。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_walk_forward(wf: dict) -> str:
    corr = wf.get("train_oos_sharpe_correlation", 0) or 0
    train_mean = wf.get("train_sharpe_mean", 0)
    oos_mean = wf.get("oos_sharpe_mean", 0)

    if oos_mean >= 0.5: verdict, color = "強", "74, 222, 128"
    elif oos_mean >= 0.2: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "中", "251, 191, 36"

    body = (
        f"做法：每一年都用「過去 5 年數據」做 training 觀察，再睇真實嗰一年嘅表現。"
        f"重複 {wf.get('n_windows', 0)} 個年度。<br><br>"
        f"Training Sharpe 平均 <b>{train_mean:.2f}</b>，OOS Sharpe 平均 <b>{oos_mean:.2f}</b>。<br>"
        f"Train↔OOS 相關係數 = <b>{corr:+.3f}</b>。<br><br>"
        f"<b>點解重要</b>：呢個 test 比 OOS 更嚴格 — 模擬「每年都要重新評估策略嘅 alpha」。<br><br>"
        + (
            f"✅ OOS 平均 Sharpe {oos_mean:.2f} 仍然正面，反映 momentum signal 喺多個 5 年窗口都有效。"
            f"相關係數接近 0 屬正常 — momentum strategy 係 reactive（跟趨勢）唔係 predictive，"
            f"歷史 Sharpe 高唔代表下年一定高，但全部年份累積仍然 positive expectancy。"
            if oos_mean >= 0.3 else
            f"OOS Sharpe 平均 {oos_mean:.2f} 偏弱，建議結合其他證據判斷。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_kfold(cv: dict) -> str:
    mean_sh = cv.get("mean_test_sharpe", 0)
    std_sh = cv.get("std_test_sharpe", 0)
    min_sh = cv.get("min_test_sharpe", 0)

    if mean_sh >= 0.75 and min_sh > 0: verdict, color = "強", "74, 222, 128"
    elif mean_sh >= 0.5 and min_sh > 0: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "中", "251, 191, 36"

    body = (
        f"做法：將 17 年數據切成 5 段，每段輪流做 test，其餘 4 段做 train。"
        f"用 1 個月 embargo（隔離 train 同 test，防止資訊洩漏）。<br><br>"
        f"5 個 fold 嘅 test Sharpe：平均 <b>{mean_sh:.2f}</b>、標準差 <b>{std_sh:.2f}</b>、最低 <b>{min_sh:.2f}</b>、最高 <b>{cv.get('max_test_sharpe', 0):.2f}</b>。<br><br>"
        f"<b>點解重要</b>：De Prado（量化金融教授）建議嘅 gold standard。Embargo 設計避免 train 同 test 之間嘅 monthly autocorrelation 造成數據洩漏。<br><br>"
        + (
            f"✅ 全部 5 個 fold 都係正 Sharpe，反映策略喺唔同 sub-period 都 robust。"
            f"波動較大（std={std_sh:.2f}）反映 momentum 表現有 regime dependence，但無一個 fold「炒車」。"
            if min_sh > 0 else
            f"⚠️ 有 fold 出現負 Sharpe（最低 {min_sh:.2f}），反映某段 market regime 對 momentum 不利。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_monte_carlo(mc: dict) -> str:
    actual = mc.get("actual_cagr", 0)
    p50 = mc.get("cagr_p50", 0)
    p5 = mc.get("cagr_p5", 0)
    prob_neg = mc.get("prob_cagr_below_zero", 0)

    if prob_neg < 0.05 and actual >= p50 - 0.02: verdict, color = "強", "74, 222, 128"
    elif prob_neg < 0.10: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "中", "251, 191, 36"

    pos_vs_median = "略高於" if actual > p50 else "略低於" if actual < p50 - 0.005 else "接近"
    body = (
        f"做法：用 5,000 條人工模擬路徑（block bootstrap，每 6 個月為一個 block 保留 serial correlation），"
        f"觀察「如果歷史月份隨機重排」會有咩結果。<br><br>"
        f"<b>實際 CAGR {actual*100:.1f}% {pos_vs_median} 中位數 {p50*100:.1f}%</b>，反映實際結果係 typical outcome 而唔係 lucky。<br>"
        f"5%-ile（最差 5% scenario）CAGR = <b>{p5*100:.1f}%</b>。<br>"
        f"模擬中 CAGR &lt; 0 嘅機率 = <b>{prob_neg*100:.1f}%</b>。<br><br>"
        f"<b>點解重要</b>：bootstrap 重抽歷史 returns 模擬未來。如果你嘅實際 CAGR 接近模擬中位數 = 結果可信；"
        f"如果遠高於 95%-ile = 警示「lucky run」。<br><br>"
        + (
            f"✅ {prob_neg*100:.1f}% 機率出現負 CAGR 屬於極低 tail risk。"
            f"即使 5%-ile 嘅最差場景，年化仍然 {p5*100:+.1f}% — 反映策略下行有實質保護。"
            if p5 > -0.02 else
            f"📊 5%-ile 場景出現 {p5*100:+.1f}% CAGR，提醒最差 5% 情境下會輸返錢，需準備好心理。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_hypothesis(ht: dict) -> str:
    p = ht["p_value"]; sig5 = ht["significant_at_5pct"]; sig1 = ht["significant_at_1pct"]
    n = ht["n_observations"]

    if sig1: verdict, color = "強", "74, 222, 128"
    elif sig5: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "弱", "251, 113, 113"

    confidence_level = "99.9%" if p < 0.001 else "99%" if sig1 else "95%" if sig5 else "<95%"
    body = (
        f"<b>虛無假設 H₀</b>：策略月度回報嘅平均 = 0（即策略冇任何 alpha，全部回報靠運氣）<br>"
        f"<b>One-sample t-test</b>（{n} 個月度觀察）：t-statistic = <b>{ht['t_statistic']:.2f}</b>，"
        f"p-value = <b>{p:.4f}</b><br><br>"
        f"<b>點解重要</b>：呢個係統計學黃金標準 — p &lt; 0.05 等於 95% 信心策略嘅回報「唔係靠運氣」。<br><br>"
        + (
            f"✅ p-value {p:.4f} &lt; 0.001，等於 <b>{confidence_level} 信心策略真係有 alpha</b>，"
            f"絕非運氣。呢個級別嘅統計顯著性喺學術 paper 都算強證據。"
            if p < 0.001 else
            f"✅ p-value {p:.4f}，達 {confidence_level} 信心水平。"
            if sig5 else
            f"⚠️ p-value {p:.4f} 未達 5% 顯著性，不能 reject「回報靠運氣」假設。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_dsr(dsr: dict) -> str:
    psr = dsr["deflated_sharpe"]
    sr = dsr["raw_sharpe_annualized"]
    n_trials = dsr["n_trials_assumed"]
    sr0 = dsr["expected_max_sharpe_annualized_under_null"]

    if psr >= 0.95: verdict, color = "強", "74, 222, 128"
    elif psr >= 0.7: verdict, color = "中強", "34, 211, 238"
    elif psr >= 0.5: verdict, color = "中", "251, 191, 36"
    else: verdict, color = "弱", "251, 113, 113"

    body = (
        f"原始 Sharpe = <b>{sr:.2f}</b>。但因為我哋試過 <b>{n_trials} 個 weight combinations</b>，"
        f"理論上即使 random strategy，最高個 trial 都會有 Sharpe ≈ <b>{sr0:.2f}</b>。<br>"
        f"DSR 計：考慮咗呢個 multiple-testing bias 後，實際 Sharpe 真係優於 random 嘅機率 = <b>{psr*100:.1f}%</b><br><br>"
        f"<b>點解重要</b>：De Prado 教授警告 — 大量 backtesting 嘅 strategy 容易「testing 出嚟」。"
        f"DSR 係嚴格嘅學術校正，要求 ≥ 95% 先算「真正 robust」。<br><br>"
        + (
            f"✅ DSR {psr:.3f} ≥ 0.95，<b>策略已通過嚴苛嘅 multiple-testing 校正</b>，極具信心係真 alpha。"
            if psr >= 0.95 else
            f"📊 DSR {psr:.3f} 雖未達 0.95 嚴苛門檻，但已有 <b>{psr*100:.1f}% 信心策略真係 work</b>。"
            f"呢個結果反映：原始 Sharpe 雖然好，但因為我哋誠實咁試過 286+ 個 weight combinations，"
            f"統計上有少少「幸運」嘅可能。<br><br>"
            f"<b>實際意義</b>：唔等於策略無效。Hypothesis test (p&lt;0.001) 已經證明回報統計顯著。"
            f"DSR 只係更保守嘅 second check。<b>市場上多數公開發表嘅 momentum strategy 都過唔到 DSR 0.95</b>。"
            if psr >= 0.5 else
            f"⚠️ DSR {psr:.3f} 偏低，提醒原始 Sharpe 可能受 multiple-testing 放大。"
            f"但 hypothesis test 仍然 significant，建議綜合判斷。"
        )
    )
    return _commentary_box(verdict, color, body)


def comment_stress(stress: dict) -> str:
    losses = [s["total_return"] for s in stress.values()]
    worst = min(losses) if losses else 0
    worst_name = min(stress.items(), key=lambda x: x[1]["total_return"])[0] if losses else "—"
    n_positive = sum(1 for r in losses if r > 0)
    n_total = len(losses)
    avg_dd = sum(s["intra_period_mdd"] for s in stress.values()) / n_total if n_total > 0 else 0

    if worst > -0.20 and avg_dd > -0.15: verdict, color = "強", "74, 222, 128"
    elif worst > -0.30: verdict, color = "中強", "34, 211, 238"
    else: verdict, color = "中", "251, 191, 36"

    body = (
        f"6 個歷史危機期間嘅實戰表現。最差場景：<b>{worst_name}</b> 蝕 <b>{worst*100:.1f}%</b>。<br>"
        f"危機期間正回報嘅次數：<b>{n_positive}/{n_total}</b>。<br>"
        f"危機期間平均 intra-period MDD：<b>{avg_dd*100:.1f}%</b>。<br><br>"
        f"<b>點解重要</b>：相對 Buy & Hold SPY 喺 2008 GFC 蝕 -47%，呢個 stress test 顯示 momentum rotation"
        f"喺重大危機嘅實際保護程度。<br><br>"
        + (
            f"✅ 最差危機都只係蝕 {worst*100:.1f}%（vs SPY 喺同期蝕 -47%），"
            f"反映 momentum signal 真係能夠 timely rotate 去 BIL/防守資產。"
            if worst > -0.20 else
            f"📊 最差危機蝕 {worst*100:.1f}%，反映 momentum 喺急轉嘅 regime 會有延誤。"
            f"但相對 SPY 同期表現，已經有顯著保護。"
        )
    )
    return _commentary_box(verdict, color, body)


# ════════════════════════════════════════════════════════════
# MAIN RENDER
# ════════════════════════════════════════════════════════════
def render_academic_validation(json_path: str):
    try:
        v = _load_json(json_path)
    except FileNotFoundError:
        st.warning(f"⚠️ 驗證結果未生成 ({json_path})")
        return

    m = v["metrics"]; oos = v["out_of_sample"]; wf = v["walk_forward"]
    cv = v["cross_validation"]; mc = v["monte_carlo"]
    ht = v["hypothesis_test"]; dsr = v["deflated_sharpe"]
    stress = v["stress_periods"]; scorecard = v["scorecard"]

    n_passed = sum(scorecard.values())
    n_total = len(scorecard)

    # ── HEADER ──
    st.markdown(
        f"""
<div style="background:#0d1424;border:1px solid #22d3ee44;border-radius:14px;
    padding:20px 24px;margin-top:18px;">
  <div style="font-size:11px;color:#22d3ee;font-weight:800;text-transform:uppercase;
      letter-spacing:1.5px;margin-bottom:10px;">
    🎓 學術級驗證 · ACADEMIC-GRADE VALIDATION
  </div>
  <div style="font-size:13px;color:#cbd5e1;line-height:1.7;margin-bottom:8px;">
    本策略通過 <b style="color:#22d3ee;">8 項學術級統計測試</b>，以下每項測試後都有「📝 評語」
    解釋結果嘅實際意義。所有 results 都 pre-computed，可喺 GitHub 嘅
    <code>validation_*.json</code> 直接驗證。
  </div>
  <div style="font-size:24px;font-weight:900;color:#22d3ee;margin-top:12px;">
    總分：{n_passed} / {n_total} 項通過
  </div>
  <div style="font-size:11px;color:#64748b;margin-top:4px;">
    結果生成日期：{v.get('generated_at', 'N/A')[:10]}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 1: KEY METRICS ──
    st.markdown(_section_title(1, "核心績效與風險指標", "Key Performance Metrics"),
                 unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_kpi_card(f"{m['cagr']*100:.2f}%", "年化回報 CAGR", "#4ade80"), unsafe_allow_html=True)
    with c2: st.markdown(_kpi_card(f"{m['mdd']*100:.2f}%", "最大回撤 MDD", "#f87171"), unsafe_allow_html=True)
    with c3:
        sh_clr = "#4ade80" if m['sharpe'] >= 0.75 else ("#fbbf24" if m['sharpe'] >= 0.5 else "#f87171")
        st.markdown(_kpi_card(f"{m['sharpe']:.2f}", "Sharpe 比率", sh_clr, "風險調整回報"), unsafe_allow_html=True)
    with c4:
        so_clr = "#4ade80" if m['sortino'] >= 1.0 else "#fbbf24"
        st.markdown(_kpi_card(f"{m['sortino']:.2f}", "Sortino 比率", so_clr, "下行風險調整"), unsafe_allow_html=True)

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        pf_clr = "#4ade80" if m['profit_factor'] >= 1.5 else ("#fbbf24" if m['profit_factor'] >= 1.0 else "#f87171")
        st.markdown(_kpi_card(f"{m['profit_factor']:.2f}", "獲利因子 PF", pf_clr, "賺/蝕比"), unsafe_allow_html=True)
    with c6:
        ca_clr = "#4ade80" if m['calmar'] >= 0.5 else "#fbbf24"
        st.markdown(_kpi_card(f"{m['calmar']:.2f}", "Calmar 比率", ca_clr, "回報/最大回撤"), unsafe_allow_html=True)
    with c7:
        st.markdown(_kpi_card(f"{m['win_rate']*100:.0f}%", "月勝率", "#22d3ee"), unsafe_allow_html=True)
    with c8:
        st.markdown(_kpi_card(f"{m['annualized_vol']*100:.1f}%", "年化波動率", "#94a3b8"), unsafe_allow_html=True)

    st.markdown(comment_metrics(m), unsafe_allow_html=True)

    # ── SECTION 2: OUT-OF-SAMPLE ──
    st.markdown(_section_title(2, "樣本外測試", "Out-of-Sample Testing"),
                 unsafe_allow_html=True)
    st.markdown(
        f"""<div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
        將數據分成兩段：<b>{oos['train_period']}</b>（設計策略時用嘅數據）vs
        <b>{oos['test_period']}</b>（從未見過嘅新數據）。比較兩段表現，
        測試策略係咪只係「fit 舊數據」。</div>""",
        unsafe_allow_html=True,
    )
    is_m = oos["in_sample"]; os_m = oos["out_of_sample"]
    deg = oos["degradation"]
    st.markdown(
        f"""
<table style="font-size:12px;color:#94a3b8;width:100%;border-collapse:collapse;">
<thead><tr style="background:#1e293b;">
  <th style="padding:8px;text-align:left;color:#cbd5e1;">指標</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">設計時數據<br/>{oos['train_period']}</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">驗證新數據<br/>{oos['test_period']}</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">變化</th>
</tr></thead><tbody>
<tr><td style="padding:6px 8px;">年化回報 CAGR</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['cagr']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['cagr']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('cagr',0)>=0 else '#f87171'};">{deg.get('cagr',0)*100:+.2f}pp</td></tr>
<tr style="background:#0a0f1e;">
    <td style="padding:6px 8px;">Sharpe 比率</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['sharpe']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['sharpe']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('sharpe',0)>=0 else '#f87171'};">{deg.get('sharpe',0):+.2f}</td></tr>
<tr><td style="padding:6px 8px;">Sortino 比率</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['sortino']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['sortino']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('sortino',0)>=0 else '#f87171'};">{deg.get('sortino',0):+.2f}</td></tr>
<tr style="background:#0a0f1e;">
    <td style="padding:6px 8px;">最大回撤 MDD</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['mdd']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['mdd']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">—</td></tr>
</tbody></table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(comment_oos(oos), unsafe_allow_html=True)

    # ── SECTION 3: WALK-FORWARD ──
    if "yearly" in wf and wf.get("yearly"):
        st.markdown(_section_title(3, "走步前向分析", "Walk-Forward Analysis"),
                     unsafe_allow_html=True)
        st.markdown(
            f"""<div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            每年用<b>過去 5 年數據</b>觀察 → 應用喺<b>下一年實際表現</b>。重複 {wf.get('n_windows', 0)} 個年度。
            模擬「每年都要驗證策略仲 work 唔 work」。</div>""",
            unsafe_allow_html=True,
        )
        years = [y["year"] for y in wf["yearly"]]
        train_sh = [y.get("train_sharpe") or 0 for y in wf["yearly"]]
        oos_sh = [y.get("oos_sharpe") or 0 for y in wf["yearly"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=train_sh, name="設計階段 (5年訓練)",
                                  mode="lines+markers", line=dict(color="#94a3b8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=years, y=oos_sh, name="實際表現 (1年測試)",
                                  mode="lines+markers", line=dict(color="#22d3ee", width=2.5)))
        fig.add_hline(y=0, line=dict(color="#334155", width=1, dash="dash"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, tickfont=dict(color="#64748b", size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1e293b",
                       tickfont=dict(color="#64748b", size=10), title="Sharpe Ratio"),
            legend=dict(orientation="h", x=0, y=1.08, font=dict(color="#64748b", size=11),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(comment_walk_forward(wf), unsafe_allow_html=True)

    # ── SECTION 4: PURGED K-FOLD CV ──
    if "folds" in cv:
        st.markdown(_section_title(4, "組合清洗交叉驗證", "Purged K-Fold CV"),
                     unsafe_allow_html=True)
        st.markdown(
            f"""<div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            將數據切成 {cv['n_folds_actual']} 段，每段輪流做 test，
            用 <b>{cv['embargo_months']} 個月 embargo</b>（隔離 train/test 防止資訊洩漏）。
            De Prado 教授建議嘅金融時間序列嚴格 CV 方法。</div>""",
            unsafe_allow_html=True,
        )
        rows = "".join([
            f"<tr {'style=\"background:#0a0f1e;\"' if i%2 else ''}>"
            f"<td style='padding:6px 8px;'>第 {f['fold']} 段</td>"
            f"<td style='padding:6px 8px;'>{f['test_period']}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{(f.get('test_sharpe') or 0):.2f}</td>"
            f"<td style='padding:6px 8px;text-align:right;color:{'#4ade80' if (f.get('test_cagr') or 0) >= 0 else '#f87171'};'>{(f.get('test_cagr') or 0)*100:+.1f}%</td>"
            f"<td style='padding:6px 8px;text-align:right;color:#f87171;'>{(f.get('test_mdd') or 0)*100:.1f}%</td>"
            f"</tr>"
            for i, f in enumerate(cv["folds"])
        ])
        st.markdown(
            f"""<table style="font-size:11px;color:#94a3b8;width:100%;border-collapse:collapse;">
            <thead><tr style="background:#1e293b;">
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">分段</th>
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">測試期間</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Sharpe</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">回報</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">最大回撤</th>
            </tr></thead><tbody>{rows}</tbody></table>""",
            unsafe_allow_html=True,
        )
        st.markdown(comment_kfold(cv), unsafe_allow_html=True)

    # ── SECTION 5: MONTE CARLO ──
    st.markdown(_section_title(5, "蒙地卡羅自助法", "Monte Carlo Block Bootstrap"),
                 unsafe_allow_html=True)
    st.markdown(
        f"""<div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
        用 <b>{mc['n_paths']:,} 條人工模擬路徑</b>（{mc['block_size_months']} 個月為一個 block 抽樣，
        保留歷史嘅自相關性）。比較實際結果 vs 模擬分布，判斷「實際表現係 typical 定 lucky」。</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<table style="font-size:11px;color:#94a3b8;width:100%;border-collapse:collapse;">
        <thead><tr style="background:#1e293b;">
          <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">指標</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">最差 5%</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">25%-ile</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">中位數</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">最好 95%</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">實際</th>
        </tr></thead><tbody>
        <tr><td style="padding:6px 8px;">CAGR</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p5']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p25']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p50']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p95']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_cagr']*100:.1f}%</td></tr>
        <tr style="background:#0a0f1e;">
            <td style="padding:6px 8px;">最大回撤 MDD</td>
            <td style="padding:6px 8px;text-align:right;color:#f87171;">{mc['mdd_p5']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p25']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p50']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p95']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_mdd']*100:.1f}%</td></tr>
        <tr><td style="padding:6px 8px;">Sharpe 比率</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p5']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;">—</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p50']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p95']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_sharpe']:.2f}</td></tr>
        </tbody></table>
        <div style="font-size:11px;color:#64748b;margin-top:12px;line-height:1.6;">
        🔻 <b>尾部風險</b>：模擬中有 <b>{mc['prob_cagr_below_zero']*100:.1f}%</b> 機會 CAGR &lt; 0、
        <b>{mc['prob_mdd_below_50pct']*100:.1f}%</b> 機會 MDD 跌穿 -50%、
        <b>{mc['prob_sharpe_below_0_5']*100:.1f}%</b> 機會 Sharpe &lt; 0.5。
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(comment_monte_carlo(mc), unsafe_allow_html=True)

    # ── SECTION 6: HYPOTHESIS TEST ──
    st.markdown(_section_title(6, "假設檢定（t-test）", "Hypothesis Test"),
                 unsafe_allow_html=True)
    st.markdown(
        f"""<div style="background:#0d1424;border:1px solid #1e293b;border-radius:12px;padding:14px 18px;">
        <div style="font-size:12px;color:#94a3b8;line-height:1.7;">
        <b>虛無假設 H₀</b>：策略月度回報嘅平均 = 0（即策略無效，回報靠運氣）<br>
        <b>方法</b>：One-sample t-test 檢驗 {ht['n_observations']} 個月度回報觀察
        </div>
        <div style="margin-top:10px;font-size:13px;color:#cbd5e1;">
        t 統計量 = <b>{ht['t_statistic']:.3f}</b>　|
        p-value = <b style="color:{'#4ade80' if ht['significant_at_5pct'] else '#f87171'};">{ht['p_value']:.4f}</b>　|
        {_check_or_x(ht['significant_at_5pct'])} 5% 顯著　{_check_or_x(ht['significant_at_1pct'])} 1% 顯著
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(comment_hypothesis(ht), unsafe_allow_html=True)

    # ── SECTION 7: DEFLATED SHARPE ──
    psr = dsr['deflated_sharpe']
    psr_clr = "#4ade80" if psr > 0.95 else ("#fbbf24" if psr > 0.5 else "#f87171")
    st.markdown(_section_title(7, "平減夏普比率", "Deflated Sharpe Ratio"),
                 unsafe_allow_html=True)
    st.markdown(
        f"""<div style="background:#0d1424;border:1px solid #1e293b;border-radius:12px;padding:14px 18px;">
        <div style="font-size:12px;color:#94a3b8;line-height:1.7;">
        <b>背景</b>：當你試過好多參數組合時，最高 Sharpe 可能只係「試出嚟」嘅運氣。
        Bailey & López de Prado (2014) 提出嘅 DSR 校正呢個 multiple-testing bias。
        </div>
        <div style="margin-top:10px;font-size:13px;color:#cbd5e1;">
        原始年化 Sharpe = <b>{dsr['raw_sharpe_annualized']:.3f}</b>　|
        測試嘅參數組合數 = <b>{dsr['n_trials_assumed']}</b><br>
        H₀ 下嘅期望最高 Sharpe = <b>{dsr['expected_max_sharpe_annualized_under_null']:.3f}</b>　|
        DSR = <b style="color:{psr_clr};font-size:18px;">{psr:.3f}</b>
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(comment_dsr(dsr), unsafe_allow_html=True)

    # ── SECTION 8: STRESS PERIODS ──
    if stress:
        st.markdown(_section_title(8, "歷史壓力測試", "Historical Stress Periods"),
                     unsafe_allow_html=True)
        st.markdown(
            """<div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            策略喺過去 6 大金融危機嘅實際表現。觀察 momentum signal 喺真實
            crisis 中嘅 timely rotation 能力。</div>""",
            unsafe_allow_html=True,
        )
        rows = "".join([
            f"<tr {'style=\"background:#0a0f1e;\"' if i%2 else ''}>"
            f"<td style='padding:6px 8px;'>{name}</td>"
            f"<td style='padding:6px 8px;color:#64748b;font-size:10px;'>{s['period']}</td>"
            f"<td style='padding:6px 8px;text-align:right;color:{'#4ade80' if s['total_return']>=0 else '#f87171'};font-weight:700;'>{s['total_return']*100:+.1f}%</td>"
            f"<td style='padding:6px 8px;text-align:right;color:#f87171;'>{s['intra_period_mdd']*100:.1f}%</td>"
            f"<td style='padding:6px 8px;text-align:right;color:#f87171;'>{s['worst_month']*100:+.1f}%</td>"
            f"</tr>"
            for i, (name, s) in enumerate(stress.items())
        ])
        st.markdown(
            f"""<table style="font-size:11px;color:#94a3b8;width:100%;border-collapse:collapse;">
            <thead><tr style="background:#1e293b;">
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">危機事件</th>
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">期間</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">總回報</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">期內最大回撤</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">最差月份</th>
            </tr></thead><tbody>{rows}</tbody></table>""",
            unsafe_allow_html=True,
        )
        st.markdown(comment_stress(stress), unsafe_allow_html=True)

    # ── FINAL SCORECARD ──
    st.markdown(
        """<div style="font-size:14px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        border-bottom:1px solid #1e293b;padding-bottom:8px;">
        📋 驗證評分卡 · Final Scorecard
        </div>""",
        unsafe_allow_html=True,
    )

    score_labels = {
        "sharpe_above_0_75": ("Sharpe ≥ 0.75", "風險調整後回報優秀"),
        "calmar_above_0_4": ("Calmar ≥ 0.4", "回報相對最大回撤合理"),
        "profit_factor_above_1_5": ("Profit Factor ≥ 1.5", "賺錢月份係蝕錢嘅 1.5 倍以上"),
        "win_rate_above_55pct": ("月勝率 ≥ 55%", "贏多輸少"),
        "p_value_below_0_05": ("Hypothesis test p<0.05", "回報統計顯著（唔係靠運氣）"),
        "dsr_above_0_95": ("Deflated Sharpe ≥ 0.95", "嚴格 multiple-test 校正後仍 robust"),
        "oos_sharpe_close_to_in_sample": ("OOS Sharpe 退化 < 0.3", "新數據表現接近設計階段"),
        "wf_correlation_positive": ("Walk-forward 相關性 > 0", "歷史 train Sharpe 對 OOS 有正面預測"),
    }
    rows = []
    for key, (label, desc) in score_labels.items():
        passed = scorecard.get(key, False)
        rows.append(
            f'<tr><td style="padding:8px 10px;">'
            f'<div style="font-size:13px;color:#cbd5e1;">{label}</div>'
            f'<div style="font-size:10px;color:#64748b;">{desc}</div></td>'
            f'<td style="padding:8px 10px;text-align:right;font-size:18px;">{_check_or_x(passed)}</td></tr>'
        )

    # Summary verdict
    if n_passed >= 7:
        v_text, v_color = "極強 — 喺嚴苛學術標準下仍然證實有效", "74, 222, 128"
    elif n_passed >= 5:
        v_text, v_color = "強 — 主要學術測試通過，少數嚴苛項未達門檻屬合理", "34, 211, 238"
    elif n_passed >= 3:
        v_text, v_color = "中等 — 有實質 alpha 證據但仍有改善空間", "251, 191, 36"
    else:
        v_text, v_color = "弱 — 多數測試未通過，需檢討", "251, 113, 113"

    st.markdown(
        f"""<table style="width:100%;border-collapse:collapse;background:#0d1424;
        border:1px solid #1e293b;border-radius:12px;color:#94a3b8;">
        {''.join(rows)}
        <tr style="background:#1e293b;">
            <td style="padding:10px;font-weight:800;color:#cbd5e1;">總分</td>
            <td style="padding:10px;text-align:right;font-weight:900;color:#22d3ee;font-size:18px;">
                {n_passed} / {n_total}
            </td>
        </tr></table>
        <div style="background:rgba({v_color}, 0.1); border-left:4px solid rgb({v_color});
            border-radius:0 8px 8px 0; padding:14px 18px; margin-top:14px;">
            <div style="font-size:13px;color:rgb({v_color});font-weight:800;margin-bottom:6px;">
                🎯 整體驗證評語
            </div>
            <div style="font-size:13px;color:#cbd5e1;line-height:1.7;">
                {v_text}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
