"""
Academic validation UI rendering — shared between Basic and Premium apps.
Loads pre-computed validation JSON and renders professional validation section.

Drop into your Streamlit app and call:
    render_academic_validation("validation_premium.json")
"""
import json
import os
import streamlit as st
import plotly.graph_objects as go


def _load_json(filepath: str) -> dict:
    """Robust load — try multiple paths for Streamlit Cloud compatibility."""
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


def render_academic_validation(json_path: str):
    """Renders complete academic validation section. Loads JSON, displays all tests."""
    try:
        v = _load_json(json_path)
    except FileNotFoundError:
        st.warning(f"⚠️ Validation results not found ({json_path}). "
                   "Run `python run_validation.py` to generate.")
        return

    m = v["metrics"]; oos = v["out_of_sample"]; wf = v["walk_forward"]
    cv = v["cross_validation"]; mc = v["monte_carlo"]
    ht = v["hypothesis_test"]; dsr = v["deflated_sharpe"]
    stress = v["stress_periods"]; scorecard = v["scorecard"]

    n_passed = sum(scorecard.values())
    n_total = len(scorecard)

    # ── HEADER + SCORECARD ──
    st.markdown(
        f"""
<div style="background:#0d1424;border:1px solid #22d3ee44;border-radius:14px;
    padding:20px 24px;margin-top:18px;">
  <div style="font-size:11px;color:#22d3ee;font-weight:800;text-transform:uppercase;
      letter-spacing:1.5px;margin-bottom:10px;">
    🎓 ACADEMIC-GRADE VALIDATION · 學術級驗證
  </div>
  <div style="font-size:13px;color:#cbd5e1;line-height:1.7;margin-bottom:8px;">
    本 strategy 通過 <b style="color:#22d3ee;">8 項學術級統計測試</b>。所有 results
    pre-computed，可在 GitHub repo 嘅 <code>validation_*.json</code> 重現。
  </div>
  <div style="font-size:24px;font-weight:900;color:#22d3ee;margin-top:12px;">
    {n_passed} / {n_total} 項通過
  </div>
  <div style="font-size:11px;color:#64748b;margin-top:4px;">
    Generated: {v.get('generated_at', 'N/A')[:19]}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 1: KEY PERFORMANCE METRICS ──
    st.markdown(
        """<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        📊 1. 核心績效與風險指標 · Key Performance Metrics
        </div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_kpi_card(f"{m['cagr']*100:.2f}%", "CAGR", "#4ade80"), unsafe_allow_html=True)
    with c2: st.markdown(_kpi_card(f"{m['mdd']*100:.2f}%", "Max Drawdown", "#f87171"), unsafe_allow_html=True)
    with c3:
        sh_clr = "#4ade80" if m['sharpe'] >= 0.75 else ("#fbbf24" if m['sharpe'] >= 0.5 else "#f87171")
        st.markdown(_kpi_card(f"{m['sharpe']:.2f}", "Sharpe Ratio", sh_clr, "年化, RF=0%"), unsafe_allow_html=True)
    with c4:
        so_clr = "#4ade80" if m['sortino'] >= 1.0 else "#fbbf24"
        st.markdown(_kpi_card(f"{m['sortino']:.2f}", "Sortino Ratio", so_clr, "下行風險調整"), unsafe_allow_html=True)

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        pf_clr = "#4ade80" if m['profit_factor'] >= 1.5 else ("#fbbf24" if m['profit_factor'] >= 1.0 else "#f87171")
        st.markdown(_kpi_card(f"{m['profit_factor']:.2f}", "Profit Factor", pf_clr, "Gross Win/Loss"), unsafe_allow_html=True)
    with c6:
        ca_clr = "#4ade80" if m['calmar'] >= 0.5 else "#fbbf24"
        st.markdown(_kpi_card(f"{m['calmar']:.2f}", "Calmar Ratio", ca_clr, "CAGR/|MDD|"), unsafe_allow_html=True)
    with c7:
        st.markdown(_kpi_card(f"{m['win_rate']*100:.0f}%", "Win Rate", "#22d3ee", "月勝率"), unsafe_allow_html=True)
    with c8:
        st.markdown(_kpi_card(f"{m['annualized_vol']*100:.1f}%", "Annualized Vol", "#94a3b8", "波動率"), unsafe_allow_html=True)

    # ── SECTION 2: OUT-OF-SAMPLE TESTING ──
    st.markdown(
        f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        🔀 2. 樣本外測試 · Out-of-Sample Testing
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
        Train on {oos['train_period']} → Test on {oos['test_period']}.
        Compares performance on data NOT used during strategy design.
        </div>""",
        unsafe_allow_html=True,
    )
    is_m = oos["in_sample"]; os_m = oos["out_of_sample"]
    deg = oos["degradation"]
    st.markdown(
        f"""
<table style="font-size:12px;color:#94a3b8;width:100%;border-collapse:collapse;">
<thead><tr style="background:#1e293b;">
  <th style="padding:8px;text-align:left;color:#cbd5e1;">Metric</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">In-Sample ({oos['train_period']})</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">Out-of-Sample ({oos['test_period']})</th>
  <th style="padding:8px;text-align:right;color:#cbd5e1;">Degradation</th>
</tr></thead><tbody>
<tr><td style="padding:6px 8px;">CAGR</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['cagr']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['cagr']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('cagr',0)>=0 else '#f87171'};">{deg.get('cagr',0)*100:+.2f}pp</td></tr>
<tr style="background:#0a0f1e;">
    <td style="padding:6px 8px;">Sharpe</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['sharpe']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['sharpe']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('sharpe',0)>=0 else '#f87171'};">{deg.get('sharpe',0):+.2f}</td></tr>
<tr><td style="padding:6px 8px;">Sortino</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['sortino']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['sortino']:.2f}</td>
    <td style="padding:6px 8px;text-align:right;color:{'#4ade80' if deg.get('sortino',0)>=0 else '#f87171'};">{deg.get('sortino',0):+.2f}</td></tr>
<tr style="background:#0a0f1e;">
    <td style="padding:6px 8px;">MDD</td>
    <td style="padding:6px 8px;text-align:right;">{is_m['mdd']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">{os_m['mdd']*100:.2f}%</td>
    <td style="padding:6px 8px;text-align:right;">—</td></tr>
</tbody></table>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 3: WALK-FORWARD ──
    if "yearly" in wf and wf.get("yearly"):
        st.markdown(
            f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
            text-transform:uppercase;letter-spacing:1.5px;">
            🚶 3. 走步前向分析 · Walk-Forward Analysis
            </div>
            <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            Rolling 5-year train window → test on next year. Repeated for {wf['n_windows']} years.
            Train↔OOS Sharpe correlation = <b style="color:{'#4ade80' if (wf.get('train_oos_sharpe_correlation') or 0) > 0 else '#f87171'};">
            {wf.get('train_oos_sharpe_correlation', 0):.3f}</b> (positive = strategy edges persist out-of-sample)
            </div>""",
            unsafe_allow_html=True,
        )
        years = [y["year"] for y in wf["yearly"]]
        train_sh = [y.get("train_sharpe") or 0 for y in wf["yearly"]]
        oos_sh = [y.get("oos_sharpe") or 0 for y in wf["yearly"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=train_sh, name="In-Sample (5yr train)",
                                  mode="lines+markers", line=dict(color="#94a3b8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=years, y=oos_sh, name="Out-of-Sample (1yr test)",
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

    # ── SECTION 4: PURGED K-FOLD CV ──
    if "folds" in cv:
        st.markdown(
            f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
            text-transform:uppercase;letter-spacing:1.5px;">
            🔁 4. 組合清洗交叉驗證 · Purged K-Fold Cross-Validation
            </div>
            <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            {cv['n_folds_actual']}-fold CV with {cv['embargo_months']}-month embargo (de Prado method
            to prevent label leakage in financial time series).
            Mean test Sharpe: <b style="color:#22d3ee;">{cv['mean_test_sharpe']:.2f}</b>,
            std: <b>{cv['std_test_sharpe']:.2f}</b>, range:
            [<b>{cv['min_test_sharpe']:.2f}</b>, <b>{cv['max_test_sharpe']:.2f}</b>]
            </div>""",
            unsafe_allow_html=True,
        )
        rows = "".join([
            f"<tr {'style=\"background:#0a0f1e;\"' if i%2 else ''}>"
            f"<td style='padding:6px 8px;'>Fold {f['fold']}</td>"
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
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">Fold</th>
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">Test Period</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Test Sharpe</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Test CAGR</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Test MDD</th>
            </tr></thead><tbody>{rows}</tbody></table>""",
            unsafe_allow_html=True,
        )

    # ── SECTION 5: MONTE CARLO BOOTSTRAP ──
    st.markdown(
        f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        🎲 5. 蒙地卡羅自助法 · Monte Carlo Block Bootstrap
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
        {mc['n_paths']:,} synthetic paths via {mc['block_size_months']}-month block bootstrap
        (preserves serial correlation). Shows distribution of plausible outcomes if this strategy
        were repeatedly randomized.
        </div>""",
        unsafe_allow_html=True,
    )

    # MC distribution table
    st.markdown(
        f"""<table style="font-size:11px;color:#94a3b8;width:100%;border-collapse:collapse;">
        <thead><tr style="background:#1e293b;">
          <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">Metric</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">5%-ile (Bear)</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">25%-ile</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">50%-ile (Median)</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">95%-ile (Bull)</th>
          <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Actual</th>
        </tr></thead><tbody>
        <tr><td style="padding:6px 8px;">CAGR</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p5']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p25']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p50']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['cagr_p95']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_cagr']*100:.1f}%</td></tr>
        <tr style="background:#0a0f1e;">
            <td style="padding:6px 8px;">MDD</td>
            <td style="padding:6px 8px;text-align:right;color:#f87171;">{mc['mdd_p5']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p25']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p50']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;">{mc['mdd_p95']*100:.1f}%</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_mdd']*100:.1f}%</td></tr>
        <tr><td style="padding:6px 8px;">Sharpe</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p5']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;">—</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p50']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;">{mc['sharpe_p95']:.2f}</td>
            <td style="padding:6px 8px;text-align:right;color:#22d3ee;font-weight:800;">{mc['actual_sharpe']:.2f}</td></tr>
        </tbody></table>
        <div style="font-size:11px;color:#64748b;margin-top:12px;line-height:1.6;">
        🔻 Tail risk: <b>{mc['prob_cagr_below_zero']*100:.1f}%</b> chance CAGR &lt; 0,
        <b>{mc['prob_mdd_below_50pct']*100:.1f}%</b> chance MDD &gt; 50%,
        <b>{mc['prob_sharpe_below_0_5']*100:.1f}%</b> chance Sharpe &lt; 0.5
        </div>""",
        unsafe_allow_html=True,
    )

    # ── SECTION 6: HYPOTHESIS TEST ──
    st.markdown(
        f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        📐 6. 假設檢定 · Hypothesis Test
        </div>
        <div style="background:#0d1424;border:1px solid #1e293b;border-radius:12px;padding:14px 18px;">
        <div style="font-size:12px;color:#94a3b8;line-height:1.7;">
        H₀ (虛無假設): 策略月度回報嘅平均值 = 0（即策略冇 alpha）<br>
        Test: One-sample t-test on {ht['n_observations']} monthly observations
        </div>
        <div style="margin-top:10px;font-size:13px;color:#cbd5e1;">
        t-statistic = <b>{ht['t_statistic']:.3f}</b>　|
        p-value = <b style="color:{'#4ade80' if ht['significant_at_5pct'] else '#f87171'};">{ht['p_value']:.4f}</b>　|
        {_check_or_x(ht['significant_at_5pct'])} 5% level　{_check_or_x(ht['significant_at_1pct'])} 1% level
        </div>
        <div style="margin-top:8px;font-size:11px;color:#64748b;font-style:italic;">
        {ht['interpretation']}
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ── SECTION 7: DEFLATED SHARPE RATIO ──
    psr = dsr['deflated_sharpe']
    psr_clr = "#4ade80" if psr > 0.95 else ("#fbbf24" if psr > 0.5 else "#f87171")
    st.markdown(
        f"""<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        🛡️ 7. 平減夏普比率 · Deflated Sharpe Ratio (Bailey & López de Prado)
        </div>
        <div style="background:#0d1424;border:1px solid #1e293b;border-radius:12px;padding:14px 18px;">
        <div style="font-size:12px;color:#94a3b8;line-height:1.7;">
        Adjusts Sharpe ratio for <b>multiple-testing bias</b>. With many parameter combinations
        tested, the highest Sharpe is naturally inflated. DSR computes the probability
        the observed Sharpe is genuinely better than random across N trials.
        </div>
        <div style="margin-top:10px;font-size:13px;color:#cbd5e1;">
        Raw Sharpe (annualized) = <b>{dsr['raw_sharpe_annualized']:.3f}</b>　|
        Trials tested = <b>{dsr['n_trials_assumed']}</b><br>
        Expected max Sharpe under H₀ = <b>{dsr['expected_max_sharpe_annualized_under_null']:.3f}</b>　|
        DSR = <b style="color:{psr_clr};font-size:18px;">{psr:.3f}</b>
        </div>
        <div style="margin-top:10px;padding:10px 12px;background:#060c1a;border-radius:8px;
            font-size:11px;color:#94a3b8;line-height:1.6;font-style:italic;">
        {dsr['interpretation']}
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ── SECTION 8: STRESS PERIODS ──
    if stress:
        st.markdown(
            """<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
            text-transform:uppercase;letter-spacing:1.5px;">
            🔥 8. 壓力測試 · Historical Stress Periods
            </div>""",
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
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">Crisis</th>
              <th style="padding:6px 8px;text-align:left;color:#cbd5e1;">Period</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Total Return</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Intra-Period MDD</th>
              <th style="padding:6px 8px;text-align:right;color:#cbd5e1;">Worst Month</th>
            </tr></thead><tbody>{rows}</tbody></table>""",
            unsafe_allow_html=True,
        )

    # ── FINAL SCORECARD ──
    st.markdown(
        """<div style="font-size:13px;color:#cbd5e1;font-weight:800;margin:24px 0 10px;
        text-transform:uppercase;letter-spacing:1.5px;">
        📋 驗證評分卡 · Final Scorecard
        </div>""",
        unsafe_allow_html=True,
    )

    score_labels = {
        "sharpe_above_0_75": "Sharpe ≥ 0.75",
        "calmar_above_0_4": "Calmar ≥ 0.4",
        "profit_factor_above_1_5": "Profit Factor ≥ 1.5",
        "win_rate_above_55pct": "Win Rate ≥ 55%",
        "p_value_below_0_05": "Hypothesis test significant @ 5%",
        "dsr_above_0_95": "Deflated Sharpe ≥ 0.95 (multiple-test corrected)",
        "oos_sharpe_close_to_in_sample": "OOS Sharpe within 0.3 of in-sample",
        "wf_correlation_positive": "Walk-forward train↔OOS correlation > 0",
    }
    rows = []
    for key, label in score_labels.items():
        passed = scorecard.get(key, False)
        rows.append(
            f'<tr><td style="padding:8px 10px;font-size:13px;">{label}</td>'
            f'<td style="padding:8px 10px;text-align:right;font-size:18px;">{_check_or_x(passed)}</td></tr>'
        )
    st.markdown(
        f"""<table style="width:100%;border-collapse:collapse;background:#0d1424;
        border:1px solid #1e293b;border-radius:12px;color:#94a3b8;">
        {''.join(rows)}
        <tr style="background:#1e293b;">
            <td style="padding:10px;font-weight:800;color:#cbd5e1;">總分</td>
            <td style="padding:10px;text-align:right;font-weight:900;color:#22d3ee;font-size:18px;">
                {n_passed} / {n_total}
            </td>
        </tr></table>""",
        unsafe_allow_html=True,
    )
