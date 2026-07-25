"""真实寿命预测、模型证据与风险解释页面。"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import BackendAPIError, get_json, post_json
from views.components import render_empty_state, render_kpi_card, render_section_header


PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,18,26,.42)",
    font=dict(color="#9db1bb", size=11),
    margin=dict(l=20, r=20, t=45, b=20),
    xaxis=dict(gridcolor="rgba(126,176,198,.08)"),
    yaxis=dict(gridcolor="rgba(126,176,198,.08)"),
)


def render_lifetime_workspace():
    st.markdown(render_section_header(
        "LIFETIME INTELLIGENCE",
        "寿命预测与风险区间",
        "RWTH删失感知队列、运行历史、物理应力与XGBoost短期趋势融合。",
    ), unsafe_allow_html=True)

    input_col, scenario_col = st.columns([1, 1], gap="large")
    with input_col:
        st.markdown("#### 当前资产状态")
        c1, c2 = st.columns(2)
        current_soh = c1.number_input("当前 SOH / %", 50., 105., 95., step=.1)
        current_cycle = c2.number_input("当前循环数", 0, 50000, 500, step=10)
        c3, c4 = st.columns(2)
        eol_soh = c3.selectbox("EOL 阈值", [90., 85., 80.], index=2, format_func=lambda x: f"{x:.0f}% SOH")
        cycles_per_day = c4.number_input("每日等效循环", .1, 100., 1., step=.5)
        history_file = st.file_uploader(
            "可选：上传历史SOH（cycle_index, soh_pct）",
            type=["csv"],
            help="至少8个历史点可启用AI短期趋势校准；SOH>98%时按验证门禁不用于远期RUL。",
        )

    with scenario_col:
        st.markdown("#### 未来运行场景")
        c1, c2 = st.columns(2)
        temperature = c1.number_input("工作温度 / °C", -30., 80., 25.)
        temp_spread = c2.number_input("包内温差 / °C", 0., 50., 3.)
        c3, c4 = st.columns(2)
        charge_rate = c3.number_input("充电倍率 / C", .05, 15., 1.)
        discharge_rate = c4.number_input("放电倍率 / C", .05, 15., 1.)
        c5, c6 = st.columns(2)
        soc_min = c5.number_input("SOC 下限 / %", 0., 99., 0.)
        soc_max = c6.number_input("SOC 上限 / %", 1., 100., 100.)
        capacity = st.number_input(
            "单体额定容量 / Ah",
            .1,
            2000.,
            50.,
            step=.5,
            help="RWTH训练电芯为1.2Ah；公司50Ah电芯将触发OOD提示与更宽的寿命区间。",
        )

    run = st.button("执行寿命预测", type="primary", use_container_width=True, key="run_lifetime")
    if run:
        if soc_min >= soc_max:
            st.error("SOC下限必须小于SOC上限。")
        else:
            history = _parse_history(history_file, current_cycle, current_soh)
            payload = {
                "current_soh_pct": current_soh,
                "current_cycle": current_cycle,
                "eol_soh_pct": eol_soh,
                "cycles_per_day": cycles_per_day,
                "max_horizon_cycles": 10000,
                "scenario": {
                    "temperature_c": temperature,
                    "temperature_spread_c": temp_spread,
                    "c_rate_charge": charge_rate,
                    "c_rate_discharge": discharge_rate,
                    "soc_min_pct": soc_min,
                    "soc_max_pct": soc_max,
                    "nominal_capacity_ah": capacity,
                },
                "history": history,
            }
            try:
                with st.spinner("正在融合队列、物理应力与AI趋势..."):
                    response = post_json("/api/v1/lifetime/predict", payload, timeout=90)
                st.session_state["lifetime_result"] = response["payload"]
            except BackendAPIError as exc:
                st.error(str(exc))

    result = st.session_state.get("lifetime_result")
    if result:
        _render_lifetime_result(result)
    else:
        st.markdown(render_empty_state(
            "RUL / SOH / OOD",
            "等待寿命分析任务",
            "配置当前资产状态与未来运行场景。平台将返回RUL、置信区间、域外检测、应力分解和证据边界。",
            "后端接口：POST /api/v1/lifetime/predict",
        ), unsafe_allow_html=True)


def render_evidence_workspace():
    st.markdown(render_section_header(
        "MODEL EVIDENCE",
        "数据覆盖与模型证据",
        "展示模型能证明什么、不能证明什么，以及各EOL阈值的真实事件覆盖。",
    ), unsafe_allow_html=True)
    try:
        evidence = get_json("/api/v1/lifetime/evidence", timeout=20)["payload"]
    except BackendAPIError as exc:
        st.error(str(exc))
        st.markdown(render_empty_state(
            "EVIDENCE OFFLINE", "证据服务暂不可用", "请启动FastAPI后端以读取RWTH审计结果。"
        ), unsafe_allow_html=True)
        return

    thresholds = evidence.get("thresholds", [])
    cols = st.columns(max(len(thresholds), 1))
    for col, item in zip(cols, thresholds):
        threshold = item["eol_soh_threshold"]
        coverage = item["event_fraction"]
        color = "#4ad89f" if coverage >= .45 else "#f1b95b" if coverage >= .25 else "#ff6b72"
        with col:
            st.markdown(render_kpi_card(
                f"{threshold:.0%} EOL EVENTS",
                f"{item['events']} / {item['cells']}",
                f"右删失 {item['right_censored']} · 覆盖 {coverage:.1%}",
                color,
                "OBSERVED",
            ), unsafe_allow_html=True)

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        fig = go.Figure()
        labels = [f"{item['eol_soh_threshold']:.0%}" for item in thresholds]
        fig.add_bar(name="真实EOL事件", x=labels, y=[item["events"] for item in thresholds], marker_color="#45d2be")
        fig.add_bar(name="右删失", x=labels, y=[item["right_censored"] for item in thresholds], marker_color="#334b59")
        fig.update_layout(title="EOL Evidence Coverage", barmode="stack", height=360, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        model = evidence.get("model", {})
        metrics = model.get("one_step_test_metrics") or {}
        st.markdown(render_section_header("DEPLOYED MODEL", "当前部署证据", "自动选择验证误差最低的残差模型。"), unsafe_allow_html=True)
        st.metric("XGBoost Test RMSE", f"{metrics.get('RMSE', 0):.6f}")
        st.metric("已验证SOH跨度", f"{model.get('validated_prediction_horizon_cycles', 128)} cycles")
        st.metric("RUL AI门禁", model.get("rul_ai_activation_rule", "SOH ≤ 0.98"))
        st.warning("80% EOL证据有限，远期RUL必须结合宽区间和公司数据校准。")


def _parse_history(history_file, current_cycle, current_soh):
    if history_file is None:
        return []
    try:
        frame = pd.read_csv(io.BytesIO(history_file.getvalue()))
        required = {"cycle_index", "soh_pct"}
        if not required.issubset(frame.columns):
            st.warning("历史CSV缺少 cycle_index 或 soh_pct，已按无历史模式运行。")
            return []
        clean = frame[["cycle_index", "soh_pct"]].dropna().sort_values("cycle_index")
        clean = clean[clean["cycle_index"] <= current_cycle]
        return clean.to_dict("records")
    except Exception as exc:
        st.warning(f"历史CSV解析失败：{exc}")
        return []


def _render_lifetime_result(result):
    prediction = result["prediction"]
    uncertainty = result["uncertainty"]
    domain = result["domain_check"]
    model = result["model"]
    interval = uncertainty["rul_interval_cycles"]
    confidence = uncertainty["confidence_score"]
    color = "#4ad89f" if confidence >= .75 else "#f1b95b" if confidence >= .5 else "#ff6b72"

    st.markdown("---")
    cols = st.columns(4)
    cards = [
        ("ESTIMATED RUL", f"{prediction['rul_cycles']:,} cycles", f"约 {prediction['rul_days']:,.0f} days", color, "HYBRID"),
        ("RUL INTERVAL", f"{interval[0]:,} – {interval[1]:,}", uncertainty["interval_basis"], "#4d8dff", "UNCERTAINTY"),
        ("CONFIDENCE", f"{confidence:.0%}", uncertainty["confidence_level"].upper(), color, "CALIBRATED"),
        ("DOMAIN STATUS", "OUT OF DOMAIN" if domain["is_ood"] else "IN DOMAIN", f"OOD score {domain['ood_score']:.2f}", "#ff6b72" if domain["is_ood"] else "#4ad89f", "GUARDRAIL"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(render_kpi_card(*card), unsafe_allow_html=True)

    left, right = st.columns([1.45, .55], gap="large")
    with left:
        trajectory = pd.DataFrame(result["trajectory"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trajectory["cycle"], y=trajectory["soh"] * 100,
            mode="lines", name="Projected SOH",
            line=dict(color="#45d2be", width=3),
            fill="tozeroy", fillcolor="rgba(69,210,190,.06)",
        ))
        fig.add_hline(y=prediction["eol_soh_threshold"] * 100, line_dash="dash", line_color="#ff6b72", annotation_text="EOL")
        fig.update_layout(title="Monotonic SOH Projection", height=390, xaxis_title="Cycle", yaxis_title="SOH / %", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(render_section_header("FUSION SOURCES", "速率融合来源", "权重由历史覆盖与验证门禁决定。"), unsafe_allow_html=True)
        for source in result.get("rate_sources", []):
            st.progress(float(source["fusion_weight"]), text=f"{source['source']} · {source['fusion_weight']:.1%}")
        st.caption(f"AI loaded: {model.get('ai_loaded')} · AI used: {model.get('ai_used_for_prediction')}")
        for warning in result.get("warnings", []):
            st.warning(warning)
