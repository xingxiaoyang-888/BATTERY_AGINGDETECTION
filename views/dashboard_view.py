"""工业级运行控制台与仿真结果工作区。"""

from collections import namedtuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import DANGER_COLOR, SUCCESS_COLOR, THEME_COLOR, WARNING_COLOR
from utils.api_client import BackendAPIError, post_json
from utils.data_parser import parse_profile_csv
from utils.db_manager import get_history, save_simulation
from utils.report_generator import generate_pdf_report
from views.components import (
    render_ai_insight_box,
    render_empty_state,
    render_kpi_card,
    render_section_header,
)
from views.lifetime_view import render_evidence_workspace, render_lifetime_workspace
from views.sidebar_view import fetch_backend_state, render_sidebar
from views.spatial_view import render_3d_aging_map_view, render_3d_pack_thermal_view


PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,18,26,.42)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=48, b=24),
    font=dict(color="#9db1bb", size=11),
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(126,176,198,.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(126,176,198,.08)", zeroline=False),
)

WORKSPACES = ["运行仿真", "寿命分析", "模型证据", "资产记录"]


def render_dashboard():
    backend = fetch_backend_state()
    _render_topbar(backend)
    current = st.session_state.get("workspace_page", WORKSPACES[0])
    page = st.segmented_control(
        "工作区导航",
        WORKSPACES,
        default=current if current in WORKSPACES else WORKSPACES[0],
        label_visibility="collapsed",
        key="workspace_selector",
        width="stretch",
    ) or WORKSPACES[0]
    st.session_state["workspace_page"] = page

    config = render_sidebar(show_simulation_controls=page == "运行仿真")
    if page == "运行仿真":
        _render_simulation_workspace(config)
    elif page == "寿命分析":
        render_lifetime_workspace()
    elif page == "模型证据":
        render_evidence_workspace()
    else:
        _render_asset_records()


def _render_topbar(backend):
    health = backend.get("health") or {}
    backend_state = "ONLINE" if backend.get("online") else "OFFLINE"
    backend_color = "#4ad89f" if backend.get("online") else "#f1b95b"
    st.markdown(
        f"""
        <header class="app-topbar">
            <div class="app-title">
                <div class="section-eyebrow">SODIUM TWIN OPERATIONS / V3.0</div>
                <h1>钠离子电池数字孪生运行控制台</h1>
                <p>物理仿真 · 热安全 · 寿命风险 · 空间诊断 · 工程证据</p>
            </div>
            <div class="system-state">
                <span class="system-chip">API <strong style="color:{backend_color}">{backend_state}</strong></span>
                <span class="system-chip">FMU <strong>{health.get('fmu_count', 0):02d} READY</strong></span>
                <span class="system-chip">LIFE AI <strong>VALIDATED</strong></span>
                <span class="system-chip">MODE <strong>Na-ION</strong></span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_simulation_workspace(config):
    st.markdown(render_section_header(
        "PHYSICAL TWIN WORKSPACE",
        "电—热—老化联合仿真",
        "配置电池包、动态电流、热管理边界与故障工况，驱动FMI 2.0联合求解。",
    ), unsafe_allow_html=True)

    profile = None
    uploaded = config.get("uploaded_file")
    if uploaded is not None:
        profile, message = parse_profile_csv(uploaded)
        if profile:
            config["sim_duration"] = profile["duration"]
            st.success(f"CSV工况已验证：{len(profile['points'])}个节点，持续{profile['duration']:.1f}秒。")
        else:
            st.error(message)

    if config.get("run_btn"):
        _run_simulation(config, profile)

    result = st.session_state.get("sim_result")
    if not result:
        _render_simulation_empty(config)
        return
    _render_simulation_result(*result)


def _run_simulation(config, profile):
    payload = {
        "duration_s": config.get("sim_duration", 600.),
        "pack_current": config.get("pack_current", 50.),
        "current_profile": profile["points"] if profile else None,
        "env_temp": config.get("env_temp", 25.),
        "initial_cell_temp": config.get("initial_cell_temp", config.get("env_temp", 25.)),
        "coolant_inlet_temp": config.get("coolant_inlet_temp", config.get("env_temp", 25.)),
        "coolant_flow_kg_s": config.get("coolant_flow_kg_s", .035),
        "cooling_ua_w_per_k": config.get("cooling_ua_w_per_k", 2.),
        "cell_capacity_ah": config.get("cell_capacity", 50.),
        "init_soc": config.get("init_soc", 80.),
        "init_soh": config.get("init_soh", 100.),
        "series_num": config.get("series_num", 8),
        "parallel_num": config.get("parallel_num", 2),
        "fault_mode": config.get("fault_mode", 1),
        "fault_s_index": config.get("fault_s_index", 1),
        "fault_p_index": config.get("fault_p_index", 1),
        "fault_severity": config.get("fault_severity", 0.),
    }
    try:
        with st.spinner("正在执行FMU联合求解与空间矩阵重组..."):
            api_data = post_json("/api/v1/simulate/predict", payload, timeout=180)["payload"]
        series = api_data["time_series"]
        frame = pd.DataFrame({
            "Time": series["time"],
            "Pack_Voltage": series["voltage"],
            "Pack_Current": series["current"],
            "SOC": series["soc"],
            "Max_Temp": series["t_max"],
            "Min_Temp": series.get("t_min", series["t_max"]),
        })
        frame["Delta_T"] = frame["Max_Temp"] - frame["Min_Temp"]
        kpis = api_data["summary"]
        kpis["soh_loss"] = kpis["loss_ppm"] / 1e6
        kpis["warning"] = kpis.get("warnings") or "Normal"
        st.session_state["sim_result"] = (
            frame, kpis, config.copy(),
            api_data.get("spatial_thermal_matrix", []),
            api_data.get("spatial_soh_matrix", []),
        )
        save_simulation(st.session_state.get("username", "Guest"), config, kpis)
        st.toast("联合解算完成，结果已写入运行工作区。")
    except BackendAPIError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"仿真响应解析失败：{exc}")


def _render_simulation_empty(config):
    columns = st.columns(4)
    cells = config.get("series_num", 8) * config.get("parallel_num", 2)
    previews = [
        ("PACK TOPOLOGY", f"{config.get('series_num', 8)}S{config.get('parallel_num', 2)}P", f"{cells} cells", "#45d2be", "CONFIG"),
        ("CURRENT PROFILE", f"{config.get('pack_current', 0):.0f} A", config.get("profile_mode", "恒流工况"), "#4d8dff", "INPUT"),
        ("THERMAL BOUNDARY", f"{config.get('env_temp', 25):.0f} °C", f"Flow {config.get('coolant_flow_kg_s', .035):.3f} kg/s", "#f1b95b", "BOUNDARY"),
        ("INITIAL STATE", f"{config.get('init_soc', 80):.0f}% SOC", f"{config.get('init_soh', 100):.0f}% SOH", "#4ad89f", "STATE"),
    ]
    for column, card in zip(columns, previews):
        column.markdown(render_kpi_card(*card), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(render_empty_state(
        "READY FOR SOLVER",
        "数字孪生求解任务尚未启动",
        "在左侧完成电池包、工况、热管理和故障配置，然后启动联合解算。寿命分析与模型证据可通过顶部工作区独立访问。",
        "Solver chain: FastAPI → FMU Runtime → Spatial Matrix → Diagnostics",
    ), unsafe_allow_html=True)


def _render_simulation_result(frame, kpis, config, thermal_matrix, soh_matrix):
    info, report = st.columns([3.2, .8], vertical_alignment="center")
    info.success(f"解算完成 · {len(frame):,} points · {config.get('series_num')}S{config.get('parallel_num')}P · {config.get('sim_duration')} s")
    pdf = generate_pdf_report(st.session_state.get("username"), config, kpis, frame)
    report.download_button("导出工程报告", pdf, "SodiumTwin_Report.pdf", "application/pdf", use_container_width=True)

    columns = st.columns(4)
    soh = kpis["soh"]
    temp = kpis["max_temp"]
    cards = [
        ("PACK SOH", f"{soh:.3f}%", f"Loss {kpis['loss_ppm']:.1f} ppm", SUCCESS_COLOR if soh >= 90 else WARNING_COLOR, "HEALTH"),
        ("MAX TEMPERATURE", f"{temp:.1f} °C", f"ΔT max {kpis['delta_t']:.1f} °C", DANGER_COLOR if temp > 55 else SUCCESS_COLOR, "THERMAL"),
        ("DISCHARGE SOP", f"{kpis['sop_dch']:.1f} kW", "10 s pulse boundary", THEME_COLOR, "POWER"),
        ("EQUIVALENT CYCLES", f"{kpis.get('equivalent_full_cycles', 0):.3f}", f"{kpis.get('ah_throughput', 0):.1f} Ah throughput", "#4d8dff", "AGING"),
    ]
    for column, card in zip(columns, cards):
        column.markdown(render_kpi_card(*card), unsafe_allow_html=True)

    overview, electrical, spatial = st.tabs(["运行总览", "电热与功率", "三维空间诊断"])
    with overview:
        _render_overview_charts(frame, kpis)
    with electrical:
        _render_electrothermal_charts(frame, kpis)
    with spatial:
        MockData = namedtuple("MockData", ["temp_matrix_frames"])
        render_3d_pack_thermal_view(MockData(thermal_matrix), config["series_num"], config["parallel_num"])
        render_3d_aging_map_view(soh_matrix[-1] if soh_matrix else None, config["series_num"], config["parallel_num"])


def _render_overview_charts(frame, kpis):
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.Pack_Voltage, name="Voltage", line=dict(color="#45d2be", width=2.4)), secondary_y=False)
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.SOC, name="SOC", line=dict(color="#4d8dff", width=1.8)), secondary_y=True)
        figure.update_layout(title="Pack Electrical Response", height=380, **PLOT_LAYOUT)
        figure.update_yaxes(title_text="Voltage / V", secondary_y=False)
        figure.update_yaxes(title_text="SOC / %", secondary_y=True)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.Max_Temp, name="T max", line=dict(color="#ff6b72", width=2)))
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.Min_Temp, name="T min", line=dict(color="#45d2be", width=1.5), fill="tonexty", fillcolor="rgba(69,210,190,.07)"))
        figure.update_layout(title="Thermal Envelope", height=380, **PLOT_LAYOUT)
        st.plotly_chart(figure, use_container_width=True)

    st.markdown(render_ai_insight_box("运行诊断", _diagnostics(kpis)), unsafe_allow_html=True)


def _render_electrothermal_charts(frame, kpis):
    left, right = st.columns(2, gap="large")
    with left:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.Pack_Current, name="Current", line=dict(color="#4d8dff", width=2)), secondary_y=False)
        figure.add_trace(go.Scatter(x=frame.Time, y=frame.Delta_T, name="ΔT", line=dict(color="#f1b95b", width=2)), secondary_y=True)
        figure.update_layout(title="Current / Temperature Spread", height=370, **PLOT_LAYOUT)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        power = np.abs(frame.Pack_Voltage * frame.Pack_Current / 1000)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=frame.Time, y=[kpis["sop_dch"]] * len(frame), name="SOP limit", line=dict(color="#45d2be", width=1), fill="tozeroy", fillcolor="rgba(69,210,190,.08)"))
        figure.add_trace(go.Scatter(x=frame.Time, y=power, name="Actual power", line=dict(color="#e9f3f6", width=2)))
        figure.update_layout(title="Dynamic Power Envelope", height=370, **PLOT_LAYOUT)
        st.plotly_chart(figure, use_container_width=True)


def _render_asset_records():
    st.markdown(render_section_header(
        "ASSET RECORDS",
        "仿真任务与工程报告",
        "保留最近10次仿真配置和关键健康指标，支持追溯与复现实验。",
    ), unsafe_allow_html=True)
    history = get_history(st.session_state.get("username", "Guest"))
    if not history:
        st.markdown(render_empty_state("NO RECORDS", "暂无资产记录", "完成一次数字孪生仿真后，任务将自动写入本地审计数据库。"), unsafe_allow_html=True)
        return
    rows = []
    for item in history:
        cfg, kpi = item["config"], item["kpis"]
        rows.append({
            "时间": item["time"],
            "规格": f"{cfg.get('series_num', '?')}S{cfg.get('parallel_num', '?')}P",
            "时长 / s": cfg.get("sim_duration", "-"),
            "SOH / %": kpi.get("soh", "-"),
            "最高温 / °C": kpi.get("max_temp", "-"),
            "温差 / °C": kpi.get("delta_t", "-"),
            "状态": kpi.get("warning", "Normal"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _diagnostics(kpis):
    insights = []
    if kpis["max_temp"] > 55:
        insights.append(f"<b>热安全红线：</b>最高温{kpis['max_temp']:.1f}°C，建议立即降低负载并检查冷却回路。")
    elif kpis["max_temp"] > 45:
        insights.append(f"<b>热负荷偏高：</b>最高温{kpis['max_temp']:.1f}°C，需要关注持续升温趋势。")
    else:
        insights.append(f"<b>热状态正常：</b>最高温{kpis['max_temp']:.1f}°C，仍需结合长时工况判断。")
    if kpis["delta_t"] > 5:
        insights.append(f"<b>均温性风险：</b>包内温差{kpis['delta_t']:.1f}°C，建议检查流量分配与接触热阻。")
    else:
        insights.append(f"<b>均温性良好：</b>包内温差{kpis['delta_t']:.1f}°C。")
    return insights
