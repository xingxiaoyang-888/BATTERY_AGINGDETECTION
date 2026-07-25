"""仿真工作区侧边栏。"""

import os

import requests
import streamlit as st

API_BASE = os.getenv("BATTERY_API_BASE", "http://localhost:8000")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_backend_state():
    state = {"online": False, "health": None, "fmus": None}
    try:
        health = requests.get(f"{API_BASE}/api/v1/health", timeout=2.5)
        configs = requests.get(f"{API_BASE}/api/v1/fmu/configurations", timeout=3)
        if health.ok:
            state["online"] = True
            state["health"] = health.json()
        if configs.ok:
            state["fmus"] = configs.json()
    except requests.RequestException:
        pass
    return state


def render_sidebar(show_simulation_controls=True):
    backend = fetch_backend_state()
    config = {"backend_online": backend["online"]}
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">Na</div>
                <div><strong>SODIUM TWIN</strong><small>OPERATIONS CONSOLE</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        status_class = "online" if backend["online"] else "standby"
        status_text = "BACKEND ONLINE" if backend["online"] else "BACKEND OFFLINE"
        st.markdown(
            f'<div class="sidebar-status"><i class="status-dot {status_class}"></i>{status_text}'
            f'<span>{API_BASE.replace("http://", "")}</span></div>',
            unsafe_allow_html=True,
        )

        if show_simulation_controls:
            _render_simulation_controls(config, backend.get("fmus"))
        else:
            st.markdown(
                """
                <div class="sidebar-context">
                    <div class="section-eyebrow">WORKSPACE MODE</div>
                    <h4>数据分析工作区</h4>
                    <p>当前页面使用独立寿命与证据接口，不需要先运行FMU。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-user-label">CURRENT OPERATOR</div>', unsafe_allow_html=True)
        user_col, exit_col = st.columns([1.35, .65], vertical_alignment="center")
        with user_col:
            st.markdown(f"**{st.session_state.get('username', 'Guest')}**")
        with exit_col:
            if st.button("退出", use_container_width=True, key="sidebar_logout"):
                st.session_state["logged_in"] = False
                st.session_state.pop("sim_result", None)
                st.rerun()
    return config


def _render_simulation_controls(config, fmu_data):
    st.markdown('<div class="sidebar-section-title">SIMULATION SETUP</div>', unsafe_allow_html=True)

    with st.expander("01 · 电池包与电芯", expanded=True):
        if fmu_data and fmu_data.get("configurations"):
            configs = fmu_data["configurations"]
            ns_options = [item["ns"] for item in configs]
            default_ns = fmu_data.get("default_ns", ns_options[0])
            ns = st.selectbox(
                "串联数量 Ns",
                ns_options,
                index=ns_options.index(default_ns) if default_ns in ns_options else 0,
                format_func=lambda value: f"{value}s · 约 {value * 3.1:.0f} V",
            )
            details = next(item["np_options"] for item in configs if item["ns"] == ns)
            ready = [item for item in details if item.get("has_fmu")]
            np_options = [item["np"] for item in (ready or details)]
            np_value = st.selectbox(
                "并联数量 Np",
                np_options,
                format_func=lambda value: _np_label(value, details),
            )
            has_fmu = any(item["np"] == np_value and item.get("has_fmu") for item in details)
        else:
            c1, c2 = st.columns(2)
            ns = c1.number_input("Ns", min_value=1, value=8, step=1)
            np_value = c2.number_input("Np", min_value=1, value=2, step=1)
            has_fmu = False
        capacity = st.number_input("单体额定容量 / Ah", min_value=.1, max_value=2000., value=50., step=.5)
        config.update(series_num=int(ns), parallel_num=int(np_value), cell_capacity=float(capacity))
        cells = int(ns) * int(np_value)
        state = "FMU READY" if has_fmu else "LOCAL FALLBACK"
        st.caption(f"{cells} CELLS · {int(ns)}S{int(np_value)}P · {state}")

    with st.expander("02 · 运行工况", expanded=True):
        profile_mode = st.radio("工况来源", ["恒流工况", "CSV 时变工况"], horizontal=True)
        config["profile_mode"] = profile_mode
        if profile_mode == "CSV 时变工况":
            config["uploaded_file"] = st.file_uploader("上传 Time / Current CSV", type=["csv"])
            template_path = os.path.join("assets", "templates", "cycle_profile.csv")
            if os.path.exists(template_path):
                with open(template_path, "rb") as file:
                    st.download_button("下载工况模板", file, "cycle_profile.csv", "text/csv", use_container_width=True)
            config["pack_current"] = 0.0
            config["sim_duration"] = 1200.0
        else:
            config["uploaded_file"] = None
            c1, c2 = st.columns(2)
            config["pack_current"] = c1.number_input("总线电流 / A", value=50., step=10.)
            config["sim_duration"] = c2.number_input("仿真时长 / s", min_value=10, value=600, step=60)

    with st.expander("03 · 热管理边界", expanded=False):
        config["env_temp"] = st.slider("环境温度 / °C", -30, 60, 25)
        c1, c2 = st.columns(2)
        config["initial_cell_temp"] = c1.number_input("初始电芯温度", -30., 90., float(config["env_temp"]))
        config["coolant_inlet_temp"] = c2.number_input("冷却液入口温度", -30., 80., float(config["env_temp"]))
        config["coolant_flow_kg_s"] = st.number_input("冷却液质量流量 / kg·s⁻¹", .001, 2., .035, step=.005, format="%.3f")
        config["cooling_ua_w_per_k"] = st.number_input("等效换热系数 UA / W·K⁻¹", .1, 100., 2., step=.5)

    with st.expander("04 · 初始状态与故障", expanded=False):
        c1, c2 = st.columns(2)
        config["init_soc"] = c1.number_input("初始 SOC / %", 0., 100., 80.)
        config["init_soh"] = c2.number_input("初始 SOH / %", 50., 100., 100.)
        enable_fault = st.toggle("启用故障注入", value=False)
        if enable_fault:
            config["fault_mode"] = st.selectbox("故障模式", [2, 3, 4, 5, 6], format_func=lambda x: f"MODE {x}")
            c3, c4 = st.columns(2)
            config["fault_s_index"] = c3.number_input("串联位置", 1, int(ns), 1)
            config["fault_p_index"] = c4.number_input("并联位置", 1, int(np_value), 1)
            config["fault_severity"] = st.slider("故障严重度", 0., 1., .5, .05)
        else:
            config.update(fault_mode=1, fault_s_index=1, fault_p_index=1, fault_severity=0.)

    config["run_btn"] = st.button(
        "启动数字孪生解算",
        type="primary",
        use_container_width=True,
        disabled=not bool(config.get("backend_online", True)),
        key="run_simulation",
    )


def _np_label(value, details):
    detail = next((item for item in details if item["np"] == value), {})
    state = "READY" if detail.get("has_fmu") else "PENDING"
    return f"{value}p · {detail.get('total_cells', '?')} cells · {state}"
