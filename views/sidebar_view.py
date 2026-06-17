# views/sidebar_view.py
import streamlit as st
import os
import requests
import json

API_BASE = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_fmu_configs():
    """从后端获取可用的 FMU 规格列表（缓存60秒）"""
    try:
        resp = requests.get(f"{API_BASE}/api/v1/fmu/configurations", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def render_sidebar():
    """
    渲染侧边栏并返回配置字典
    包含：电池规格（级联Ns/Np选择）、工况配置、环境参数、初始状态
    """
    config = {}

    # ── 获取可用 FMU 配置 ──
    fmu_data = fetch_fmu_configs()

    with st.sidebar:
        st.markdown("### ⚙️ 仿真参数配置")
        st.caption("Simulation Configuration")
        st.markdown("---")

        # === 1. 电池包规格（级联选择 Ns → Np）===
        st.markdown("#### 1. 电池包规格 (Pack)")

        if fmu_data is None:
            # 后端未启动时的降级方案
            st.warning("⚠️ 后端未连接，使用本地默认配置")
            c1, c2 = st.columns(2)
            with c1:
                config['series_num'] = st.number_input("串联数 (Ns)", value=8, step=1, min_value=1)
            with c2:
                config['parallel_num'] = st.number_input("并联数 (Np)", value=2, step=1, min_value=1)
            cell_cap = st.number_input("单体容量 (Ah)", value=50.0, step=0.5, format="%.1f")
        else:
            # ── 正常模式：从 API 获取选项 ──
            configs = fmu_data['configurations']
            default_ns = fmu_data.get('default_ns', 8)
            default_np = fmu_data.get('default_np', 2)

            # 构建 Ns 选项列表
            ns_options = [c['ns'] for c in configs]
            # 找到默认 Ns 的索引
            try:
                ns_default_idx = ns_options.index(default_ns)
            except ValueError:
                ns_default_idx = 0

            selected_ns = st.selectbox(
                "串联数 Ns（串联组数）",
                options=ns_options,
                index=ns_default_idx,
                format_func=lambda x: f"{x}s  ({x * 3.1:.0f}V 额定)",
                help="选择电池包串联组数。不同规格对应不同电压平台和应用场景。"
            )

            # 根据选中的 Ns 过滤可用的 Np
            np_details = []
            for c in configs:
                if c['ns'] == selected_ns:
                    np_details = c['np_options']
                    break

            # 构建 Np 选项（显示是否有 FMU）
            np_values = [d['np'] for d in np_details]
            np_labels = {}
            try:
                np_default_idx = np_values.index(default_np)
            except ValueError:
                np_default_idx = 0

            # 判断当前选中 Np 是否在可用列表中
            np_to_use = np_values[np_default_idx] if np_values else 1

            selected_np = st.selectbox(
                "并联数 Np（每组并联单体数）",
                options=np_values,
                index=np_default_idx,
                format_func=lambda x: _format_np_label(x, np_details, np_values),
                help="绿色 ✓ = FMU 已就绪可仿真；灰色 ⏳ = 仅有 .mo 模型，需先导出 FMU"
            )

            config['series_num'] = selected_ns
            config['parallel_num'] = selected_np

            # 规格信息
            for d in np_details:
                if d['np'] == selected_np:
                    cells = d['total_cells']
                    voltage = d['nominal_voltage']
                    has_fmu = d['has_fmu']
                    break
            else:
                cells = selected_ns * selected_np
                voltage = round(selected_ns * 3.1, 1)
                has_fmu = False

            cell_cap = st.number_input("单体容量 (Ah)", value=50.0, step=0.5, format="%.1f",
                                       help="钠离子电芯典型容量 50Ah")

            # 实时规格展示
            total_energy = voltage * selected_np * cell_cap / 1000.0
            fmu_status = "✅ FMU就绪" if has_fmu else "⏳ 需导出FMU"
            st.info(
                f"📦 {selected_ns}s{selected_np}p | {cells}电芯 | ~{voltage}V | ~{total_energy:.1f}kWh\n\n"
                f"FMU状态: {fmu_status}"
            )

            config['cell_capacity'] = cell_cap

        # === 2. 运行工况 (Profile) ===
        st.markdown("#### 2. 运行工况 (Profile)")
        profile_mode = st.radio("工况来源", ["⚡ 恒流模式 (CC)", "📂 导入 CSV 文件"], horizontal=True)
        config['profile_mode'] = profile_mode

        if "CSV" in profile_mode:
            config['uploaded_file'] = st.file_uploader("上传工况 (Time, Current)", type=['csv'])

            template_path = os.path.join("assets", "templates", "cycle_profile.csv")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f:
                    st.download_button(
                        label="📥 下载 CSV 模板",
                        data=f,
                        file_name="cycle_template.csv",
                        mime="text/csv",
                        help="下载标准格式模板，填入数据后上传。"
                    )
            else:
                st.caption("⚠️ 模板文件未找到 (assets/templates/cycle_profile.csv)")

            if config.get('uploaded_file'):
                st.success(f"✅ 已加载: {config['uploaded_file'].name}")
            else:
                st.caption("⚠️ 未上传，将使用默认空载演示")

            config['sim_duration'] = 1200
            config['pack_current'] = 0.0
        else:
            c3, c4 = st.columns(2)
            with c3:
                config['pack_current'] = st.number_input("总线电流 (A)", value=50.0, step=10.0,
                                                         help="正值=放电，负值=充电")
            with c4:
                config['sim_duration'] = st.number_input("时长 (s)", value=600, step=60)
            config['uploaded_file'] = None

        # === 3. 环境边界 (Env) ===
        st.markdown("#### 3. 环境边界 (Env)")
        config['env_temp'] = st.slider("环境温度 (°C)", -30, 60, 25)
        config['cooling_type'] = st.selectbox(
            "热管理系统 (TMS)",
            ["Natural", "Air Cooling", "Liquid Cooling", "Liquid Heating", "Immersion"]
        )

        # === 4. 初始状态 ===
        st.markdown("#### 4. 初始状态")
        c5, c6 = st.columns(2)
        with c5:
            config['init_soc'] = st.number_input("Init SOC (%)", 0.0, 100.0, 80.0,
                                                 help="钠离子电池初始荷电状态")
        with c6:
            config['init_soh'] = st.number_input("Init SOH (%)", 50.0, 100.0, 100.0)

        # === 预留：故障注入（默认无故障） ===
        config['fault_mode'] = 1
        config['fault_s_index'] = 1
        config['fault_p_index'] = 1
        config['fault_severity'] = 0.0

        st.markdown("---")
        config['run_btn'] = st.button("🚀 启动仿真 (START)", type="primary", use_container_width=True)

        if st.button("⬅️ 退出登录"):
            st.session_state['logged_in'] = False
            st.rerun()

    return config


def _format_np_label(np_val: int, np_details: list, np_values: list) -> str:
    """格式化 Np 选项标签：显示是否有 FMU"""
    for d in np_details:
        if d['np'] == np_val:
            cells = d['total_cells']
            has_fmu = d['has_fmu']
            status = "✅" if has_fmu else "⏳"
            return f"{np_val}p  ({cells}电芯)  {status}"
    return f"{np_val}p"
