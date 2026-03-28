# views/sidebar_view.py
import streamlit as st
import os

def render_sidebar():
    """
    渲染侧边栏并返回配置字典
    包含：BOM配置、工况选择（带模板下载）、环境参数、初始状态
    """
    config = {}
    
    with st.sidebar:
        st.markdown("### ⚙️ 仿真参数配置")
        st.caption("Simulation Configuration")
        st.markdown("---")

        # === 1. 电池规格 (BOM) ===
        st.markdown("#### 1. 电池规格 (BOM)")
        config['bom_type'] = st.selectbox(
            "材料体系 (Chemistry)", 
            ["NCM811 (高镍)", "NCM523 (通用)", "LFP (铁锂)", "LTO (钛酸锂)"]
        )
        cell_cap = st.number_input("单体容量 (Ah)", value=100.0, step=0.5, format="%.1f")
        
        st.caption("成组方式 (Pack Configuration)")
        c1, c2 = st.columns(2)
        with c1: 
            config['series_num'] = st.number_input("串联数 (S)", value=96, step=1)
        with c2: 
            config['parallel_num'] = st.number_input("并联数 (P)", value=1, step=1)
            
        # 实时计算并展示 Pack 规格
        config['cell_capacity'] = cell_cap
        total_vol = config['series_num'] * 3.7
        total_cap = config['parallel_num'] * cell_cap
        total_energy = total_vol * total_cap / 1000
        st.info(f"📦 Pack规格: {total_vol:.0f}V | {total_cap:.0f}Ah | {total_energy:.1f}kWh")

        # === 2. 运行工况 (Profile) ===
        st.markdown("#### 2. 运行工况 (Profile)")
        profile_mode = st.radio("工况来源", ["📂 导入 CSV 文件", "⚡ 恒流模式 (CC)"], horizontal=True)
        config['profile_mode'] = profile_mode
        
        if "CSV" in profile_mode:
            config['uploaded_file'] = st.file_uploader("上传工况 (Time, Current)", type=['csv'])
            
            # === [新增功能] 下载模板按钮 ===
            # 检查模板文件是否存在，存在则显示下载按钮
            # 注意：需确保 assets/templates/cycle_profile.csv 文件已创建
            template_path = os.path.join("assets", "templates", "cycle_profile.csv")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f:
                    st.download_button(
                        label="📥 下载 CSV 模板",
                        data=f,
                        file_name="cycle_template.csv",
                        mime="text/csv",
                        help="下载标准格式模板，填入数据后上传。格式要求：两列，表头为 Time, Current"
                    )
            else:
                # 如果文件还没创建，提示一下（仅开发阶段可见）
                st.caption("⚠️ 模板文件未找到 (assets/templates/cycle_profile.csv)")
            
            if config['uploaded_file']: 
                st.success(f"✅ 已加载: {config['uploaded_file'].name}")
            else: 
                st.caption("⚠️ 未上传，将使用默认空载演示")
            
            # CSV 模式下，时长由文件决定，这里给个默认值占位
            config['sim_duration'] = 1200 
            config['pack_current'] = 0.0
        else:
            # 恒流模式
            c3, c4 = st.columns(2)
            with c3: 
                config['pack_current'] = st.number_input("总线电流 (A)", value=80.0, step=10.0)
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
        with c5: config['init_soc'] = st.number_input("Init SOC (%)", 0.0, 100.0, 90.0)
        with c6: config['init_soh'] = st.number_input("Init SOH (%)", 50.0, 100.0, 100.0)

        st.markdown("---")
        config['run_btn'] = st.button("🚀 启动仿真 (START)", type="primary", use_container_width=True)
        
        # 退出登录按钮
        if st.button("⬅️ 退出登录"):
            st.session_state['logged_in'] = False
            st.rerun()
            
    return config