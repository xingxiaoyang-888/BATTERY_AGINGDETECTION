# views/dashboard_view.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- MVC 模块导入 ---
from models.simulation_engine import engine
from models.aging_algorithm import AgingModel
from views.sidebar_view import render_sidebar
from views.components import render_kpi_card, render_ai_insight_box
from views.spatial_view import render_3d_pack, render_3d_aging_map
from utils.data_parser import parse_profile_csv
from config.settings import THEME_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, NEUTRAL_COLOR
# [导入] 数据库和报告工具
from utils.db_manager import save_simulation, get_history
from utils.report_generator import generate_pdf_report

def get_ai_diagnosis(kpis, category):
    """AI 智能诊断逻辑 (Business Logic Layer)"""
    insights = []
    if category == "thermal":
        t_max = kpis['max_temp']
        if t_max > 55:
            insights.append(f"🔴 **严重过热告警**：当前最高温 **{t_max:.1f}°C** 已突破安全红线 (55°C)。系统面临热失控风险，建议立即切断负载并开启最大冷却功率。")
        elif t_max > 45:
            insights.append(f"🟡 **温度偏高**：核心温度达到 **{t_max:.1f}°C**，建议检查冷却系统是否满负荷运行。")
        else:
            insights.append(f"✅ **温度正常**：电池工作在最佳温度区间 ({t_max:.1f}°C)，热管理状态良好。")
            
        dt = kpis['avg_delta_t']
        if dt > 8:
            insights.append(f"⚠️ **均温性异常**：模组内部温差高达 **{dt:.1f}°C** (标准值 <5°C)。疑似液冷管道局部堵塞或导热垫失效。")
        elif dt > 5:
            insights.append(f"ℹ️ **温差略大**：当前温差 **{dt:.1f}°C**，需持续关注。")
        else:
            insights.append(f"✅ **均温性优异**：温差控制在 **{dt:.1f}°C**，散热均匀。")

    elif category == "power":
        sop = kpis['sop_dch']
        insights.append(f"🚀 **动力性能评估**：当前电池状态 (SOC/Temp) 允许的最大瞬时放电功率为 **{sop:.1f} kW**。")
        if kpis['soc'] < 20:
             insights.append("📉 **低电量限制**：由于 SOC 低于 20%，BMS 已自动限制功率输出以保护电芯。")
        elif kpis['max_temp'] < 0:
             insights.append("❄️ **低温限制**：环境温度过低，功率输出受限，建议开启电池加热系统。")
        else:
             insights.append("✅ **性能全开**：电池处于最佳输出窗口，可支持急加速或高负载爬坡。")
    return insights

def render_dashboard():
    # === 侧边栏用户中心 ===
    with st.sidebar:
        user = st.session_state.get('username', 'Guest')
        st.markdown(f"### 👤 操作员: {user}")
        st.image(f"https://api.dicebear.com/7.x/bottts/png?seed={user}", width=80)
        
        with st.expander("📜 历史仿真记录"):
            history = get_history(user)
            if not history:
                st.caption("暂无记录")
            for rec in history:
                st.markdown(f"**{rec['time']}**")
                st.caption(f"SOH: {rec['kpis'].get('soh',0):.2f}% | MaxT: {rec['kpis'].get('max_temp',0):.1f}C")
                st.divider()

    # 1. 渲染配置
    config = render_sidebar()
    
    # 2. CSV 处理
    if config['uploaded_file'] is not None:
        profile_data, msg = parse_profile_csv(config['uploaded_file'])
        if profile_data: config['sim_duration'] = profile_data['duration']
        else: st.error(msg)

    st.markdown("<h1>🔋 数字孪生可视化驾驶舱</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0a0'>Real-time Electro-Thermal-Aging Simulation</p>", unsafe_allow_html=True)

    # 4. 启动仿真
    if config['run_btn']:
        with st.spinner("正在解算多物理场耦合方程 (Solving ODEs)..."):
            engine.series_num = config['series_num']
            engine.parallel_num = config['parallel_num']
            engine.cell_capacity = config['cell_capacity']
            
            df, kpis = engine.run_profile(
                config['sim_duration'], config['pack_current'], config['env_temp'], 
                config['init_soc'], config['init_soh'], config['cooling_type']
            )
            st.session_state['sim_result'] = (df, kpis, config)
            
            # 自动保存到数据库
            save_simulation(st.session_state.get('username', 'Guest'), config, kpis)
            st.toast("✅ 仿真数据已归档至云端数据库")
# 5. 结果展示
    if 'sim_result' in st.session_state:
        df, kpis, cfg = st.session_state['sim_result']
        
        # === 🎯 [修复] 报告生成区 ===
        # 使用 expander 容器，保证它占据独立空间，不会被遮挡
        with st.container():
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.info(f"✅ 仿真完成 | 数据点: {len(df)} | 耗时: {cfg.get('sim_duration',0)}s")
            
            with col_btn:
                # 尝试生成报告
                pdf_data = generate_pdf_report(st.session_state.get('username'), cfg, kpis, df)
                
                if pdf_data:
                    # 成功则显示下载按钮
                    st.download_button(
                        label="📄 导出 PDF 报告",
                        data=pdf_data,
                        file_name=f"Report_{st.session_state.get('username')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary" # 设为高亮样式
                    )
                else:
                    # 失败则显示错误提示，而不是隐藏
                    st.error("⚠️ 报告生成失败 (包含不支持的字符)")

        st.markdown("---")
        # === KPI 卡片 ===
        k1, k2, k3, k4 = st.columns(4)
        soh_val = kpis['soh']
        soh_color = DANGER_COLOR if soh_val < 80 else (WARNING_COLOR if soh_val < 90 else SUCCESS_COLOR)
        
        t_max_val = kpis['max_temp']
        t_color = DANGER_COLOR if t_max_val > 55 else (WARNING_COLOR if t_max_val > 45 else SUCCESS_COLOR)
        
        status_msg = kpis['warning']
        status_c = DANGER_COLOR if "Risk" in status_msg else (WARNING_COLOR if "Warning" in status_msg else SUCCESS_COLOR)
        loss_ppm = kpis['soh_loss'] * 1e6

        with k1: st.markdown(render_kpi_card("SOH (健康度)", f"{soh_val:.4f}%", f"Loss: {loss_ppm:.1f} ppm", soh_color), unsafe_allow_html=True)
        with k2: st.markdown(render_kpi_card("Max Temp", f"{t_max_val:.1f} °C", f"Avg: {df['Max_Temp'].mean():.1f}°C", t_color), unsafe_allow_html=True)
        with k3: st.markdown(render_kpi_card("SOP (放电边界)", f"{kpis['sop_dch']:.1f} kW", "10s Pulse", THEME_COLOR), unsafe_allow_html=True)
        with k4: st.markdown(render_kpi_card("系统诊断", "Monitoring", status_msg, status_c), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === 图表区域 ===
        t1, t2, t3, t4, t5 = st.tabs(["⚡ 电气特性", "🔥 热力学特性", "📉 功率边界", "🧊 3D 空间孪生", "⏳ 寿命预测"])
        
        layout_cfg = dict(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", height=400, margin=dict(l=20, r=20, t=40, b=20), font=dict(color="#ffffff"), title_font=dict(color="#ffffff"), legend=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0)"))

        with t1:
            fig_elec = make_subplots(specs=[[{"secondary_y": True}]])
            fig_elec.add_trace(go.Scatter(x=df["Time"], y=df["Pack_Voltage"], name="Pack Voltage (V)", line=dict(color=THEME_COLOR, width=2)), secondary_y=False)
            fig_elec.add_trace(go.Scatter(x=df["Time"], y=df["Pack_Current"], name="Current (A)", line=dict(color=NEUTRAL_COLOR, width=1, dash='dot')), secondary_y=True)
            fig_elec.update_layout(title="系统级电压/电流响应 (Pack Level)", **layout_cfg)
            st.plotly_chart(fig_elec, use_container_width=True)

        with t2:
            fig_therm = make_subplots(specs=[[{"secondary_y": True}]])
            fig_therm.add_trace(go.Scatter(x=df["Time"], y=df["Max_Temp"], name="Max Temp", line=dict(color=DANGER_COLOR, width=1)), secondary_y=False)
            fig_therm.add_trace(go.Scatter(x=df["Time"], y=df["Min_Temp"], name="Min Temp", line=dict(color=SUCCESS_COLOR, width=1), fill='tonexty'), secondary_y=False)
            fig_therm.add_trace(go.Scatter(x=df["Time"], y=df["Delta_T"], name="ΔT (温差)", line=dict(color=WARNING_COLOR, width=2)), secondary_y=True)
            fig_therm.add_hline(y=55.0, line_dash="dash", line_color="red", annotation_text="❌ Limit 55°C", annotation_font_color="red", annotation_position="top left", secondary_y=False)
            fig_therm.update_layout(title="热场分布与温差", **layout_cfg)
            st.plotly_chart(fig_therm, use_container_width=True)
            st.markdown(render_ai_insight_box("🌡️ AI 热管理诊断报告", get_ai_diagnosis(kpis, "thermal")), unsafe_allow_html=True)

        with t3:
            fig_sop = go.Figure()
            fig_sop.add_trace(go.Scatter(x=df["Time"], y=df["SOP_Charge_kW"], name="SOP Charge (kW)", line=dict(color="#ab47bc", width=0)))
            fig_sop.add_trace(go.Scatter(x=df["Time"], y=df["SOP_Discharge_kW"], name="SOP Discharge (kW)", line=dict(color=THEME_COLOR, width=0), fill='tonexty')) 
            actual_p = abs(df["Pack_Voltage"] * df["Pack_Current"] / 1000)
            fig_sop.add_trace(go.Scatter(x=df["Time"], y=actual_p, name="Actual Power (kW)", line=dict(color="white", width=2, dash='solid')))
            fig_sop.update_layout(title="SOP 动态功率边界", **layout_cfg)
            st.plotly_chart(fig_sop, use_container_width=True)
            st.markdown(render_ai_insight_box("🚀 动力性能评估", get_ai_diagnosis(kpis, "power")), unsafe_allow_html=True)

        with t4:
            fig_3d = render_3d_pack(df["Max_Temp"].iloc[-1], df["Min_Temp"].iloc[-1], cfg['series_num'])
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown("""<div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;">💡 <b>操作指南:</b> 旋转/缩放查看电芯热场。</div>""", unsafe_allow_html=True)
        
        with t5:
            small_layout = layout_cfg.copy()
            small_layout['height'] = 300
            current_loss_rate = kpis['soh_loss'] if kpis['soh_loss'] > 0 else 1e-9
            cycles_to_eol = (kpis['soh'] - 80.0) / (current_loss_rate * 100)
            years_to_eol = cycles_to_eol / 365.0 
            
            st.markdown("""<style>div[data-testid="metric-container"] {background-color: rgba(255, 255, 255, 0.05);border: 1px solid rgba(255, 255, 255, 0.1);padding: 10px; border-radius: 5px;}div[data-testid="metric-container"] label { color: #a0a0a0; }</style>""", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("预计剩余寿命 (Years)", f"{years_to_eol:.1f} 年", delta="基于当前工况", delta_color="off")
            m2.metric("等效循环寿命 (Cycles)", f"{int(cycles_to_eol)} 次", f"Target: 3000")
            m3.metric("当前单次损耗 (Loss)", f"{kpis['soh_loss']*1e6:.1f} PPM", "-0.5 PPM vs Std", delta_color="inverse")
            st.markdown("---")

            cycles = np.arange(0, 3000, 50)
            soh_future = kpis['soh'] - (current_loss_rate * cycles * 100)
            fig_eol = go.Figure()
            fig_eol.add_trace(go.Scatter(x=cycles, y=soh_future, mode='lines', name='Predicted SOH', line=dict(color=THEME_COLOR, width=3), fill='tozeroy', fillcolor='rgba(0, 242, 255, 0.1)'))
            fig_eol.add_hline(y=80, line_dash="dash", line_color="#ff1744", annotation_text="EOL (80%) End-of-Life", annotation_font_color="#ff1744")
            fig_eol.update_layout(title="📈 全生命周期衰减预测 (Lifetime Prediction)", xaxis_title="Equivalent Cycles", yaxis_title="SOH (%)", yaxis_range=[75, 100], **small_layout)
            st.plotly_chart(fig_eol, use_container_width=True)

            c_bottom_L, c_bottom_R = st.columns([1.5, 1])
            with c_bottom_L:
                fig_aging_3d = render_3d_aging_map(kpis['soh_loss'], cfg['series_num'])
                st.plotly_chart(fig_aging_3d, use_container_width=True)
            with c_bottom_R:
                avg_t = df['Max_Temp'].mean()
                avg_soc = df['SOC'].mean()
                avg_i = abs(df['Pack_Current'].mean() / cfg['parallel_num'])
                cal_factor = np.exp(-24000/(8.314*(avg_t+273.15))) * np.exp(0.05*(avg_soc/100))
                cyc_factor = avg_i * 0.1
                total = cal_factor + cyc_factor
                p_cal, p_cyc = (cal_factor/total)*100, (cyc_factor/total)*100
                fig_pie = go.Figure(data=[go.Pie(labels=['Calendar', 'Cycle'], values=[p_cal, p_cyc], hole=.5, marker=dict(colors=[WARNING_COLOR, THEME_COLOR]), textinfo='percent+label', showlegend=False)])
                fig_pie.update_layout(title="🔍 衰减机理归因", **small_layout)
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("👈 请在左侧配置参数并点击 '🚀 启动仿真' 以生成数据孪生。")