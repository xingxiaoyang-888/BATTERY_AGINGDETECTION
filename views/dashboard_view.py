# views/dashboard_view.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import requests # [新增] 用于前后端通信
from collections import namedtuple # 用于构建兼容数据的临时对象

# --- MVC 模块导入 ---
# 注意：我们不再导入 engine，而是直接通过 API 请求
from models.aging_algorithm import AgingModel
from views.sidebar_view import render_sidebar
from views.components import render_kpi_card, render_ai_insight_box
# 导入我们刚刚升级好的新版 3D 渲染函数
from views.spatial_view import render_3d_pack_thermal_view, render_3d_aging_map_view
from utils.data_parser import parse_profile_csv
from config.settings import THEME_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, NEUTRAL_COLOR
from utils.db_manager import save_simulation, get_history
from utils.report_generator import generate_pdf_report

# [新增] 后端 API 地址
API_URL = "http://localhost:8000/api/v1/simulate/predict"

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
            
        dt = kpis['delta_t'] # 适配新版 API 字段
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

    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
        <h1 style="margin:0;font-size:26px;font-weight:700;color:#e8f0f8;">
        Sodium-Ion Battery Digital Twin
        </h1>
        <span style="background:rgba(0,242,255,0.12);color:#00f2ff;padding:2px 10px;
                     border-radius:4px;font-size:12px;letter-spacing:1px;">
        FMI 2.0 / AI-HYBRID
        </span>
    </div>
    <p style="color:#5a6a80;font-size:13px;margin-top:0;">
    Industrial-grade electro-thermal-aging simulation platform · Multi-FMU architecture · 26 pack configurations
    </p>
    """, unsafe_allow_html=True)

    # 4. 启动云端仿真
    if config['run_btn']:
        with st.spinner("正在呼叫云端物理引擎进行联合解算..."):
            try:
                # 🚀 将前端 UI 数据打包成 HTTP 请求发给 FastAPI
                payload = {
                    "duration_s": config.get('sim_duration', 600.0),
                    "pack_current": config.get('pack_current', 50.0),
                    "env_temp": config.get('env_temp', 25.0),
                    "init_soc": config.get('init_soc', 80.0),
                    "init_soh": config.get('init_soh', 100.0),
                    "series_num": config.get('series_num', 8),
                    "parallel_num": config.get('parallel_num', 2),
                    # 预留故障注入接口，如果没有则默认正常
                    "fault_mode": config.get('fault_mode', 1),
                    "fault_s_index": config.get('fault_s_index', 3),
                    "fault_p_index": config.get('fault_p_index', 1),
                    "fault_severity": config.get('fault_severity', 0.5)
                }

                # 发起网络请求
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # 如果后端报错，这里会拦截
                
                # 解析云端返回的数据
                result_json = response.json()
                api_data = result_json['payload']
                
                # 重组 DataFrame 以兼容旧版图表代码
                ts = api_data['time_series']
                df = pd.DataFrame({
                    "Time": ts['time'],
                    "Pack_Voltage": ts['voltage'],
                    "Pack_Current": ts['current'],
                    "SOC": ts['soc'],
                    "Max_Temp": ts['t_max'],
                    "Min_Temp": ts.get('t_min', ts['t_max']),
                    "Delta_T": [a - b for a, b in zip(ts['t_max'], ts.get('t_min', ts['t_max']))]
                })
                
                # 重组 KPI 数据
                kpis = api_data['summary']
                kpis['soh_loss'] = kpis['loss_ppm'] / 1e6 # 转换回小数以兼容寿命评估
                kpis['warning'] = kpis['warnings'] if kpis['warnings'] else "Normal"
                
                # 存入 Session
                st.session_state['sim_result'] = (df, kpis, config,
                    api_data['spatial_thermal_matrix'],
                    api_data.get('spatial_soh_matrix', []))
                
                # 自动保存到数据库
                save_simulation(st.session_state.get('username', 'Guest'), config, kpis)
                st.toast("✅ 云端解算成功！数据已同步至大屏。")
                
            except requests.exceptions.RequestException as e:
                st.error(f"🔌 云端引擎连接失败，请确认 server.py (端口 8000) 已启动！\n报错信息: {e}")
            except Exception as e:
                st.error(f"❌ 数据解析异常: {e}")

    # 5. 结果展示
    if 'sim_result' in st.session_state:
        df, kpis, cfg, thermal_matrix, soh_matrix = st.session_state['sim_result']
        
        # === 报告生成区 ===
        with st.container():
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.info(f"✅ 云端计算完成 | 数据点: {len(df)} | 耗时: {cfg.get('sim_duration',0)}s")
            
            with col_btn:
                pdf_data = generate_pdf_report(st.session_state.get('username'), cfg, kpis, df)
                if pdf_data:
                    st.download_button("📄 导出 PDF 报告", data=pdf_data, file_name=f"Report_{st.session_state.get('username')}.pdf", mime="application/pdf", use_container_width=True, type="primary")
                else:
                    st.error("⚠️ 报告生成失败")

        st.markdown("---")
        
        # === KPI 卡片 ===
        k1, k2, k3, k4 = st.columns(4)
        soh_val = kpis['soh']
        soh_color = DANGER_COLOR if soh_val < 80 else (WARNING_COLOR if soh_val < 90 else SUCCESS_COLOR)
        
        t_max_val = kpis['max_temp']
        t_color = DANGER_COLOR if t_max_val > 55 else (WARNING_COLOR if t_max_val > 45 else SUCCESS_COLOR)
        
        status_msg = kpis['warning']
        status_c = DANGER_COLOR if "失控" in status_msg or "Risk" in status_msg else (WARNING_COLOR if "预警" in status_msg else SUCCESS_COLOR)
        
        with k1: st.markdown(render_kpi_card("SOH (健康度)", f"{soh_val:.4f}%", f"Loss: {kpis['loss_ppm']:.1f} ppm", soh_color), unsafe_allow_html=True)
        with k2: st.markdown(render_kpi_card("Max Temp", f"{t_max_val:.1f} °C", f"Avg: {df['Max_Temp'].mean():.1f}°C", t_color), unsafe_allow_html=True)
        with k3: st.markdown(render_kpi_card("SOP (放电边界)", f"{kpis['sop_dch']:.1f} kW", "10s Pulse", THEME_COLOR), unsafe_allow_html=True)
        with k4: st.markdown(render_kpi_card("系统诊断", "Monitoring", status_msg, status_c), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === 图表区域 ===
        t1, t2, t3, t4, t5 = st.tabs(["⚡ 电气特性", "🔥 热力学特性", "📉 功率边界", "🧊 3D 空间孪生", "⏳ 寿命预测"])
        
        layout_cfg = dict(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(12,16,24,0.6)',
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(color="#a0b0c0", size=12),
            title_font=dict(color="#e0e8f0", size=15, family='Arial'),
            legend=dict(font=dict(color="#8892a4"), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
        )

        with t1:
            fig_elec = make_subplots(specs=[[{"secondary_y": True}]])
            fig_elec.add_trace(go.Scatter(
                x=df["Time"], y=df["Pack_Voltage"], name="Pack Voltage (V)",
                line=dict(color=THEME_COLOR, width=2.5, shape='spline'),
                fill='tozeroy', fillcolor='rgba(0,242,255,0.04)',
            ), secondary_y=False)
            fig_elec.add_trace(go.Scatter(
                x=df["Time"], y=df["Pack_Current"], name="Current (A)",
                line=dict(color='#8892a4', width=1.5, dash='dot'),
            ), secondary_y=True)
            fig_elec.update_layout(title="Pack Voltage / Current Response", **layout_cfg)
            fig_elec.update_yaxes(title_text="Voltage (V)", secondary_y=False, gridcolor='rgba(255,255,255,0.04)')
            fig_elec.update_yaxes(title_text="Current (A)", secondary_y=True, gridcolor='rgba(255,255,255,0.02)')
            st.plotly_chart(fig_elec, use_container_width=True)

        with t2:
            fig_therm = make_subplots(specs=[[{"secondary_y": True}]])
            fig_therm.add_trace(go.Scatter(
                x=df["Time"], y=df["Max_Temp"], name="T_max",
                line=dict(color=DANGER_COLOR, width=2),
                fill='tozeroy', fillcolor='rgba(255,68,68,0.05)',
            ), secondary_y=False)
            fig_therm.add_trace(go.Scatter(
                x=df["Time"], y=df["Min_Temp"], name="T_min",
                line=dict(color=SUCCESS_COLOR, width=1.5),
                fill='tonexty', fillcolor='rgba(0,230,118,0.06)',
            ), secondary_y=False)
            fig_therm.add_trace(go.Scatter(
                x=df["Time"], y=df["Delta_T"], name="ΔT",
                line=dict(color=WARNING_COLOR, width=2, dash='dash'),
            ), secondary_y=True)
            fig_therm.add_hline(y=55.0, line_dash="dash", line_color="#ff4444",
                                annotation_text="Limit 55°C", annotation_font_color="#ff4444",
                                annotation_position="top left", secondary_y=False)
            fig_therm.update_layout(title="Thermal Distribution & ΔT", **layout_cfg)
            fig_therm.update_yaxes(title_text="°C", secondary_y=False, gridcolor='rgba(255,255,255,0.04)')
            fig_therm.update_yaxes(title_text="ΔT (°C)", secondary_y=True, gridcolor='rgba(255,255,255,0.02)')
            st.plotly_chart(fig_therm, use_container_width=True)
            st.markdown(render_ai_insight_box(" Thermal Diagnostics", get_ai_diagnosis(kpis, "thermal")), unsafe_allow_html=True)

        with t3:
            fig_sop = go.Figure()
            sop_limit = [kpis['sop_dch']] * len(df)
            actual_p = abs(df["Pack_Voltage"] * df["Pack_Current"] / 1000)
            fig_sop.add_trace(go.Scatter(
                x=df["Time"], y=sop_limit, name="SOP Limit",
                line=dict(color=THEME_COLOR, width=0.5), fill='tozeroy',
                fillcolor='rgba(0,242,255,0.1)',
            ))
            fig_sop.add_trace(go.Scatter(
                x=df["Time"], y=actual_p, name="Actual Power",
                line=dict(color='#e8e8e8', width=2.5),
            ))
            fig_sop.update_layout(title="Dynamic SOP Power Envelope", **layout_cfg)
            fig_sop.update_yaxes(title_text="Power (kW)", gridcolor='rgba(255,255,255,0.04)')
            st.plotly_chart(fig_sop, use_container_width=True)
            st.markdown(render_ai_insight_box(" Power Assessment", get_ai_diagnosis(kpis, "power")), unsafe_allow_html=True)

        with t4:
            # 🚀 调用最新版的真实空间热场渲染！
            # 构建一个兼容格式的对象传给 spatial_view
            MockTimeSeriesData = namedtuple('MockTimeSeriesData', ['temp_matrix_frames'])
            ts_mock = MockTimeSeriesData(temp_matrix_frames=thermal_matrix)
            
            render_3d_pack_thermal_view(ts_mock, cfg.get('series_num', 8), cfg.get('parallel_num', 2))
            st.markdown("""<div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;">💡 <b>操作指南:</b> 拖动上方滑块可进行时间轴录像回放，旋转缩放查看每一个电芯的数据。</div>""", unsafe_allow_html=True)
        
        with t5:
            small_layout = layout_cfg.copy()
            small_layout['height'] = 320
            current_loss_rate = kpis['soh_loss'] if kpis['soh_loss'] > 0 else 1e-9
            cycles_to_eol = (kpis['soh'] - 80.0) / (current_loss_rate * 100)
            years_to_eol = cycles_to_eol / 365.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Est. Remaining Life", f"{years_to_eol:.1f} yr",
                      delta=f"~{int(cycles_to_eol)} cycles", delta_color="off")
            m2.metric("Cycle Life Target", "5,000",
                      delta="Na-Ion Standard", delta_color="off")
            m3.metric("Per-Run Degradation", f"{kpis['loss_ppm']:.1f} ppm",
                      delta=f"SOH={kpis['soh']:.4f}%", delta_color="inverse")
            st.markdown("---")

            cycles = np.arange(0, 5000, 50)
            soh_future = kpis['soh'] - (current_loss_rate * cycles * 100)
            fig_eol = go.Figure()
            fig_eol.add_trace(go.Scatter(
                x=cycles, y=soh_future, mode='lines', name='Predicted SOH',
                line=dict(color=THEME_COLOR, width=3, shape='spline'),
                fill='tozeroy', fillcolor='rgba(0,242,255,0.08)',
            ))
            fig_eol.add_hline(y=80, line_dash="dash", line_color="#ff4444",
                              annotation_text="EOL 80%", annotation_font_color="#ff4444")
            fig_eol.add_hline(y=90, line_dash="dot", line_color="#ffaa00",
                              annotation_text="Caution 90%", annotation_font_color="#ffaa00")
            fig_eol.update_layout(
                title="Lifetime SOH Degradation Forecast",
                xaxis_title="Equivalent Full Cycles", yaxis_title="SOH (%)",
                yaxis_range=[75, 100], **small_layout)
            st.plotly_chart(fig_eol, use_container_width=True)

            c_bottom_L, c_bottom_R = st.columns([1.5, 1])
            with c_bottom_L:
                # 使用 FMU 真实 SOH 分布（最后一帧），降级为空列表
                render_3d_aging_map_view(soh_matrix[-1] if soh_matrix else None,
                                         cfg.get('series_num', 8), cfg.get('parallel_num', 2))
            with c_bottom_R:
                avg_t = df['Max_Temp'].mean()
                avg_soc = df['SOC'].mean()
                avg_i = abs(df['Pack_Current'].mean() / cfg.get('parallel_num', 1))
                cal_factor = np.exp(-24000/(8.314*(avg_t+273.15))) * np.exp(0.05*(avg_soc/100))
                cyc_factor = avg_i * 0.1
                total = cal_factor + cyc_factor
                p_cal, p_cyc = (cal_factor/total)*100, (cyc_factor/total)*100
                fig_pie = go.Figure(data=[go.Pie(labels=['Calendar', 'Cycle'], values=[p_cal, p_cyc], hole=.5, marker=dict(colors=[WARNING_COLOR, THEME_COLOR]), textinfo='percent+label', showlegend=False)])
                fig_pie.update_layout(title="🔍 衰减机理归因", **small_layout)
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("👈 请在左侧配置参数并点击 '🚀 启动云端仿真' 以生成数据孪生。")