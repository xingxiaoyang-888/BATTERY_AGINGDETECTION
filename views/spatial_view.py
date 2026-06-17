# views/spatial_view.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from models.data_structures import TimeSeriesData

# ── 工业配色 ──────────────────────────────────────
CELL_GAP = 0.15          # 电芯间距
CELL_W = 0.7             # 电芯宽度
CELL_H = 1.6             # 电芯高度 (比例接近真实 18650/方形电芯)
MODULE_GAP = 1.0         # 模组间距（每8串一组）
COOLING_H = -0.5         # 冷板 Z 位置

COOLANT_COLOR = '#00bcd4'
BUS_BAR_COLOR = '#b0bec5'
CASING_COLOR = '#37474f'
GRID_COLOR = '#263238'


def _build_cell_mesh(x, y, z, w, d, h):
    """构建单个电芯的 3D 长方体网格顶点"""
    ox, oy, oz = x - w/2, y - d/2, z
    verts = [
        [ox, oy, oz], [ox+w, oy, oz], [ox+w, oy+d, oz], [ox, oy+d, oz],  # 底面
        [ox, oy, oz+h], [ox+w, oy, oz+h], [ox+w, oy+d, oz+h], [ox, oy+d, oz+h],  # 顶面
    ]
    faces = [
        [0,1,2,3], [4,5,6,7],  # 底 / 顶
        [0,1,5,4], [1,2,6,5],  # 前 / 右
        [2,3,7,6], [3,0,4,7],  # 后 / 左
    ]
    x_face, y_face, z_face = [], [], []
    for f in faces:
        for v in f:
            x_face.append(verts[v][0])
            y_face.append(verts[v][1])
            z_face.append(verts[v][2])
        x_face.append(None); y_face.append(None); z_face.append(None)
    return x_face, y_face, z_face


def _add_colorbar_trace(fig, values, title, x_pos=0.85):
    """不添加独立 trace，仅通过 marker 已绑定的 colorbar 工作"""
    pass


def render_3d_pack_thermal_view(ts_data: TimeSeriesData, Ns: int, Np: int):
    """工业级 3D 电池包热场数字孪生 — 立体电芯 + 冷板 + 母线 + 外壳"""
    st.subheader(" Battery Pack 3D Thermal Digital Twin")

    if not ts_data.temp_matrix_frames:
        st.info("请先启动解算以生成热场数据")
        return

    total_frames = len(ts_data.temp_matrix_frames)
    selected_idx = st.slider("Timeline (s)", 0, total_frames - 1, total_frames - 1)
    T = np.array(ts_data.temp_matrix_frames[selected_idx])

    t_min, t_max = float(np.min(T)), float(np.max(T))
    mid_temp = (t_min + t_max) / 2

    traces = []

    # ── 1. 电芯 3D 立体盒 ──
    cell_x, cell_y, cell_z, cell_temps, cell_labels = [], [], [], [], []
    for s in range(Ns):
        module_idx = s // 8
        x_offset = module_idx * MODULE_GAP
        for p in range(Np):
            cx = p * (CELL_W + CELL_GAP) + x_offset
            cy = s * (CELL_W + CELL_GAP)
            cell_temp = T[s, p]
            cell_x.append(cx + CELL_W/2)
            cell_y.append(cy + CELL_W/2)
            cell_z.append(0)
            cell_temps.append(cell_temp)
            cell_labels.append(f"S{s+1}P{p+1}: {cell_temp:.1f}°C")

            # 立体盒 — intensity 用于每顶点着色
            xf, yf, zf = _build_cell_mesh(cx, cy, 0, CELL_W, CELL_W, CELL_H)
            traces.append(go.Mesh3d(
                x=xf, y=yf, z=zf,
                intensity=[cell_temp]*len(xf),
                colorscale='Turbo',
                intensitymode='vertex',
                cmin=t_min, cmax=t_max,
                showscale=False,
                hoverinfo='skip',
            ))

    # ── 2. 冷板（底部平面） ──
    total_x = (Np * (CELL_W + CELL_GAP) - CELL_GAP) + (Ns // 8) * MODULE_GAP
    total_y = Ns * (CELL_W + CELL_GAP) - CELL_GAP
    cp_x = [-0.3, total_x + 0.3, total_x + 0.3, -0.3]
    cp_y = [-0.3, -0.3, total_y + 0.3, total_y + 0.3]
    cp_z = [COOLING_H]*4
    traces.append(go.Mesh3d(
        x=cp_x, y=cp_y, z=cp_z,
        color=COOLANT_COLOR, opacity=0.15,
        showscale=False, hoverinfo='skip', name='Cooling Plate'
    ))

    # ── 3. 母线排（连接并联电芯顶部） ──
    for s in range(Ns):
        module_idx = s // 8
        x_off = module_idx * MODULE_GAP
        for p in range(Np):
            cx = p * (CELL_W + CELL_GAP) + x_off
            cy = s * (CELL_W + CELL_GAP)
            traces.append(go.Scatter3d(
                x=[cx + CELL_W/2, cx + CELL_W/2],
                y=[cy + CELL_W/2, cy + CELL_W/2],
                z=[CELL_H, CELL_H + 0.3],
                mode='lines', line=dict(color=BUS_BAR_COLOR, width=1),
                hoverinfo='skip',
            ))

    # ── 4. 电芯温度标注点（顶部小球） ──
    traces.append(go.Scatter3d(
        x=cell_x, y=cell_y, z=[CELL_H + 0.3]*len(cell_x),
        mode='markers+text',
        marker=dict(size=6, color=cell_temps, colorscale='Turbo',
                    cmin=t_min, cmax=t_max, colorbar=dict(
                        title=dict(text="°C", font=dict(color='#ccc')),
                        x=0.88, len=0.5, thickness=12,
                        tickfont=dict(color='#ccc'),
                    )),
        text=[f"{t:.1f}°C" for t in cell_temps],
        textfont=dict(size=9, color='white'),
        textposition='top center',
        hovertext=cell_labels,
        hoverinfo='text',
        name='Cells'
    ))

    # ── 5. 外壳线框 ──
    case_x = [-0.5, total_x + 0.5, total_x + 0.5, -0.5, -0.5,
              -0.5, total_x + 0.5, total_x + 0.5, -0.5, -0.5,
              total_x + 0.5, total_x + 0.5, total_x + 0.5, total_x + 0.5, -0.5, -0.5]
    case_y = [-0.5, -0.5, total_y + 0.5, total_y + 0.5, -0.5,
              -0.5, -0.5, total_y + 0.5, total_y + 0.5, -0.5,
              -0.5, total_y + 0.5, total_y + 0.5, -0.5, -0.5, total_y + 0.5]
    case_z = [COOLING_H - 0.15, COOLING_H - 0.15, COOLING_H - 0.15, COOLING_H - 0.15, COOLING_H - 0.15,
              CELL_H + 0.8, CELL_H + 0.8, CELL_H + 0.8, CELL_H + 0.8, CELL_H + 0.8,
              CELL_H + 0.8, CELL_H + 0.8, COOLING_H - 0.15, COOLING_H - 0.15, COOLING_H - 0.15, CELL_H + 0.8]
    traces.append(go.Scatter3d(
        x=case_x, y=case_y, z=case_z, mode='lines',
        line=dict(color=CASING_COLOR, width=2), name='Pack Casing', hoverinfo='skip'
    ))

    # ── 冷却液流向箭头 ──
    mid_y = total_y / 2
    traces.append(go.Scatter3d(
        x=[-0.6], y=[-0.6], z=[COOLING_H],
        mode='markers+text',
        marker=dict(size=8, symbol='diamond', color='#00e676'),
        text=["INLET"], textfont=dict(color='#00e676', size=11),
        hoverinfo='skip', name='Coolant In'
    ))
    traces.append(go.Scatter3d(
        x=[total_x + 0.6], y=[total_y + 0.6], z=[COOLING_H],
        mode='markers+text',
        marker=dict(size=8, symbol='diamond', color='#ff5252'),
        text=["OUTLET"], textfont=dict(color='#ff5252', size=11),
        hoverinfo='skip', name='Coolant Out'
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"Max: {t_max:.1f}°C  |  Min: {t_min:.1f}°C  |  ΔT: {t_max - t_min:.2f}°C",
            font=dict(color='#ccc', size=14)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[COOLING_H - 0.5, CELL_H + 1.2]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            aspectmode='data',
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Frame {selected_idx+1}/{total_frames}  |  {Ns}×{Np} cells  |  "
               f"Coolant: {COOLANT_COLOR} inlet → outlet")


def render_3d_aging_map_view(soh_matrix, Ns: int, Np: int):
    """工业级 SOH 空间分布 — 立体电芯 + 最差电芯高亮"""
    st.subheader(" SOH Health Distribution Map")

    if soh_matrix is None or len(soh_matrix) == 0:
        st.info("SOH 空间数据不可用，请使用新版 FMU")
        return

    S = np.array(soh_matrix) * 100  # → 百分比
    soh_min, soh_max = float(np.min(S)), float(np.max(S))

    traces = []
    cell_labels = []

    # ── 电芯立体盒 ──
    for s in range(Ns):
        module_idx = s // 8
        x_off = module_idx * MODULE_GAP
        for p in range(Np):
            cx = p * (CELL_W + CELL_GAP) + x_off
            cy = s * (CELL_W + CELL_GAP)
            val = S[s, p]
            cell_labels.append(f"S{s+1}P{p+1}: SOH {val:.2f}%")

            xf, yf, zf = _build_cell_mesh(cx, cy, 0, CELL_W, CELL_W, CELL_H * 0.7)
            traces.append(go.Mesh3d(
                x=xf, y=yf, z=zf,
                intensity=[val]*len(xf),
                colorscale='RdYlGn',
                intensitymode='vertex',
                cmin=soh_min, cmax=soh_max,
                showscale=False,
                hoverinfo='skip',
            ))

    # ── 顶部小球（带色标） ──
    cell_x, cell_y, cell_z_vals, cell_soh = [], [], [], []
    worst_idx = np.unravel_index(np.argmin(S), S.shape)
    for s in range(Ns):
        module_idx = s // 8
        x_off = module_idx * MODULE_GAP
        for p in range(Np):
            cell_x.append(p * (CELL_W + CELL_GAP) + CELL_W/2 + x_off)
            cell_y.append(s * (CELL_W + CELL_GAP) + CELL_W/2)
            cell_z_vals.append(CELL_H * 0.7 + 0.15)
            cell_soh.append(S[s, p])

    traces.append(go.Scatter3d(
        x=cell_x, y=cell_y, z=cell_z_vals,
        mode='markers+text',
        marker=dict(size=5, color=cell_soh, colorscale='RdYlGn',
                    cmin=soh_min, cmax=soh_max,
                    colorbar=dict(
                        title=dict(text="SOH %", font=dict(color='#ccc')),
                        x=0.88, len=0.5, thickness=12,
                        tickfont=dict(color='#ccc'),
                        tickformat='.1f',
                    )),
        text=[f"{v:.2f}%" for v in cell_soh],
        textfont=dict(size=8, color='white'),
        textposition='top center',
        hovertext=cell_labels,
        hoverinfo='text',
        name='SOH'
    ))

    # ── 最差电芯高亮 ──
    ws, wp = worst_idx
    w_x = wp * (CELL_W + CELL_GAP) + CELL_W/2 + (ws // 8) * MODULE_GAP
    w_y = ws * (CELL_W + CELL_GAP) + CELL_W/2
    # 红色闪烁边框
    for dz in np.linspace(0, CELL_H * 0.7, 4):
        ring = np.linspace(0, 2*np.pi, 20)
        traces.append(go.Scatter3d(
            x=[w_x + 0.5*np.cos(a) for a in ring],
            y=[w_y + 0.5*np.sin(a) for a in ring],
            z=[dz]*20, mode='lines',
            line=dict(color='#ff1744', width=2),
            hoverinfo='skip', name='Worst Cell'
        ))

    # ── 底部基座 ──
    total_x = (Np * (CELL_W + CELL_GAP) - CELL_GAP) + (Ns // 8) * MODULE_GAP
    total_y = Ns * (CELL_W + CELL_GAP) - CELL_GAP
    base_z = [-0.2]*4
    traces.append(go.Mesh3d(
        x=[-0.3, total_x+0.3, total_x+0.3, -0.3],
        y=[-0.3, -0.3, total_y+0.3, total_y+0.3],
        z=base_z, color=GRID_COLOR, opacity=0.3,
        showscale=False, hoverinfo='skip', name='Base'
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"SOH Range: {soh_min:.2f}% ~ {soh_max:.2f}%  |  "
                 f"Worst: S{ws+1}P{wp+1} ({S[ws,wp]:.2f}%)",
            font=dict(color='#ccc', size=14)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode='data',
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.error(f" Worst Cell: S{ws+1}P{wp+1} (SOH={S[ws,wp]:.2f}%) "
             f"— This cell dictates the entire pack End-of-Life.")
