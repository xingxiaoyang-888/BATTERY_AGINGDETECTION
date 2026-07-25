"""电池包三维热场与健康空间诊断视图。"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from models.data_structures import TimeSeriesData


CELL_W = 0.82
CELL_D = 0.66
CELL_H = 1.18
CELL_GAP_X = 0.18
CELL_GAP_Y = 0.18
MODULE_GAP_X = 0.72
MODULE_GAP_Y = 0.72
SERIES_PER_MODULE = 8
SERIES_COLUMNS = 4
COOLING_Z = -0.24

THERMAL_SCALE = [
    [0.00, "#216869"],
    [0.35, "#45d2be"],
    [0.62, "#f1b95b"],
    [0.82, "#f28a55"],
    [1.00, "#ff5f6d"],
]
SOH_SCALE = [
    [0.00, "#ff5f6d"],
    [0.42, "#f1b95b"],
    [0.72, "#88cf8f"],
    [1.00, "#45d2be"],
]


def _cell_layout(ns: int, np_: int):
    """将串并联拓扑映射到紧凑的模组化物理排布。"""
    module_count = max(1, math.ceil(ns / SERIES_PER_MODULE))
    series_columns = min(SERIES_COLUMNS, ns)
    module_rows_of_cells = math.ceil(min(SERIES_PER_MODULE, ns) / series_columns)
    pitch_x = CELL_W + CELL_GAP_X
    pitch_y = CELL_D + CELL_GAP_Y
    module_width = series_columns * pitch_x
    module_depth = module_rows_of_cells * np_ * pitch_y
    module_columns = max(
        1,
        math.ceil(math.sqrt(module_count * module_depth / max(module_width, 0.1))),
    )

    cells = []
    for s in range(ns):
        module_index = s // SERIES_PER_MODULE
        module_column = module_index % module_columns
        module_row = module_index // module_columns
        series_slot = s % SERIES_PER_MODULE
        origin_x = module_column * (module_width + MODULE_GAP_X)
        origin_y = module_row * (module_depth + MODULE_GAP_Y)
        for p in range(np_):
            cells.append({
                "s": s,
                "p": p,
                "module": module_index,
                "x": origin_x + (series_slot % series_columns) * pitch_x,
                "y": origin_y + ((series_slot // series_columns) * np_ + p) * pitch_y,
            })
    return cells


def _matrix_or_error(matrix, ns: int, np_: int, name: str):
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        st.error(f"{name}矩阵无法解析，请检查后端空间数据。")
        return None
    if values.shape != (ns, np_):
        st.error(f"{name}矩阵维度为 {values.shape}，与当前 {ns}S{np_}P 拓扑不一致。")
        return None
    if not np.all(np.isfinite(values)):
        st.error(f"{name}矩阵包含无效数值，暂时无法渲染三维视图。")
        return None
    return values


def _value_range(values, minimum_span):
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min < minimum_span:
        midpoint = (value_min + value_max) / 2
        return midpoint - minimum_span / 2, midpoint + minimum_span / 2
    return value_min, value_max


def _append_box(vertices, triangles, intensities, x, y, z, width, depth, height, value):
    base = len(vertices)
    vertices.extend([
        (x - width / 2, y - depth / 2, z),
        (x + width / 2, y - depth / 2, z),
        (x + width / 2, y + depth / 2, z),
        (x - width / 2, y + depth / 2, z),
        (x - width / 2, y - depth / 2, z + height),
        (x + width / 2, y - depth / 2, z + height),
        (x + width / 2, y + depth / 2, z + height),
        (x - width / 2, y + depth / 2, z + height),
    ])
    triangles.extend([
        (base + 0, base + 2, base + 1), (base + 0, base + 3, base + 2),
        (base + 4, base + 5, base + 6), (base + 4, base + 6, base + 7),
        (base + 0, base + 1, base + 5), (base + 0, base + 5, base + 4),
        (base + 1, base + 2, base + 6), (base + 1, base + 6, base + 5),
        (base + 2, base + 3, base + 7), (base + 2, base + 7, base + 6),
        (base + 3, base + 0, base + 4), (base + 3, base + 4, base + 7),
    ])
    intensities.extend([value] * 8)


def _cell_mesh(cells, values, colorscale, cmin, cmax, colorbar_title):
    vertices, triangles, intensities = [], [], []
    for cell in cells:
        value = float(values[cell["s"], cell["p"]])
        _append_box(
            vertices,
            triangles,
            intensities,
            cell["x"],
            cell["y"],
            0,
            CELL_W,
            CELL_D,
            CELL_H,
            value,
        )
    x, y, z = zip(*vertices)
    i, j, k = zip(*triangles)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        intensity=intensities,
        intensitymode="vertex",
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        flatshading=True,
        lighting=dict(ambient=0.68, diffuse=0.72, roughness=0.82, specular=0.12),
        lightposition=dict(x=100, y=180, z=240),
        colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color="#9db1bb", size=11)),
            thickness=12,
            len=0.56,
            x=0.98,
            tickfont=dict(color="#9db1bb", size=10),
            outlinecolor="rgba(126,176,198,.22)",
        ),
        hoverinfo="skip",
        showscale=True,
        name="Cells",
    )


def _bounds(cells):
    x_values = [cell["x"] for cell in cells]
    y_values = [cell["y"] for cell in cells]
    return (
        min(x_values) - CELL_W / 2,
        max(x_values) + CELL_W / 2,
        min(y_values) - CELL_D / 2,
        max(y_values) + CELL_D / 2,
    )


def _plate_trace(bounds):
    x0, x1, y0, y1 = bounds
    vertices, triangles, intensities = [], [], []
    _append_box(
        vertices,
        triangles,
        intensities,
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        COOLING_Z,
        x1 - x0 + 0.7,
        y1 - y0 + 0.7,
        0.12,
        0,
    )
    x, y, z = zip(*vertices)
    i, j, k = zip(*triangles)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="#197f92",
        opacity=0.42,
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.5, roughness=0.9),
        hoverinfo="skip",
        showscale=False,
        name="液冷板",
    )


def _coolant_trace(bounds):
    x0, x1, y0, y1 = bounds
    channel_count = min(6, max(3, math.ceil((y1 - y0) / 3)))
    channel_y = np.linspace(y0, y1, channel_count)
    line_x, line_y, line_z = [], [], []
    for index, y in enumerate(channel_y):
        direction = (x0 - 0.25, x1 + 0.25) if index % 2 == 0 else (x1 + 0.25, x0 - 0.25)
        line_x.extend([direction[0], direction[1], None])
        line_y.extend([y, y, None])
        line_z.extend([COOLING_Z + 0.13, COOLING_Z + 0.13, None])
    return go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(color="#45d2be", width=4),
        opacity=0.75,
        hoverinfo="skip",
        name="冷却流道",
    )


def _busbar_trace(cells, ns: int):
    line_x, line_y, line_z = [], [], []
    for s in range(ns):
        group = [cell for cell in cells if cell["s"] == s]
        if len(group) < 2:
            continue
        group.sort(key=lambda cell: cell["p"])
        line_x.extend([cell["x"] for cell in group] + [None])
        line_y.extend([cell["y"] for cell in group] + [None])
        line_z.extend([CELL_H + 0.055] * len(group) + [None])
    return go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(color="#c7d4d9", width=5),
        opacity=0.76,
        hoverinfo="skip",
        name="并联母排",
    )


def _contact_trace(cells, values, value_name, suffix, colorscale, cmin, cmax):
    customdata = [
        [cell["s"] + 1, cell["p"] + 1, cell["module"] + 1, float(values[cell["s"], cell["p"]])]
        for cell in cells
    ]
    return go.Scatter3d(
        x=[cell["x"] for cell in cells],
        y=[cell["y"] for cell in cells],
        z=[CELL_H + 0.065] * len(cells),
        mode="markers",
        marker=dict(
            size=4,
            color=[item[3] for item in customdata],
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            line=dict(color="rgba(232,244,247,.6)", width=0.7),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>S%{customdata[0]} · P%{customdata[1]}</b><br>"
            "模组 %{customdata[2]}<br>"
            f"{value_name} %{{customdata[3]:.2f}}{suffix}<extra></extra>"
        ),
        name="电芯触点",
        showlegend=False,
    )


def _cell_body_trace(cells, values, value_name, suffix, colorscale, cmin, cmax):
    """为不完整支持Mesh3d的浏览器提供始终可见的电芯实体层。"""
    cell_count = len(cells)
    marker_size = max(7, min(26, 72 / math.sqrt(max(cell_count, 1))))
    customdata = [
        [cell["s"] + 1, cell["p"] + 1, cell["module"] + 1, float(values[cell["s"], cell["p"]])]
        for cell in cells
    ]
    return go.Scatter3d(
        x=[cell["x"] for cell in cells],
        y=[cell["y"] for cell in cells],
        # 放在电芯顶面之上，避免部分WebGL实现仅写入Mesh3d深度而不绘制面片时被遮挡。
        z=[CELL_H + 0.08] * cell_count,
        mode="markers",
        marker=dict(
            size=marker_size,
            symbol="square",
            color=[item[3] for item in customdata],
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            opacity=0.96,
            line=dict(color="rgba(217,235,240,.82)", width=1.2),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>S%{customdata[0]} · P%{customdata[1]}</b><br>"
            "模组 %{customdata[2]}<br>"
            f"{value_name} %{{customdata[3]:.2f}}{suffix}<extra></extra>"
        ),
        name="电芯实体",
        showlegend=False,
    )


def _focus_trace(cell, value, label, suffix, color):
    return go.Scatter3d(
        x=[cell["x"]],
        y=[cell["y"]],
        z=[CELL_H + 0.26],
        mode="markers+text",
        marker=dict(
            size=10,
            symbol="diamond",
            color=color,
            line=dict(color="#f4fbfc", width=2),
        ),
        text=[f"{label}<br>S{cell['s'] + 1}P{cell['p'] + 1}"],
        textposition="top center",
        textfont=dict(color="#e9f3f6", size=10),
        hovertemplate=f"<b>{label}</b><br>{value:.2f}{suffix}<extra></extra>",
        showlegend=False,
    )


def _scene_layout(bounds, title):
    x0, x1, y0, y1 = bounds
    width = max(x1 - x0, 1)
    depth = max(y1 - y0, 1)
    return dict(
        template="plotly_dark",
        height=620,
        margin=dict(l=0, r=12, t=52, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.01, font=dict(color="#c9d8de", size=13)),
        scene=dict(
            xaxis=dict(visible=False, range=[x0 - 0.8, x1 + 0.8]),
            yaxis=dict(visible=False, range=[y0 - 0.8, y1 + 0.8]),
            zaxis=dict(visible=False, range=[COOLING_Z - 0.18, CELL_H + 0.72]),
            bgcolor="rgba(5,15,22,.52)",
            aspectmode="manual",
            aspectratio=dict(x=width / max(width, depth), y=depth / max(width, depth), z=0.24),
            camera=dict(eye=dict(x=1.45, y=1.55, z=1.15), up=dict(x=0, y=0, z=1)),
        ),
        showlegend=False,
        font=dict(color="#9db1bb", size=11),
        hoverlabel=dict(bgcolor="#0b1a24", bordercolor="#45d2be", font=dict(color="#e9f3f6")),
        uirevision="battery-spatial-camera",
    )


def _render_summary(items):
    content = "".join(
        f'<div class="spatial-stat"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>'
        for label, value, detail in items
    )
    st.markdown(f'<div class="spatial-summary">{content}</div>', unsafe_allow_html=True)


def render_3d_pack_thermal_view(ts_data: TimeSeriesData, Ns: int, Np: int):
    """渲染热场数字孪生、时间帧选择和热点定位。"""
    frames = getattr(ts_data, "temp_matrix_frames", None)
    if not frames:
        st.info("当前结果没有空间热场数据，请确认所选FMU支持电芯温度矩阵输出。")
        return

    total_frames = len(frames)
    toolbar_left, toolbar_right = st.columns([1.2, 1], gap="large")
    with toolbar_left:
        frame_index = st.slider(
            "热场时间帧",
            min_value=0,
            max_value=total_frames - 1,
            value=total_frames - 1,
            format="帧 %d",
            key="thermal_frame_index",
        )
    matrix = _matrix_or_error(frames[frame_index], Ns, Np, "温度")
    if matrix is None:
        return

    cells = _cell_layout(Ns, Np)
    hottest_flat = int(np.argmax(matrix))
    with toolbar_right:
        selected_flat = st.selectbox(
            "重点电芯",
            options=list(range(Ns * Np)),
            index=hottest_flat,
            format_func=lambda index: (
                f"S{index // Np + 1}P{index % Np + 1} · "
                f"{matrix[index // Np, index % Np]:.2f} °C"
            ),
            key="thermal_focus_cell",
        )

    selected_cell = cells[selected_flat]
    selected_value = float(matrix[selected_cell["s"], selected_cell["p"]])
    t_min, t_max = float(np.min(matrix)), float(np.max(matrix))
    t_mean = float(np.mean(matrix))
    time_values = getattr(ts_data, "timestamps", None)
    time_label = (
        f"{time_values[frame_index]:.1f} s"
        if time_values and frame_index < len(time_values)
        else f"{frame_index + 1} / {total_frames}"
    )
    _render_summary([
        ("MAX TEMPERATURE", f"{t_max:.2f} °C", f"S{hottest_flat // Np + 1}P{hottest_flat % Np + 1}"),
        ("PACK AVERAGE", f"{t_mean:.2f} °C", f"{Ns * Np} cells"),
        ("TEMPERATURE SPREAD", f"{t_max - t_min:.2f} °C", "max - min"),
        ("ACTIVE FRAME", time_label, f"frame {frame_index + 1}"),
    ])

    cmin, cmax = _value_range(matrix, 0.5)
    bounds = _bounds(cells)
    figure = go.Figure()
    figure.add_trace(_plate_trace(bounds))
    figure.add_trace(_coolant_trace(bounds))
    figure.add_trace(_cell_mesh(cells, matrix, THERMAL_SCALE, cmin, cmax, "温度 / °C"))
    figure.add_trace(_cell_body_trace(cells, matrix, "温度", " °C", THERMAL_SCALE, cmin, cmax))
    figure.add_trace(_busbar_trace(cells, Ns))
    figure.add_trace(_contact_trace(cells, matrix, "温度", " °C", THERMAL_SCALE, cmin, cmax))
    figure.add_trace(_focus_trace(selected_cell, selected_value, "重点电芯", " °C", "#ffdd7a"))
    figure.update_layout(**_scene_layout(bounds, f"3D热场 · {Ns}S{Np}P · 模组化排布"))
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        key="thermal_pack_3d",
    )


def render_3d_aging_map_view(soh_matrix, Ns: int, Np: int):
    """渲染SOH空间分布并标识限制整包寿命的最差电芯。"""
    if soh_matrix is None or len(soh_matrix) == 0:
        st.info("当前结果没有SOH空间矩阵，请确认所选FMU支持单体健康状态输出。")
        return
    matrix = _matrix_or_error(soh_matrix, Ns, Np, "SOH")
    if matrix is None:
        return
    if float(np.max(matrix)) <= 1.5:
        matrix = matrix * 100

    cells = _cell_layout(Ns, Np)
    worst_flat = int(np.argmin(matrix))
    worst_cell = cells[worst_flat]
    worst_value = float(matrix[worst_cell["s"], worst_cell["p"]])
    soh_min, soh_max = float(np.min(matrix)), float(np.max(matrix))
    _render_summary([
        ("PACK-LIMITING SOH", f"{soh_min:.3f}%", f"S{worst_flat // Np + 1}P{worst_flat % Np + 1}"),
        ("PACK AVERAGE SOH", f"{float(np.mean(matrix)):.3f}%", f"{Ns * Np} cells"),
        ("SOH SPREAD", f"{soh_max - soh_min:.4f}%", "cell imbalance"),
        ("HEALTH BAND", "NORMAL" if soh_min >= 80 else "EOL RISK", "80% threshold"),
    ])

    cmin, cmax = _value_range(matrix, 0.2)
    bounds = _bounds(cells)
    figure = go.Figure()
    figure.add_trace(_plate_trace(bounds))
    figure.add_trace(_cell_mesh(cells, matrix, SOH_SCALE, cmin, cmax, "SOH / %"))
    figure.add_trace(_cell_body_trace(cells, matrix, "SOH", "%", SOH_SCALE, cmin, cmax))
    figure.add_trace(_busbar_trace(cells, Ns))
    figure.add_trace(_contact_trace(cells, matrix, "SOH", "%", SOH_SCALE, cmin, cmax))
    figure.add_trace(_focus_trace(worst_cell, worst_value, "最差电芯", "%", "#ff5f6d"))
    figure.update_layout(**_scene_layout(bounds, f"SOH空间分布 · {Ns}S{Np}P · 最差电芯定位"))
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        key="soh_pack_3d",
    )
    if worst_value < 80:
        st.error(f"S{worst_cell['s'] + 1}P{worst_cell['p'] + 1} 已低于80% EOL阈值，需要安排复测或维护。")
    else:
        st.caption(
            f"整包寿命受 S{worst_cell['s'] + 1}P{worst_cell['p'] + 1} 限制；"
            f"当前最低SOH为 {worst_value:.3f}%。"
        )
