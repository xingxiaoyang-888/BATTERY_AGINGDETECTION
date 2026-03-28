# views/spatial_view.py
import plotly.graph_objects as go
import numpy as np

def render_3d_pack(t_max, t_min, series_num):
    """
    [View Component] 3D 电池模组空间孪生渲染器
    """
    # 1. 几何构建 (12列布局)
    cols = 12
    rows = int(np.ceil(series_num / cols))
    
    x_coords = []
    y_coords = []
    z_coords = []
    temps = []
    cell_ids = []
    
    # 2. 热场数据生成
    center_x, center_y = (cols-1)/2, (rows-1)/2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    count = 1
    for r in range(rows):
        for c in range(cols):
            if count > series_num: break
            
            x_coords.append(c)
            y_coords.append(r)
            z_coords.append(0)
            cell_ids.append(f"Cell #{count} (M{r+1})") 
            
            # 模拟热梯度
            dist = np.sqrt((c - center_x)**2 + (r - center_y)**2)
            factor = 1 - (dist / (max_dist + 0.1)) 
            cell_temp = t_min + (t_max - t_min) * (factor ** 1.5)
            cell_temp += np.random.normal(0, 0.2)
            temps.append(cell_temp)
            count += 1

    # === [图层 1] 电芯实体 (Cells) ===
    trace_cells = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=12,
            color=temps,                
            colorscale='Jet',
            cmin=20, cmax=60,           # 锁定工业标准温度范围
            opacity=0.9,
            symbol='square',
            colorbar=dict(
                title=dict(text="Temp (°C)", font=dict(color='white')), 
                x=0.85, 
                tickfont=dict(color='white'),
                len=0.8
            )
        ),
        text=cell_ids, 
        hovertemplate="<b>%{text}</b><br>Temp: %{marker.color:.1f}°C<br>Loc: [%{x}, %{y}]<extra></extra>"
    )
    
    data = [trace_cells]

    # === [图层 2] 电池包外壳线框 (Wireframe Casing) ===
    x_len = cols - 1
    y_len = rows - 1
    
    x_box = [0, x_len, x_len, 0, 0, 0, x_len, x_len, 0, 0, x_len, x_len, x_len, x_len, 0, 0]
    y_box = [0, 0, y_len, y_len, 0, 0, 0, y_len, y_len, 0, 0, y_len, y_len, 0, 0, y_len]
    z_box = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1]
    
    trace_case = go.Scatter3d(
        x=x_box, y=y_box, z=z_box,
        mode='lines',
        line=dict(color='#00f2ff', width=3),
        hoverinfo='skip',
        name='Pack Casing'
    )
    data.append(trace_case)

    # === [图层 3] 冷却流道标识 (Coolant Flow) ===
    trace_inlet = go.Scatter3d(
        x=[-1], y=[0], z=[0],
        mode='text+markers',
        marker=dict(symbol='diamond', size=10, color='#00e676'),
        text=["💧 INLET"],
        textfont=dict(color='#00e676', size=14), 
        hoverinfo='skip'
    )
    trace_outlet = go.Scatter3d(
        x=[x_len+1], y=[y_len], z=[0],
        mode='text+markers',
        marker=dict(symbol='diamond', size=10, color='#ff1744'),
        text=["🔥 OUTLET"],
        textfont=dict(color='#ff1744', size=14), 
        hoverinfo='skip'
    )
    data.append(trace_inlet)
    data.append(trace_outlet)

    # 4. 布局优化
    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text="🔋 Digital Twin Spatial View (Pack #01)", y=0.9),
        scene=dict(
            xaxis=dict(visible=False, showspikes=False),
            yaxis=dict(visible=False, showspikes=False),
            zaxis=dict(visible=False, showspikes=False),
            bgcolor='rgba(0,0,0,0)',
            aspectmode='data' 
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
        showlegend=False
    )
    
    fig.add_annotation(
        x=0.05, y=0.05, showarrow=False,
        text="Front (Connector Side)", font=dict(color="#00f2ff", size=10),
        xref="paper", yref="paper"
    )
    
    return fig

def render_3d_aging_map(avg_loss, series_num):
    """
    [新增] 3D SOH 损耗分布图 (SOH Spatial Distribution)
    原理：利用热梯度模拟不均匀老化 (中间热->老化快，边缘冷->老化慢)
    """
    # 1. 几何构建
    cols = 12
    rows = int(np.ceil(series_num / cols))
    
    x_coords = []
    y_coords = []
    z_coords = []
    aging_factors = [] # 相对老化因子
    cell_ids = []
    
    # 2. 模拟不均匀老化场
    # 假设中心电芯老化速度是平均值的 1.2 倍，边缘是 0.8 倍
    center_x, center_y = (cols-1)/2, (rows-1)/2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    count = 1
    for r in range(rows):
        for c in range(cols):
            if count > series_num: break
            
            x_coords.append(c)
            y_coords.append(r)
            z_coords.append(0)
            cell_ids.append(f"Cell #{count}")
            
            # 计算距离中心的归一化距离
            dist = np.sqrt((c - center_x)**2 + (r - center_y)**2)
            dist_norm = 1 - (dist / (max_dist + 0.1)) # 0(边缘) -> 1(中心)
            
            # 老化因子：中心老化快(Red)，边缘老化慢(Green)
            # base_loss * (0.8 ~ 1.2)
            factor = 0.8 + (0.4 * dist_norm**1.5) 
            aging_factors.append(factor)
            count += 1

    # 3. 渲染
    # 颜色映射：Green (低损耗) -> Red (高损耗)
    trace = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=12,
            symbol='square',
            color=aging_factors,
            colorscale='RdYlGn_r', # 翻转红绿轴：红=高损耗(Bad), 绿=低损耗(Good)
            opacity=0.9,
            colorbar=dict(
                # [关键修改] 增强字体可视性：亮白、加粗、加大
                title=dict(text="Aging Rate", font=dict(color='#ffffff', size=14, family="Arial Black")), 
                x=0.9, 
                tickfont=dict(color='#ffffff', size=12, family="Arial"), 
                len=0.8,
                tickvals=[0.8, 1.0, 1.2],
                ticktext=["Slow", "Avg", "Fast"]
            )
        ),
        text=cell_ids,
        hovertemplate="<b>%{text}</b><br>Aging Rate: x%{marker.color:.2f}<br>(Relative to Avg)<extra></extra>"
    )
    
    # 4. 线框外壳 (复用之前的逻辑)
    x_len, y_len = cols - 1, rows - 1
    x_box = [0, x_len, x_len, 0, 0, 0, x_len, x_len, 0, 0, x_len, x_len, x_len, x_len, 0, 0]
    y_box = [0, 0, y_len, y_len, 0, 0, 0, y_len, y_len, 0, 0, y_len, y_len, 0, 0, y_len]
    z_box = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1]
    
    trace_case = go.Scatter3d(
        x=x_box, y=y_box, z=z_box,
        mode='lines',
        line=dict(color='#ffab00', width=2), # 使用橙色区分于热力图
        hoverinfo='skip'
    )

    fig = go.Figure(data=[trace, trace_case])
    fig.update_layout(
        # [关键修改] 增强标题字体可视性
        title=dict(text="🔋 SOH 不均匀分布云图 (Non-uniform Aging)", y=0.9, font=dict(size=16, color='#ffffff')),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)', aspectmode='data'
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#ffffff"),
        showlegend=False
    )
    # 标注木桶效应
    fig.add_annotation(
        x=0.5, y=0.05, showarrow=False,
        text="⚠️ Red cells determine the Pack Lifetime (Barrel Effect)",
        font=dict(color="#ffab00", size=12, weight="bold"), xref="paper", yref="paper"
    )
    
    return fig