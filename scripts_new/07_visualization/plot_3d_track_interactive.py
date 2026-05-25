# -*- coding: utf-8 -*-
"""
宁波地铁5号线 —— 交互式三维轨道线形 (浏览器可旋转/缩放)
"""
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.physics.train_simu import TrainTheoreticalEnergyModel

import plotly.graph_objects as go

model = TrainTheoreticalEnergyModel()
g_locs, g_vals = model.real_grad_locs, model.real_grad_vals
c_locs, c_vals = model.real_curve_locs, model.real_curve_vals

STATIONS = [
    (0, "布政"), (1450, "张家潭"), (3503, "同德路"), (4854, "石碶"),
    (5694, "雅渡"), (7583, "庙堰"), (8686, "钟公庙"), (9685, "鄞州区政府"),
    (11524, "钱湖南路"), (12711, "南高教园区"), (14185, "下应路"), (15418, "大洋江"),
    (16390, "泗港"), (18215, "曹隘"), (19542, "柳隘"), (20538, "海晏北路"),
    (21415, "民安东路"), (22151, "会展中心"), (23727, "院士路"), (24737, "盎孟港"),
    (26643, "三官堂"), (27740, "兴庄路"), (28714, "兴海南路"), (29560, "梅堰"),
    (33294, "永茂路"), (35458, "镇海大道"), (37078, "骆驼桥"),
]

# ====== 坐标计算 ======
step = 2.0
total = g_locs[-1]
pos = np.arange(0, total, step)

grad = np.interp(pos, g_locs, g_vals)
radius = np.interp(pos, c_locs, c_vals)
curv = np.where(np.abs(radius) >= 2999, 0.0, 1.0 / radius)

z = np.cumsum(grad / 1000.0 * step)
z -= z.min()

heading = np.cumsum(curv * step)
heading -= heading.mean()
y = np.cumsum(np.sin(heading) * step)
y -= y.mean()
x = pos

# 每50m采样一条线（降采样让HTML轻量）
skip = 25
xp, yp, zp = x[::skip], y[::skip], z[::skip]

# ====== 交互式 3D 图 ======
fig = go.Figure()

# 轨道主线
fig.add_trace(go.Scatter3d(
    x=xp, y=yp, z=zp,
    mode='lines',
    line=dict(color='#1a5276', width=4),
    name='轨道线形',
    hoverinfo='text',
    hovertext=[f'里程: {px:.0f} m<br>高程: {pz:.1f} m' for px, pz in zip(xp, zp)]
))

# 站点标记
st_x, st_y, st_z = [], [], []
st_names, st_labels = [], []
for i, (mile, name) in enumerate(STATIONS):
    idx = np.argmin(np.abs(x - mile))
    st_x.append(x[idx])
    st_y.append(y[idx])
    st_z.append(z[idx])
    st_names.append(name)
    is_end = (i == 0 or i == len(STATIONS) - 1)
    st_labels.append(f'<b>{name}</b>' if is_end else name)

fig.add_trace(go.Scatter3d(
    x=st_x, y=st_y, z=st_z,
    mode='markers+text',
    marker=dict(
        size=[10 if i in (0, len(STATIONS)-1) else 5 for i in range(len(STATIONS))],
        color=['#c0392b' if i in (0, len(STATIONS)-1) else '#2c3e50' for i in range(len(STATIONS))],
        line=dict(color='white', width=0.5)
    ),
    text=st_labels,
    textposition='top center',
    textfont=dict(size=[11 if i in (0, len(STATIONS)-1) else 8 for i in range(len(STATIONS))],
                  family='Microsoft YaHei'),
    name='车站',
    hoverinfo='text',
    hovertext=[f'<b>{n}</b><br>里程: {st_x[i]:.0f} m<br>高程: {st_z[i]:.1f} m'
               for i, n in enumerate(st_names)]
))

# 底部投影线
fig.add_trace(go.Scatter3d(
    x=xp, y=np.full_like(yp, min(st_y) - 5), z=zp,
    mode='lines',
    line=dict(color='#bdc3c7', width=1, dash='dot'),
    name='侧向投影',
    showlegend=False
))

fig.update_layout(
    title=dict(
        text='宁波地铁5号线 · 三维轨道线形<br><sup>鼠标拖拽旋转 | 滚轮缩放 | 右键平移 | 总长 37.1 km · 26 区间</sup>',
        font=dict(family='Microsoft YaHei', size=18)
    ),
    scene=dict(
        xaxis_title='里程 X (m)',
        yaxis_title='侧向偏移 Y (m)',
        zaxis_title='高程 Z (m)',
        aspectmode='manual',
        aspectratio=dict(x=1, y=0.08, z=0.02),
        camera=dict(eye=dict(x=-0.8, y=-1.3, z=0.6))
    ),
    font=dict(family='Microsoft YaHei'),
    margin=dict(l=0, r=0, t=60, b=0),
    hovermode='closest'
)

out = PROJECT_ROOT / "output" / "figure" / "line5_3d_interactive.html"
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("用浏览器打开即可旋转/缩放/悬停查看信息")
