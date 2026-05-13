# -*- coding: utf-8 -*-
"""绘制宁波地铁5号线三维轨道线形图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.train_simu import TrainTheoreticalEnergyModel

# ===================== 1. 读取线路数据 =====================
model = TrainTheoreticalEnergyModel()
grad_locs = model.real_grad_locs   # 坡度位置 (m)
grad_vals = model.real_grad_vals   # 坡度值 (‰)
curve_locs = model.real_curve_locs # 曲率位置 (m)
curve_vals = model.real_curve_vals # 曲线半径 (m), 3000=直线

# ===================== 2. 26站信息 =====================
STATIONS = [
    (0,     "布政"),
    (1450,  "张家潭"),
    (3503,  "同德路"),
    (4854,  "石碶"),
    (5694,  "雅渡"),
    (7583,  "庙堰"),
    (8686,  "钟公庙"),
    (9685,  "鄞州区政府"),
    (11524, "钱湖南路"),
    (12711, "南高教园区"),
    (14185, "下应路"),
    (15418, "大洋江"),
    (16390, "泗港"),
    (18215, "曹隘"),
    (19542, "柳隘"),
    (20538, "海晏北路"),
    (21415, "民安东路"),
    (22151, "会展中心"),
    (23727, "院士路"),
    (24737, "盎孟港"),
    (26643, "三官堂"),
    (27740, "兴庄路"),
    (28714, "兴海南路"),
    (29560, "梅堰"),
    (33294, "永茂路"),
    (35458, "镇海大道"),
    (37078, "骆驼桥"),
]

# ===================== 3. 高程与平面偏移计算 =====================

def compute_track_geometry(grad_locs, grad_vals, curve_locs, curve_vals,
                            total_length, step=5.0):
    """从坡度/曲率离散点推导3D轨道线形"""
    positions = np.arange(0, total_length, step)
    n = len(positions)

    # 插值坡度(‰)和曲率半径(m)
    g = np.interp(positions, grad_locs, grad_vals)        # ‰
    r = np.interp(positions, curve_locs, curve_vals)       # m, 3000=直线

    # 曲率 → 水平转角变化率 (dθ/ds = 1/r), 直线时曲率≈0
    curvature = np.where(np.abs(r) >= 2999, 0.0, 1.0 / r)

    # 高程 Z = 累计坡度积分 (起点设0)
    dz = g / 1000.0 * step  # ‰ → 每步高差
    z = np.cumsum(dz)
    z -= z[0]

    # 水平方向角 (heading angle)
    dtheta = curvature * step
    heading = np.cumsum(dtheta)
    heading -= heading[0]

    # X, Y 坐标 (X沿主方向, Y侧向偏移)
    # 主方向沿轨道前进，Y反映左右弯曲
    dx = np.cos(heading) * step
    dy = np.sin(heading) * step
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    x -= x[0]
    y -= y[0]

    return positions, x, y, z, g, r


total_len = grad_locs[-1] + 100
positions, x, y, z, grad_at_pos, curve_at_pos = \
    compute_track_geometry(grad_locs, grad_vals, curve_locs, curve_vals, total_len)

# ===================== 4. 找车站索引 =====================
station_indices = []
station_names = []
for pos, name in STATIONS:
    idx = np.argmin(np.abs(positions - pos))
    station_indices.append(idx)
    station_names.append(name)

# ===================== 5. 绘图 =====================
fig = plt.figure(figsize=(28, 10))

# -------- 子图1: 3D轨道线形 --------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# 用曲率半径映射颜色 (弯道越急颜色越暖)
radius_norm = np.clip(curve_at_pos, 300, 3000)
colors = (3000 - radius_norm) / 2700  # 0=直线, 1=最急弯

# 分线段绘制, 按曲率着色
seg_size = 1
for i in range(0, len(positions) - seg_size, seg_size):
    seg = slice(i, i + seg_size + 1)
    c = colors[i]
    # 蓝(直线) → 红(急弯)
    r_col = c
    g_col = 1 - c * 0.6
    b_col = 1 - c
    ax1.plot(x[seg], y[seg], z[seg],
             color=(r_col, g_col, b_col), linewidth=1.2)

# 标注车站
for idx, name in zip(station_indices, station_names):
    ax1.scatter(x[idx], y[idx], z[idx], s=60, c='black', zorder=5)
    ax1.text(x[idx], y[idx], z[idx] + 2, name, fontsize=7,
             ha='center', va='bottom', fontfamily='Microsoft YaHei')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Elevation (m)')
ax1.set_title('Ningbo Metro Line 5 — 3D Track Alignment', fontsize=14, fontweight='bold')
ax1.view_init(elev=25, azim=-60)

# -------- 子图2: 坡度 + 曲率纵断面 --------
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(positions, grad_at_pos, color='#8B4513', linewidth=0.8)
ax2.fill_between(positions, 0, grad_at_pos,
                  where=(grad_at_pos >= 0), color='#CD853F', alpha=0.4, label='Uphill')
ax2.fill_between(positions, 0, grad_at_pos,
                  where=(grad_at_pos < 0), color='#4682B4', alpha=0.4, label='Downhill')
ax2.axhline(y=0, color='gray', linewidth=0.5)
for idx, name in zip(station_indices, station_names):
    ax2.axvline(x=positions[idx], color='gray', linewidth=0.4, linestyle='--')
    ax2.text(positions[idx], ax2.get_ylim()[1] * 0.95, name, fontsize=5,
             rotation=90, ha='right', va='top', fontfamily='Microsoft YaHei')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Gradient (‰)')
ax2.set_title('Vertical Profile — Gradient', fontsize=12)
ax2.legend(fontsize=8)

# -------- 子图3: 曲率半径 --------
ax3 = fig.add_subplot(2, 2, 4)
# 只画曲线段 (半径<3000)
curve_mask = curve_at_pos < 2900
ax3.scatter(positions[curve_mask], curve_at_pos[curve_mask],
            s=2, c='#D2691E', alpha=0.6)
ax3.axhline(y=3000, color='gray', linewidth=0.5, linestyle=':')
for idx, name in zip(station_indices, station_names):
    ax3.axvline(x=positions[idx], color='gray', linewidth=0.4, linestyle='--')
    ax3.text(positions[idx], ax3.get_ylim()[1] * 0.95, name, fontsize=5,
             rotation=90, ha='right', va='top', fontfamily='Microsoft YaHei')
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Curve Radius (m)')
ax3.set_title('Horizontal Alignment — Curve Radius', fontsize=12)

plt.suptitle('宁波地铁5号线 — 全线三维线形与纵断面', fontsize=16,
             fontfamily='Microsoft YaHei', fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_path = PROJECT_ROOT / "output" / "figure" / "line5_3d_track.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
