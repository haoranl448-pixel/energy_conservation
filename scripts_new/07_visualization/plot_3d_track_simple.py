# -*- coding: utf-8 -*-
"""
宁波地铁5号线三维轨道线形
X = 里程 (m)
Y = 水平偏移 (m), 由曲率积分得出
Z = 高程 (m),   由坡度积分得出
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.physics.train_simu import TrainTheoreticalEnergyModel

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

# ===================== 坐标计算 =====================
step = 2.0
total = g_locs[-1]
pos = np.arange(0, total, step)
n = len(pos)

grad = np.interp(pos, g_locs, g_vals)          # ‰
radius = np.interp(pos, c_locs, c_vals)         # m

# Z: 高程 = ∫坡度·ds
z = np.cumsum(grad / 1000.0 * step)
z -= z.min()  # 起点归零

# Y: 水平偏移 = ∫∫曲率·ds²
curvature = np.where(np.abs(radius) >= 2999, 0.0, 1.0 / radius)
heading = np.cumsum(curvature * step)
heading -= heading.mean()
y = np.cumsum(np.sin(heading) * step)
y -= y.mean()

# X: 里程
x = pos

# 站索引
st_idx = [np.argmin(np.abs(x - s[0])) for s in STATIONS]

# ===================== 绘图 =====================
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(24, 8), subplot_kw={'projection': '3d'})

# 主轨道线
ax.plot(x, y, z, color='#1a5276', linewidth=1.5, zorder=5)

# 站点标记
for i, (idx, (_, name)) in enumerate(zip(st_idx, STATIONS)):
    is_end = (i == 0 or i == len(STATIONS) - 1)
    ax.scatter(x[idx], y[idx], z[idx],
               s=100 if is_end else 35,
               c='#c0392b' if is_end else '#2c3e50',
               edgecolors='white', linewidth=0.4, zorder=10)
    ax.text(x[idx], y[idx], z[idx] + 1.8, name,
            fontsize=6.5 if not is_end else 8.5,
            ha='center', va='bottom', fontweight='bold' if is_end else 'normal')

# 底部投影（辅助感知空间）
ax.plot(x, np.full_like(y, y.min() - 2), z, color='#bdc3c7',
        linewidth=0.3, alpha=0.4, zorder=1)

ax.set_xlabel('里程 X (m)', fontsize=10)
ax.set_ylabel('侧向偏移 Y (m)', fontsize=10)
ax.set_zlabel('高程 Z (m)', fontsize=10)
ax.set_title('宁波地铁5号线 · 三维轨道线形\n总长 37.1 km · 26 站 · XY里程-高程',
             fontsize=14, fontweight='bold')
ax.view_init(elev=20, azim=-50)
ax.tick_params(labelsize=7)

plt.tight_layout()

out = PROJECT_ROOT / "output" / "figure" / "line5_3d_track.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
