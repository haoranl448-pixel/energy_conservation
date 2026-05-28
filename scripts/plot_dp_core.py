# -*- coding: utf-8 -*-
"""
DP动态规划 核心逻辑流程图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis('off')

# 配色
C = {
    'start':   '#27ae60',   # 起始/结束 绿色
    'init':    '#d6eaf8',   # 初始化 浅蓝
    'loop':    '#fdebd0',   # 循环体 浅橙
    'judge':   '#f9e79f',   # 判断 黄
    'update':  '#d5f5e3',   # 更新 浅绿
    'skip':    '#e5e7e9',   # 跳过 浅灰
    'result':  '#f5b7b1',   # 结果 浅红
    'arrow':   '#2c3e50',
    'loop_a':  '#e74c3c',   # 回路箭头颜色
    'border':  '#1f618d',
}

def box(ax, x, y, w, h, text, color, fs=8, bold=False, border=None):
    if border is None: border = C['border']
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                        facecolor=color, edgecolor=border, linewidth=1.2, alpha=0.9)
    ax.add_patch(b)
    wt = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight=wt, color='#1a1a1a')

def diamond(ax, cx, cy, w, h, text, color, fs=7.5):
    """菱形判断框"""
    points = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    diamond = plt.Polygon(points, facecolor=color, edgecolor=C['border'], linewidth=1.2, alpha=0.9)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='#1a1a1a')

def arrow(ax, x1, y1, x2, y2, color=None, lw=1.5):
    if color is None: color = C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, connectionstyle='arc3,rad=0'))

def arrow_loop(ax, x1, y1, x2, y2, rad=0.4, color=None, lw=1.8):
    if color is None: color = C['loop_a']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, connectionstyle=f'arc3,rad={rad}'))

def label(ax, x, y, text, fs=7.5, color=None, bold=False, ha='center'):
    if color is None: color = C['arrow']
    wt = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fs, color=color, fontweight=wt, ha=ha, va='center')

# ================================================================
# 标题
# ================================================================
ax.text(9, 21.4, 'DP (Dynamic Programming) 排图优化 — 核心算法流程',
        ha='center', fontsize=16, fontweight='bold', color='#1a5276')

# ================================================================
# START
# ================================================================
box(ax, 6.5, 20.2, 5, 0.7, 'START', C['start'], 10, bold=True)
arrow(ax, 9, 20.2, 9, 19.6)

# ================================================================
# 初始化
# ================================================================
box(ax, 5, 18.5, 8, 0.8, '初始化\n'
    'dp[0] = 0         累计时间为 0 时累计能耗为 0\n'
    'max_run_time = T_TARGET - sum(min_dwell)',
    C['init'], 8.5)

arrow(ax, 9, 18.5, 9, 17.8)

# ================================================================
# 循环 i (区间)
# ================================================================
box(ax, 5, 16.6, 8, 0.9, '外层循环: for 每个区间 i = 0, 1, 2 ... N-1\n'
    '  创建 new_dp = {},  new_path = {}\n'
    '  获取该区间所有可选等级 options[class2..class5]',
    C['loop'], 8.5, bold=True)

arrow(ax, 9, 16.6, 9, 15.9)

# ================================================================
# 循环 t_prev (状态)
# ================================================================
box(ax, 5, 14.8, 8, 0.8, '中层循环: for 每个上一状态 (t_prev, e_prev) in dp\n'
    '  t_prev = 到达区间 i-1 的累计时间\n'
    '  e_prev = 到达区间 i-1 的最小累计能耗',
    C['loop'], 8.5)

arrow(ax, 9, 14.8, 9, 14.0)

# ================================================================
# 循环 k (等级)
# ================================================================
box(ax, 5, 12.8, 8, 0.9, '内层循环: for 每个可选等级 k in options\n'
    '  k = class2 / class3 / class4 / class5\n'
    '  duration[k] = 等级k在该区间的运行时间\n'
    '  energy[k]   = 等级k在该区间的预测能耗',
    C['loop'], 8.5)

arrow(ax, 9, 12.8, 9, 12.0)

# ================================================================
# 计算 + 判断
# ================================================================
box(ax, 5, 10.7, 8, 1.0, '计算新状态:\n'
    '  t_new = t_prev + duration[k]\n'
    '  e_new = e_prev + energy[k]',
    C['init'], 8.5)

arrow(ax, 9, 10.7, 9, 9.8)

# 菱形判断
diamond(ax, 9, 8.8, 8, 1.6, 't_new <= max_run_time\n and\n e_new < dp[t_new] ?', C['judge'], 7.5)

# Yes 分支 (左)
arrow(ax, 5, 8.8, 3, 7.5)
box(ax, 0.5, 6.8, 4.5, 1.0, '更新状态:\n'
    '  new_dp[t_new] = e_new\n'
    '  new_path[t_new] = (t_prev, k, dur, energy)\n'
    '  (记录从哪个状态、用哪个等级到达)',
    C['update'], 7.5)

# No 分支 (右)
arrow(ax, 13, 8.8, 15, 7.5)
box(ax, 13, 6.8, 4.5, 1.0, '跳过 (t_new超限 或 非最优)\n'
    '  不更新 new_dp\n'
    '  继续下一个等级',
    C['skip'], 7.5)

# 汇合
arrow(ax, 2.75, 6.8, 4, 5.5)
arrow(ax, 15.25, 6.8, 14, 5.5)
arrow(ax, 4, 5.5, 9, 5.5); arrow(ax, 14, 5.5, 9, 5.5)

# ================================================================
# 三个循环的汇合 + 回路
# ================================================================
diamond(ax, 9, 4.5, 10, 1.8, '内层循环结束?\n'
    '  所有等级遍历完?\n'
    '    -> 回到中层循环 取下一个 (t_prev, e_prev)',
    C['judge'], 7)

# 回路: 没遍历完等级 -> 回到内层循环
arrow_loop(ax, 14, 5.4, 15, 13.2, rad=0.3, color=C['loop_a'], lw=2)
label(ax, 15.8, 9.3, '下个等级\nk = next', fs=7.5, color=C['loop_a'], bold=True, ha='left')

arrow(ax, 9, 3.6, 9, 3.0)

diamond(ax, 9, 2.2, 8, 1.5, '中层循环结束?\n  所有状态遍历完?\n    dp = new_dp\n    path = new_path',
    C['judge'], 7)

# 回路: 没遍历完状态 -> 回到中层
arrow_loop(ax, 5, 3.0, 4, 14.2, rad=-0.3, color='#e67e22', lw=2)
label(ax, 3.5, 8.6, '下个\n状态', fs=7.5, color='#e67e22', bold=True, ha='right')

arrow(ax, 9, 1.45, 9, 1.05)

# ================================================================
# 外层循环汇合
# ================================================================
diamond(ax, 9, 0.3, 9, 1.2, '外层循环结束?\n  所有区间遍历完?\n    进入结果选择',
    C['judge'], 7)

# 回路: 没遍历完区间 -> 回到外层
arrow_loop(ax, 4.5, 0.9, 3, 17.2, rad=-0.2, color='#e67e22', lw=2)
label(ax, 2.5, 9, '下个区间\ni = i+1', fs=8, color='#e67e22', bold=True, ha='right')

# ================================================================
# 右侧: 结果输出区
# ================================================================
arrow(ax, 13.5, 0.3, 15, 0.3)

box(ax, 15, 0.0, 2.5, 0.6, '结束循环',
    C['result'], 7.5, bold=True)

# 顶部注释: 输入/输出
box(ax, 0.5, 20.2, 5, 0.7,
    '输入:\n'
    '  dp[0]=0, options[0..N-1][1..5]\n'
    '  max_run_time', C['start'], 7)

box(ax, 13, 20.2, 4.5, 0.7,
    '输出:\n'
    '  dp[t] = 最小能耗\n'
    '  path[i][t] = 回溯路径', C['result'], 7)

# ================================================================
# 底部: 后续步骤
# ================================================================
box(ax, 15, -1.2, 2.5, 0.8,
    'Step 3\n选出最优\nfeasible中\n能耗最低',
    C['result'], 7.5, bold=True)

box(ax, 15, -2.7, 2.5, 0.8,
    'Step 4\n停站分配\ndwell弹性\n按slack比例',
    C['result'], 7.5)

box(ax, 15, -4.2, 2.5, 0.8,
    'Step 5\n回溯还原\n逐区间方案\n+ 节能对比',
    C['result'], 7.5)

arrow(ax, 16.25, -0.4, 16.25, -1.2)
arrow(ax, 16.25, -2.0, 16.25, -2.7)
arrow(ax, 16.25, -3.5, 16.25, -4.2)

# 图例
lx, ly = 0.5, -1.5
for i, (c, t) in enumerate([
    (C['start'], '开始/输入'), (C['init'], '计算'), (C['loop'], '循环'),
    (C['judge'], '判断'), (C['update'], '更新'), (C['result'], '结果'),
]):
    px = lx + i * 2.8
    ax.add_patch(plt.Rectangle((px, ly), 0.4, 0.3, facecolor=c, edgecolor='gray', linewidth=0.5))
    ax.text(px + 0.55, ly + 0.15, t, fontsize=7, va='center')

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "output" / "figure" / "dp_core_flow.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
plt.close()
