# -*- coding: utf-8 -*-
"""
DP排图优化 逻辑流程图 — 左右布局 + 循环回路
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(26, 18))
ax.set_xlim(0, 26)
ax.set_ylim(0, 18)
ax.axis('off')

C = {
    'input':   '#d5f5e3',
    'process': '#d6eaf8',
    'decision':'#fdebd0',
    'output':  '#f5b7b1',
    'data':    '#e8daef',
    'loop':    '#e74c3c',
    'arrow':   '#2c3e50',
    'border':  '#1f618d',
    'title':   '#1a5276',
}

def box(ax, x, y, w, h, text, color, fs=8.5, bold=False, border=None, align='center'):
    if border is None: border = C['border']
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                        facecolor=color, edgecolor=border, linewidth=1.5, alpha=0.92)
    ax.add_patch(b)
    wt = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha=align, va='center',
            fontsize=fs, fontweight=wt, color='#1a1a1a')

def data_box(ax, x, y, w, h, text, fs=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                        facecolor=C['data'], edgecolor='#8e44ad', linewidth=1,
                        linestyle='--', alpha=0.85)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color='#6c3483')

def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8):
    if color is None: color = C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def arrow_curved(ax, x1, y1, x2, y2, rad=0.3, color=None, lw=1.8):
    if color is None: color = C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))

def label(ax, x, y, text, fs=8, color=None, bold=False):
    if color is None: color = C['arrow']
    wt = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fs, color=color, fontweight=wt, ha='center', va='center')

# ================================================================
# 标题
# ================================================================
ax.text(13, 17.4, '宁波地铁5号线  DP排图优化逻辑流程 (ato_class_globall_v2)',
        ha='center', fontsize=18, fontweight='bold', color=C['title'])

# ================================================================
# 左侧列 (0~6.5): 数据输入 + 上游流水线
# ================================================================
ax.text(0.3, 16.4, '输入数据', fontsize=13, fontweight='bold', color='#1a1a1a')

data_box(ax, 0.3, 14.8, 5.2, 1.2,
         '能耗菜单 CSV\n区间x等级 -> 运行时间 + 能耗\n(ato_class_energy_menu*.csv)', 7.5)
data_box(ax, 0.3, 13.3, 5.2, 1.2,
         '历史数据 CSV\n每区间 -> 历史用时 + 历史能耗\n(full_line*_validation_results.csv)', 7.5)
data_box(ax, 0.3, 11.8, 5.2, 1.2,
         '区间参数 CSV\n每区间 -> MASS 载重均值\n(section_params_trip*.csv)', 7.5)
data_box(ax, 0.3, 10.3, 5.2, 1.2,
         '停站配置字典\n每站 -> nominal / min dwell\n(STATION_DWELL_CONFIG)', 7.5)

arrow(ax, 5.5, 14.5, 7.5, 15.8)
arrow(ax, 5.5, 14.0, 7.5, 15.5)
arrow(ax, 5.5, 13.0, 7.5, 14.5)
arrow(ax, 5.5, 11.5, 7.5, 13.5)

# 上游流水线 (左下)
ax.text(0.3, 8.8, '上游流水线', fontsize=13, fontweight='bold', color='#1a1a1a')

box(ax, 0.3, 7.0, 4.8, 1.2, '1. train_ATO_v8\n提取5级相位模板\n父基因 -> .pkl', C['input'], 8)
box(ax, 0.3, 5.3, 4.8, 1.2, '2. simulate_ATO_v8\nlam/theta 二维搜索\n生成 Class1-5 速度曲线', C['input'], 8)
box(ax, 0.3, 3.6, 4.8, 1.2, '3. ato_generated_results_energy\n物理模型 + AI残差\n-> 能耗菜单 CSV', C['input'], 8)
box(ax, 0.3, 1.9, 4.8, 1.2, '4. ato_class_globall_v2 (本图)\nDP排图 + 灵活停站\n-> 最优等级组合', C['process'], 8.5, bold=True)

arrow(ax, 2.7, 7.0, 2.7, 6.5)
arrow(ax, 2.7, 5.3, 2.7, 4.8)
arrow(ax, 2.7, 3.6, 2.7, 3.1)

# ================================================================
# 中间列 (7.5~15.5): DP核心
# ================================================================
ax.text(7.5, 16.4, 'DP 核心算法', fontsize=13, fontweight='bold', color='#1a1a1a')

# 全局参数
box(ax, 7.5, 15.8, 10.5, 0.7,
    '全局参数: T_TOTAL_TARGET=694.7s | SLACK=10s | ALLOWED_CLASSES=[class2-5] | MANUAL_CONSTRAINTS',
    C['process'], 8)

arrow(ax, 12.75, 15.8, 12.75, 15.0)

# Step 1 初始化
box(ax, 7.5, 14.3, 10.5, 0.6,
    'STEP 1  初始化:  dp[0] = 0  (累计时间0 -> 累计能耗0)', C['process'], 9, bold=True)
arrow(ax, 12.75, 14.3, 12.75, 13.6)

# Step 2 DP循环
box(ax, 7.5, 11.2, 10.5, 2.2,
    'STEP 2  DP 三重循环:\n'
    '  for 每个区间 i = 0 .. N-1:\n'
    '    for 每个上一状态 (t_prev, e_prev) in dp:\n'
    '      for 每个可选等级 k in [class2, class3, class4, class5]:\n'
    '        t_new = t_prev + 等级k的运行时间\n'
    '        e_new = e_prev + 等级k的预测能耗\n'
    '        if t_new <= max_run_time and e_new < dp[t_new]:\n'
    '            更新 dp[t_new] = e_new,  记录 path[i][t_new] = (prev_t, k, dur, energy)',
    C['decision'], 8)

# ===== 循环回路箭头 (关键!) =====
# 内层循环自环 (右上)
arrow_curved(ax, 18.3, 13.0, 19.5, 14.8, rad=0.7, color=C['loop'], lw=2.5)
label(ax, 19.7, 14.0, '遍历每个\n等级k', fs=7.5, color=C['loop'], bold=True)

# 区间递进回路 (右下)
arrow_curved(ax, 17.8, 11.7, 19.8, 10.5, rad=0.5, color='#e67e22', lw=2)
label(ax, 20.0, 11.2, '下一区间\n i <- i+1', fs=7.5, color='#e67e22', bold=True)

# 状态转移自环
arrow_curved(ax, 16.5, 11.2, 16.5, 9.8, rad=0.5, color='#27ae60', lw=2)
label(ax, 16.0, 10.5, '状态\n转移', fs=7, color='#27ae60', bold=True)

# 剪枝
box(ax, 7.5, 10.4, 10.5, 0.5,
    '剪枝规则: t_new > max_run_time -> 跳过 (不可行)', C['decision'], 7.5)

arrow(ax, 12.75, 10.4, 12.75, 9.7)

# Step 3 选最优
box(ax, 7.5, 8.9, 10.5, 0.6,
    'STEP 3  选出最优: 所有 feasible dp[t] 按能耗升序排列 -> best = 能耗最低状态',
    C['process'], 9, bold=True)

arrow(ax, 12.75, 8.9, 12.75, 8.2)

# Step 4 停站分配
box(ax, 7.5, 6.6, 10.5, 1.4,
    'STEP 4  灵活停站分配 (distribute_dwell_delta):\n'
    '  dwell_total = T_TARGET - 总运行时间(best_t)\n'
    '  delta = dwell_total - sum(nominal)\n'
    '  各站按 slack 比例分配 delta:\n'
    '    dwell[i] = nominal[i] + delta * (slack[i] / total_slack)\n'
    '    锁定站 (min==nominal) -> slack=0, 不参与分配',
    C['decision'], 8)

arrow(ax, 12.75, 6.6, 12.75, 6.0)

# Step 5 回溯
box(ax, 7.5, 4.3, 10.5, 1.5,
    'STEP 5  回溯还原:\n'
    '  从 best_t 倒查 path[N-1][best_t]\n'
    '  -> (prev_t, class, duration, energy)\n'
    '  逐区间还原每个区间的:\n'
    '    选定等级 | 规划用时 | 停站时间 | 能耗 | 节能量\n'
    '  节能量(Wh) = 历史能耗 - 规划能耗',
    C['process'], 8)

arrow(ax, 12.75, 4.3, 12.75, 3.5)

# ================================================================
# 右侧列 (18.5~25.5): 输出
# ================================================================
ax.text(18.5, 16.4, '输出结果', fontsize=13, fontweight='bold', color='#1a1a1a')

box(ax, 18.5, 14.5, 7, 1.3,
    'Final_Planning_Comparison.csv\n'
    '  区间 | 等级 | 用时 | 停站 | 能耗\n'
    '  节能率: XX% (总计行)\n'
    '  总用时 = 目标时间 (精确匹配)',
    C['output'], 8, bold=True)

box(ax, 18.5, 12.5, 7, 1.3,
    'Optimized_Full_Line_Report.png\n'
    '  上: v-t 全线速度-时间对比\n'
    '  下: v-s 全线速度-距离对比\n'
    '  (灰色=历史 | 红/蓝=优化方案)',
    C['output'], 8)

box(ax, 18.5, 10.5, 7, 1.3,
    '控制台输出:\n'
    '  每站 dwell 详情 + min对比\n'
    '  总能耗(kWh) + 节能率(%)\n'
    '  总用时 vs 目标时间',
    C['output'], 7.5)

# 连接 (DP回溯 -> 输出)
arrow(ax, 18.0, 5.0, 18.5, 15.15)
arrow(ax, 18.0, 5.0, 18.5, 13.15)
arrow(ax, 18.0, 4.5, 18.5, 11.15)

# ================================================================
# 底部: 图例 + 核心公式
# ================================================================
lx, ly = 7.5, 1.5
for i, (color, text) in enumerate([
    (C['input'], '上游/输入'), (C['process'], '计算/处理'),
    (C['decision'], 'DP循环/决策'), (C['output'], '输出结果'),
    (C['data'], '数据文件'), (C['loop'], '循环回路'),
]):
    px = lx + i * 3.3
    ax.add_patch(plt.Rectangle((px, ly), 0.5, 0.35, facecolor=color, edgecolor='gray', linewidth=0.5))
    ax.text(px + 0.7, ly + 0.17, text, fontsize=7.5, va='center')

box(ax, 7.5, 0.3, 18, 0.8,
    '核心递推: dp[t_new] = min(dp[t_new],  dp[t_prev] + energy[等级k])     '
    '停站分配: dwell[i] = nominal[i] + delta * (slack[i] / sum(slack))     '
    '目标: 总时间 = T_TOTAL_TARGET  且  总能耗最小',
    C['process'], 8.5, bold=True)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "output" / "figure" / "dp_logic_flow.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
plt.close()
