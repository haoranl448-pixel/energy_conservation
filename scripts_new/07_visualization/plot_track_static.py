# -*- coding: utf-8 -*-
"""
scripts/plot_track_merged.py (V5: 特事特办分流版)
逻辑：
- 对于 "梅堰-永茂路"：启用【模糊查找 + 强力清洗】，解决文件名怪异和内部数据脏乱的问题。
- 对于 其他站点：保留【标准查找 + 普通读取】，保证原本能跑通的站点不受影响。
"""
import os
import sys
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")

# 1. 路径与字体
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 2. 站点列表
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def get_section_data(sp):
    """
    根据站点名，智能分流读取数据
    """
    search_dirs = [
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "data", "data_processed")
    ]

    # ================= [特殊通道] 梅堰-永茂路专用 =================
    if sp == "梅堰-永茂路":
        st, ed = "梅堰", "永茂路"
        target_file = None
        
        # 1. 模糊查找文件
        for d in search_dirs:
            if not os.path.exists(d): continue
            for f in glob.glob(os.path.join(d, "*.xlsx")):
                if st in os.path.basename(f) and ed in os.path.basename(f):
                    target_file = f
                    break
            if target_file: break
            
        if not target_file: return None, None, None

        # 2. 强力数据清洗
        try:
            df = pd.read_excel(target_file)
            req_cols = ['累计位移(m)', 'gradient', 'curvature']
            if not all(c in df.columns for c in req_cols): return None, None, None
            
            # 强制转数值，清理乱码
            for c in req_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=req_cols).sort_values('累计位移(m)')
            
            if df.empty: return None, None, None
            
            s = df['累计位移(m)'].values
            g = df['gradient'].values
            c = df['curvature'].values
            
            c_plot = np.abs(c)
            c_plot[(c_plot > 3000) | (c_plot <= 0)] = 3500
            return s, g, c_plot
        except: return None, None, None

    # ================= [普通通道] 其他站点专用 =================
    else:
        target_file = None
        
        # 1. 精确查找文件
        for d in search_dirs:
            files = sorted(glob.glob(os.path.join(d, f"results_{sp}*.xlsx")))
            if files:
                target_file = files[0]
                break
                
        if not target_file: return None, None, None

        # 2. 标准读取 (不进行过度清洗，防止误删好数据)
        try:
            df = pd.read_excel(target_file)
            
            if 'segment' not in df.columns: return None, None, None
            first_seg = df['segment'].unique()[0]
            trip = df[df['segment'] == first_seg].sort_values('累计位移(m)')
            
            s = trip['累计位移(m)'].values
            g = trip['gradient'].values
            c = trip['curvature'].values
            
            # 曲率处理 (画图美化)
            c_plot = np.abs(c)
            c_plot[(c_plot > 3000) | (c_plot <= 0)] = 3500
            
            return s, g, c_plot
        except: return None, None, None

# ================= 模式一：拼接长卷 =================
def plot_stitched_panorama(out_dir):
    print(">>> [1/2] 正在绘制全线拼接长图...")
    full_s, full_g, full_c = [], [], []
    station_ticks, station_labels = [], []
    current_offset = 0.0
    
    station_labels.append(LINE5_SECTIONS[0].split('-')[0])
    station_ticks.append(0)

    for sp in LINE5_SECTIONS:
        s, g, c = get_section_data(sp)
        if s is None: 
            print(f"   ⚠️ 跳过缺失: {sp}")
            s = np.linspace(0, 1000, 100); g = np.zeros_like(s); c = np.full_like(s, 3500)
        
        full_s.append(s + current_offset)
        full_g.append(g)
        full_c.append(c)
        current_offset += s[-1]
        station_ticks.append(current_offset)
        station_labels.append(sp.split('-')[-1])

    S = np.concatenate(full_s); G = np.concatenate(full_g); C = np.concatenate(full_c)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    
    ax1.plot(S, G, color='#d62728', lw=1)
    ax1.fill_between(S, G, 0, color='#d62728', alpha=0.2)
    ax1.set_ylabel("坡度 (‰)", fontsize=14, fontweight='bold', color='#d62728')
    ax1.set_title("全线纵断面 (Gradient)", fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5); ax1.axhline(0, color='black', lw=1)
    
    ax2.plot(S, C, color='#1f77b4', lw=1)
    ax2.set_ylabel("曲线半径 (m)", fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_title("全线平面 (Curvature)", fontsize=18)
    ax2.set_ylim(0, 4000)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    for ax in [ax1, ax2]:
        for x in station_ticks: ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            
    ax2.set_xticks(station_ticks)
    ax2.set_xticklabels(station_labels, rotation=90, fontsize=10)
    ax2.set_xlabel("里程 / 站点", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "track_panorama_stitched.png")
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ 长卷已保存: {save_path}")

# ================= 模式二：26合1网格图 =================
def plot_grid_view(out_dir):
    print(">>> [2/2] 正在绘制 26合1 网格图...")
    N = len(LINE5_SECTIONS)
    cols = 4; rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()
    
    for i, sp in enumerate(LINE5_SECTIONS):
        ax = axes[i]
        s, g, c = get_section_data(sp)
        
        if s is not None:
            ax_g = ax; ax_c = ax.twinx()
            ax_g.fill_between(s, g, 0, color='#d62728', alpha=0.3)
            ax_g.plot(s, g, color='#d62728', lw=1)
            ax_g.set_ylabel("坡度", color='#d62728', fontsize=8)
            ax_g.tick_params(axis='y', labelcolor='#d62728', labelsize=8)
            ax_g.set_ylim(-35, 35) 
            
            ax_c.plot(s, c, color='#1f77b4', lw=1.5)
            ax_c.set_ylabel("半径", color='#1f77b4', fontsize=8)
            ax_c.tick_params(axis='y', labelcolor='#1f77b4', labelsize=8)
            ax_c.set_ylim(0, 4000)
            
            ax.set_title(f"{i+1}. {sp}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i >= (rows-1)*cols: ax.set_xlabel("里程 (m)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "读取失败", ha='center')
            ax.set_title(sp)

    for j in range(N, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    save_path = os.path.join(out_dir, "track_grid_26in1.png")
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ 网格图已保存: {save_path}")

if __name__ == "__main__":
    out_dir = os.path.join(project_root, "output", "analysis", "track_visuals")
    os.makedirs(out_dir, exist_ok=True)
    plot_stitched_panorama(out_dir)
    plot_grid_view(out_dir)