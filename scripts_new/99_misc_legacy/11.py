# -*- coding: utf-8 -*-
"""
scripts/plot_trip6_vst_aligned_v2.py
功能：全线 v-t 和 v-s 对比图。
修复：不再依赖 CSV 里的 station_pair 列，通过速度曲线自动切分并对齐位移。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 环境与路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCHEDULE_DIR = os.path.join(project_root, "output", "schedule", "trip6_class_results")
DATA_DIR = os.path.join(project_root, "data", "data_processed")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "trip6_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SERVICE = "50714"
DWELL_TIME = 30.0 # 停站时间

STATIONS = [
        "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区", "南高教园区-下应路",
    "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘", "柳隘-海晏北路",
    "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路", "院士路-盎孟港", "盎孟港-三官堂",
    "三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰", "梅堰-永茂路","永茂路-镇海大道", "镇海大道-骆驼桥",
]

# ================= 2. 核心逻辑：自动切分与对齐 =================

def load_and_align_v2():
    # 1. 加载规划 CSV
    files = glob.glob(os.path.join(SCHEDULE_DIR, "full_vt_schedule_total3660*.csv"))
    if not files: raise FileNotFoundError("❌ 找不到规划 CSV")
    df_opt_full = pd.read_csv(files[0])
    
    # 2. 识别规划数据中的各站区间 (根据速度 > 0.1 判定)
    # 逻辑：找出每一段连续速度非0的区间
    df_opt_full['moving'] = df_opt_full['v_ms'] > 0.1
    # 标记状态切换点，生成组ID
    df_opt_full['group'] = (df_opt_full['moving'] != df_opt_full['moving'].shift()).cumsum()
    # 只保留行驶中的组
    opt_segments = [g for _, g in df_opt_full[df_opt_full['moving']].groupby('group')]

    aligned_opt_rows = []
    aligned_hist_rows = []
    
    t_h_offset = 0.0
    s_h_offset = 0.0
    
    opt_seg_idx = 0 # 规划段计数器
    
    print(f"🚀 开始逐站对齐位移 (共需对齐 {len(STATIONS)} 个区间)...")

    for sp in STATIONS:
        # --- A. 获取历史数据该段 ---
        f_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        if not os.path.exists(f_path): continue
        
        try:
            df_h_all = pd.read_excel(f_path)
            df_h_all['ser_str'] = df_h_all['服务号'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
            df_h_seg = df_h_all[df_h_all['ser_str'] == TARGET_SERVICE].sort_values('时刻')
            if df_h_seg.empty: continue
        except: continue

        # --- B. 获取对应的规划段 ---
        if opt_seg_idx >= len(opt_segments):
            print(f"⚠️ 规划数据已用尽，无法对齐站: {sp}")
            break
        
        df_o_seg = opt_segments[opt_seg_idx]
        opt_seg_idx += 1

        # --- C. 计算该站的位移修正比例 (Ratio) ---
        l_hist = df_h_seg['累计位移(m)'].max() - df_h_seg['累计位移(m)'].min()
        l_opt = df_o_seg['s_global'].max() - df_o_seg['s_global'].min()
        
        # 比例 = 历史距离 / 规划距离
        ratio = l_hist / l_opt if l_opt > 0 else 1.0
        
        # --- D. 重构历史轨迹 (时间、位移均连续化) ---
        t_h_local = df_h_seg['时刻'].values - df_h_seg['时刻'].min()
        s_h_local = df_h_seg['累计位移(m)'].values - df_h_seg['累计位移(m)'].min()
        
        aligned_hist_rows.append(pd.DataFrame({
            'time': t_h_local + t_h_offset,
            'dist': s_h_local + s_h_offset,
            'v_kmh': df_h_seg['速度(m/s)'].values * 3.6
        }))

        # --- E. 重构规划轨迹 (应用位移比例修正) ---
        # 修正位移 = 相对位移 * 比例
        s_o_relative = df_o_seg['s_global'].values - df_o_seg['s_global'].min()
        s_o_aligned = s_o_relative * ratio
        
        # 修正时间起点（对齐历史）
        t_o_local = df_o_seg['t_global'].values - df_o_seg['t_global'].min()
        
        aligned_opt_rows.append(pd.DataFrame({
            'time': t_o_local + t_h_offset, # 时间轴对齐历史起点
            'dist': s_o_aligned + s_h_offset, # 位移轴对齐历史起点
            'v_kmh': df_o_seg['v_ms'].values * 3.6
        }))

        # --- F. 更新全局偏移量 (以历史数据为准进行步进) ---
        t_h_offset += (t_h_local.max() + DWELL_TIME)
        s_h_offset += l_hist

    return pd.concat(aligned_hist_rows), pd.concat(aligned_opt_rows)

# ================= 3. 绘图执行 =================

def main():
    try:
        df_h, df_o = load_and_align_v2()
    except Exception as e:
        print(f"❌ 失败: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # 子图1: v-t 对比
    ax1.plot(df_h['time'], df_h['v_kmh'], color='#e74c3c', lw=1.0, ls='--', label='历史 (Trip 6)', alpha=0.7)
    ax1.plot(df_o['time'], df_o['v_kmh'], color='#3498db', lw=1.5, label='规划 ', alpha=0.9)
    ax1.set_title("全线速度对比 (时间轴)", fontsize=14)
    ax1.set_ylabel("速度 (km/h)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 子图2: v-s 对比 (🌟 已对齐)
    ax2.plot(df_h['dist'], df_h['v_kmh'], color='#e74c3c', lw=1.0, ls='--', label='历史 (Trip 6)', alpha=0.7)
    ax2.plot(df_o['dist'], df_o['v_kmh'], color='#2ecc71', lw=1.5, label='规划 ', alpha=0.9)
    ax2.set_title("全线速度对比(里程轴)", fontsize=14)
    ax2.set_xlabel("累计里程 (m)")
    ax2.set_ylabel("速度 (km/h)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Trip6_VST_Aligned_Final.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 成功！图表已保存至: {save_path}")
    print(f"💡 对齐原理：计算每一站历史位移与规划位移的比例，对规划 $s$ 轴进行拉伸。")

if __name__ == "__main__":
    main()