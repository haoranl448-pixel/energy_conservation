# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ===================== 1. 全局参数配置 =====================
T_TOTAL_TARGET = 605    # 目标总时长
SLACK = 15                 # 容差
DWELL_TIME = 30.0          # 停站时间
SLIP_RATIO = 1.0  # 滑移率校正
TRIP_INDEX = 0             # 对标第 1 趟历史数据

# 强制约束
MANUAL_CONSTRAINTS = {
    "泗港-曹隘": "class4"
}

# 全局等级允许范围
GLOBAL_ALLOWED_CLASSES = [ "class2", "class3" , "class4" , "class5"]

# ===================== 2. 路径配置 =====================
PROJECT_ROOT = Path(r"D:\energy_conservation")
# 原始结果数据目录（用于提取历史曲线）
DATA_DIR = PROJECT_ROOT / "data" / "data_processed"
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu1_new_v3.csv"
HIST_FILE = PROJECT_ROOT / "full_line1_validation_results.csv"
TRAJ_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v3"
OUT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_plan_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    # "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    # "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    # "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", 
    # "曹隘-柳隘",
    # "柳隘-海晏北路",
    "海晏北路-民安东路", 
    "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", 
    #"三官堂-兴庄路",
    #  "兴庄路-兴海南路",
    # "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ===================== 3. 核心算法函数 =====================

def run_optimization():
    # A. 加载数据
    df_menu = pd.read_csv(MENU_FILE)
    df_hist_summary = pd.read_csv(HIST_FILE).set_index('站间区间')
    
    # B. 计算目标
    total_dwell = (len(STATIONS) - 1) * DWELL_TIME
    t_run_target = T_TOTAL_TARGET - total_dwell
    
    def to_int(t): return int(round(t * 10))
    target_int = to_int(t_run_target)
    slack_int = to_int(SLACK)
    
    dp = {0: 0.0}
    path = []

    print(f"🚀 正在搜索最优组合 (目标运行时间: {t_run_target:.1f}s)...")

    for i, sp in enumerate(STATIONS):
        new_dp, new_path = {}, {}
        all_opts = df_menu[df_menu['站间区间'] == sp]
        options = all_opts[all_opts['运行等级'] == MANUAL_CONSTRAINTS[sp]] if sp in MANUAL_CONSTRAINTS else all_opts[all_opts['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]

        for t_prev, e_prev in dp.items():
            for _, row in options.iterrows():
                t_sum = t_prev + to_int(row['运行时长(s)'])
                if t_sum > target_int + slack_int + 200: continue
                e_sum = e_prev + row['预测能耗(Wh)']
                if t_sum not in new_dp or e_sum < new_dp[t_sum]:
                    new_dp[t_sum] = e_sum
                    new_path[t_sum] = (t_prev, row['运行等级'], row['运行时长(s)'], row['预测能耗(Wh)'])
        dp, path = new_dp, path + [new_path]

    valid_keys = [t for t in dp.keys() if target_int - slack_int <= t <= target_int + slack_int]
    if not valid_keys:
        print("❌ 无法找到可行解"); return
    
    best_t_int = min(valid_keys, key=lambda x: (dp[x], abs(x - target_int)))
    final_energy = dp[best_t_int]

    # E. 回溯结果并准备对比数据
    final_rows = []
    curr_t = best_t_int
    for i in range(len(STATIONS)-1, -1, -1):
        prev_t, c_name, t_val, e_val = path[i][curr_t]
        sp = STATIONS[i]
        final_rows.append({
            "站间区间": sp, "选定等级": c_name, "规划用时(s)": round(t_val, 0),
            "规划能耗(Wh)": round(e_val, 2), "prev_t": prev_t
        })
        curr_t = prev_t
    final_rows.reverse()

    # ===================== 4. 绘图重建（规划 vs 历史） =====================
    print("📈 正在重建对比曲线图...")
    plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    
    # 累加器
    t_opt_acc, s_opt_acc = 0.0, 0.0
    t_hist_acc, s_hist_acc = 0.0, 0.0
    
    # 全局绘图列表
    plot_data = {
        'opt_t': [], 'opt_v': [], 'opt_s': [],
        'hist_t': [], 'hist_v': [], 'hist_s': []
    }
    dwell_zones = []

    for i, row in enumerate(final_rows):
        sp, c_name = row['站间区间'], row['选定等级']
        
        # --- 1. 提取规划轨迹 ---
        df_opt = pd.read_csv(TRAJ_BASE_DIR / sp / f"{c_name}_generated_curve.csv")
        plot_data['opt_t'].extend((df_opt['time_s'] + t_opt_acc).tolist())
        plot_data['opt_v'].extend((df_opt['velocity_mps'] * 3.6).tolist())
        plot_data['opt_s'].extend((df_opt['dist_m'] + s_opt_acc).tolist())
        
        # --- 2. 提取历史轨迹 (Trip 6) ---
        df_h_all = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        target_seg = sorted(df_h_all['segment'].unique())[TRIP_INDEX]
        df_h = df_h_all[df_h_all['segment'] == target_seg].copy()
        
        h_v = df_h['速度(m/s)'].values * 3.6
        h_t = df_h['时刻'].values - df_h['时刻'].iloc[0]
        h_s = df_h['累计位移(m)'].values / SLIP_RATIO
        
        plot_data['hist_t'].extend((h_t + t_hist_acc).tolist())
        plot_data['hist_v'].extend(h_v.tolist())
        plot_data['hist_s'].extend((h_s + s_hist_acc).tolist())

        # --- 3. 计算本段末尾偏移 ---
        t_opt_acc += row['规划用时(s)']
        s_opt_acc += df_opt['dist_m'].iloc[-1]
        
        t_hist_acc += h_t[-1]
        s_hist_acc += h_s[-1]

        # --- 4. 插入停站段 ---
        if i < len(final_rows) - 1:
            dwell_zones.append((t_opt_acc, t_opt_acc + DWELL_TIME))
            # 规划停站
            plot_data['opt_t'].extend([t_opt_acc, t_opt_acc + DWELL_TIME])
            plot_data['opt_v'].extend([0, 0])
            plot_data['opt_s'].extend([s_opt_acc, s_opt_acc])
            t_opt_acc += DWELL_TIME
            
            # 历史停站 (假设历史也按标准停站对齐时间轴进行对比)
            plot_data['hist_t'].extend([t_hist_acc, t_hist_acc + DWELL_TIME])
            plot_data['hist_v'].extend([0, 0])
            plot_data['hist_s'].extend([s_hist_acc, s_hist_acc])
            t_hist_acc += DWELL_TIME

    # ===================== 5. 绘图与产出 =====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    
    # VT 图
    ax1.plot(plot_data['hist_t'], plot_data['hist_v'], color='gray', alpha=0.4, label='历史 Trip 6')
    ax1.plot(plot_data['opt_t'], plot_data['opt_v'], color='red', linewidth=1.5, label='节能规划方案')
    for start, end in dwell_zones: ax1.axvspan(start, end, color='gray', alpha=0.05)
    ax1.set_title("全线速度-时间 (v-t) 对标图"); ax1.set_ylabel("速度 (km/h)"); ax1.legend()

    # VS 图
    ax2.plot(plot_data['hist_s'], plot_data['hist_v'], color='gray', alpha=0.4, label='历史 Trip 6')
    ax2.plot(plot_data['opt_s'], plot_data['opt_v'], color='blue', linewidth=1.5, label='节能规划方案')
    ax2.set_title("全线速度-位移 (v-s) 对标图"); ax2.set_xlabel("里程 (m)"); ax2.set_ylabel("速度 (km/h)"); ax2.legend()
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Full_Line_Comparison_Report.png", dpi=300)
    print(f"✅ 对比图已保存: {OUT_DIR / 'Full_Line_Comparison_Report.png'}")

if __name__ == "__main__":
    run_optimization()