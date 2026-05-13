# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ===================== 1. 全局参数配置 =====================
#T_TOTAL_TARGET = 3656  # 全程总时长目标（含停站）
# T_TOTAL_TARGET =3654.8
#T_TOTAL_TARGET =647## "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", 
    # "曹隘-柳隘",
T_TOTAL_TARGET =605  #"布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
SLACK =15
     # 全局总时间容差。例如：3660.8 ± 15s 范围内的组合都算合格
DWELL_TIME = 30.0         # 默认停站时间

# 🌟 强制约束：指定某些站必须跑的等级
MANUAL_CONSTRAINTS = {
    # "布政-张家潭": "class3", 
    # "张家潭-同德路": "class3",
    # # "石碶-雅渡": "class3", 
    # "钟公庙-鄞州区政府": "class3",
    # # "大洋江-泗港": "class3",
    # # "三官堂-兴庄路": "class3", 
    # # "兴庄路-兴海南路": "class3",
    # # "镇海大道-骆驼桥": "class3",
    # "梅堰-永茂路":"class4",
    # "庙堰-钟公庙": "class3",
    # "钱湖南路-南高教园区": "class3",
    # "院士路-盎孟港": "class3"
    "泗港-曹隘": "class4"
}

# 🌟 全局等级允许范围
GLOBAL_ALLOWED_CLASSES = [  "class2", "class3" , "class4" , "class5"  ]

# ===================== 2. 路径配置 =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MENU_FILE = PROJECT_ROOT / "output" / "analysis" / "ato_class_energy_menu1_new_v3.csv"
#HIST_FILE = PROJECT_ROOT / "output" / "trip5_final_report" / "Trip5_Detailed_Energy_Comparison.csv"
HIST_FILE = PROJECT_ROOT / "full_line1_validation_results.csv"
TRAJ_BASE_DIR = PROJECT_ROOT / "output" / "ato_generated_results_new_v3"
OUT_DIR = PROJECT_ROOT / "output" / "schedule" / "final_plan_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SECTION_PARAMS_FILE = PROJECT_ROOT / "data" / "static" / "section_params_trip1.csv"
DATA_DIR = PROJECT_ROOT / "data" / "data_processed"
SLIP_RATIO = 1.0
TRIP_INDEX = 0

STATIONS = [
    # "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    # "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区", "南高教园区-下应路",
    # "下应路-大洋江", "大洋江-泗港", "泗港-曹隘",  "曹隘-柳隘", "柳隘-海晏北路",
    "海晏北路-民安东路", 
    "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", 
    #"三官堂-兴庄路","兴庄路-兴海南路","兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", 
    # "镇海大道-骆驼桥"
]

# ===================== 3. 核心计算函数 =====================

def run_optimization():
    # A. 加载数据
    df_menu = pd.read_csv(MENU_FILE)
    df_hist = pd.read_csv(HIST_FILE).set_index('站间区间')
    df_mass = pd.read_csv(SECTION_PARAMS_FILE)
    mass_map = dict(zip(df_mass['station_pair'], df_mass['MASS']))
    
    # B. 计算运行时间目标
    total_dwell = (len(STATIONS) - 1) * DWELL_TIME
    t_run_target = T_TOTAL_TARGET - total_dwell
    
    # C. DP 初始化 (0.1s 精度)
    def to_int(t): return int(round(t * 10))
    target_int = to_int(t_run_target)
    slack_int = to_int(SLACK)
    
    # dp[时间状态] = 最小能耗
    dp = {0: 0.0}
    path = []

    print(f"🔍 目标纯运行时间: {t_run_target:.2f}s")
    print(f"🚀 正在搜索所有 Class 组合，寻找总时长在 [{t_run_target-SLACK:.1f}s, {t_run_target+SLACK:.1f}s] 之间的最节能方案...")

    for i, sp in enumerate(STATIONS):
        new_dp, new_path = {}, {}
        all_opts = df_menu[df_menu['站间区间'] == sp]
        
        # 强制约束与全局过滤
        if sp in MANUAL_CONSTRAINTS:
            options = all_opts[all_opts['运行等级'] == MANUAL_CONSTRAINTS[sp]]
        else:
            options = all_opts[all_opts['运行等级'].isin(GLOBAL_ALLOWED_CLASSES)]

        for t_prev, e_prev in dp.items():
            for _, row in options.iterrows():
                # 注意：这里的 t_curr 是固定的 Class 时间，我们直接累加
                t_curr = round(row['运行时长(s)'], 1)
                t_sum = t_prev + to_int(t_curr)
                
                # 剪枝：如果已经明显超过了 (目标 + 容差)，不再往后算
                if t_sum > target_int + slack_int + 200: continue
                
                e_sum = e_prev + row['预测能耗(Wh)']
                if t_sum not in new_dp or e_sum < new_dp[t_sum]:
                    new_dp[t_sum] = e_sum
                    new_path[t_sum] = (t_prev, row['运行等级'], t_curr, row['预测能耗(Wh)'])
        
        dp, path = new_dp, path + [new_path]
        print(f"   [{i+1}/{len(STATIONS)}] {sp} 处理完毕，当前可行组合路径数: {len(dp)}")

    # D. 核心逻辑修改：在全局容差窗口内筛选最省电的
    valid_keys = [t for t in dp.keys() if target_int - slack_int <= t <= target_int + slack_int]
    
    if not valid_keys:
        print("❌ 错误：在当前 SLACK 范围内无法用固定的 Class 时间拼凑出目标时长。")
        # 自动建议
        min_reachable = min(dp.keys()) / 10.0
        max_reachable = max(dp.keys()) / 10.0
        print(f"💡 建议：当前 Class 组合能达到的时间范围是 [{min_reachable}s, {max_reachable}s]，请调整目标或增大 SLACK。")
        return
    
    # 🌟 在所有落入窗口的组合中，选能耗最小的那个
    # best_t_int = min(valid_keys, key=lambda x: dp[x])
    # final_energy = dp[best_t_int]
    # best_t_int = min(valid_keys, key=lambda x: (abs(x - target_int), dp[x]))#最准点的
    
    # final_energy = dp[best_t_int]



    # 在容差窗口内选能耗最低的组合，能耗相同时选更准点的
    best_t_int = min(valid_keys, key=lambda x: (dp[x], abs(x - target_int)))
    final_energy = dp[best_t_int]

    # E. 回溯结果
    final_rows = []
    curr_t = best_t_int
    for i in range(len(STATIONS)-1, -1, -1):
        prev_t, c_name, t_val, e_val = path[i][curr_t]
        sp = STATIONS[i]
        h_time = df_hist.loc[sp, '历史运行时间(s)']
        h_energy = df_hist.loc[sp, '历史能耗(Wh)']
        final_rows.append({
            "站间区间": sp,
            "选定等级": c_name,
            "规划用时(s)": round(t_val, 0),
            "历史用时(s)": round(h_time,1),
            "规划能耗(Wh)": round(e_val, 2),
            "历史能耗(Wh)": round(h_energy, 2),
            "节能量(Wh)": round(h_energy - e_val, 2),
            "MASS": round(mass_map.get(sp, np.nan), 2),
        })
        curr_t = prev_t
    
    final_rows.reverse()
    df_res = pd.DataFrame(final_rows)
    
    # F. 计算总计
    total_h_e = df_res['历史能耗(Wh)'].sum()
    total_p_e = df_res['规划能耗(Wh)'].sum()
    total_p_t = df_res['规划用时(s)'].sum() + total_dwell
    total_h_t = df_res['历史用时(s)'].sum() + total_dwell
    saving_rate = (total_h_e - total_p_e) / total_h_e * 100

    summary_row = {
        "站间区间": "--- 总计 ---", "选定等级": f"节能率: {saving_rate:.2f}%",
        "规划用时(s)": total_p_t, "历史用时(s)": total_h_t,
        "规划能耗(Wh)": round(total_p_e,2), "历史能耗(Wh)": round(total_h_e,2), "节能量(Wh)": round(total_h_e - total_p_e,2),
        "MASS": round(df_res['MASS'].sum(), 2),
    }
    df_res = pd.concat([df_res, pd.DataFrame([summary_row])], ignore_index=True)
    df_res.to_csv(OUT_DIR / "Final_Planning_Comparison.csv", index=False, encoding='utf-8-sig')

    # ===================== 4. 绘图重建（规划 vs 历史轨迹对比） =====================
    print("📈 正在重建全线对比运行图（规划 vs 历史）...")
    plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False

    t_opt_acc, s_opt_acc = 0.0, 0.0
    t_hist_acc, s_hist_acc = 0.0, 0.0
    plot_data = {'opt_t': [], 'opt_v': [], 'opt_s': [],
                 'hist_t': [], 'hist_v': [], 'hist_s': []}
    dwell_zones = []

    for i, row in enumerate(final_rows):
        sp, c_name = row['站间区间'], row['选定等级']

        # --- 规划轨迹 ---
        df_opt = pd.read_csv(TRAJ_BASE_DIR / sp / f"{c_name}_generated_curve.csv")
        plot_data['opt_t'].extend((df_opt['time_s'] + t_opt_acc).tolist())
        plot_data['opt_v'].extend((df_opt['velocity_mps'] * 3.6).tolist())
        plot_data['opt_s'].extend((df_opt['dist_m'] + s_opt_acc).tolist())

        # --- 历史轨迹 ---
        df_h_all = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        target_seg = sorted(df_h_all['segment'].unique())[TRIP_INDEX]
        df_h = df_h_all[df_h_all['segment'] == target_seg].copy()
        h_v = df_h['速度(m/s)'].values * 3.6
        h_t = df_h['时刻'].values - df_h['时刻'].iloc[0]
        h_s = df_h['累计位移(m)'].values / SLIP_RATIO

        plot_data['hist_t'].extend((h_t + t_hist_acc).tolist())
        plot_data['hist_v'].extend(h_v.tolist())
        plot_data['hist_s'].extend((h_s + s_hist_acc).tolist())

        # --- 累加偏移 ---
        t_opt_acc += row['规划用时(s)']
        s_opt_acc += df_opt['dist_m'].iloc[-1]
        t_hist_acc += h_t[-1]
        s_hist_acc += h_s[-1]

        # --- 停站段 ---
        if i < len(final_rows) - 1:
            dwell_zones.append((t_opt_acc, t_opt_acc + DWELL_TIME))
            plot_data['opt_t'].extend([t_opt_acc, t_opt_acc + DWELL_TIME])
            plot_data['opt_v'].extend([0, 0])
            plot_data['opt_s'].extend([s_opt_acc, s_opt_acc])
            t_opt_acc += DWELL_TIME

            plot_data['hist_t'].extend([t_hist_acc, t_hist_acc + DWELL_TIME])
            plot_data['hist_v'].extend([0, 0])
            plot_data['hist_s'].extend([s_hist_acc, s_hist_acc])
            t_hist_acc += DWELL_TIME

    # --- 出图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    ax1.plot(plot_data['hist_t'], plot_data['hist_v'], color='gray', alpha=0.4, linewidth=1.0, label='历史实际轨迹')
    ax1.plot(plot_data['opt_t'], plot_data['opt_v'], color='red', linewidth=1.2, label='节能规划方案')
    for start, end in dwell_zones: ax1.axvspan(start, end, color='gray', alpha=0.05)
    ax1.set_title(f"全线 v-t 对比图 | 规划总时间: {total_p_t:.1f}s | 节能率: {saving_rate:.2f}%")
    ax1.set_ylabel("速度 (km/h)"); ax1.legend()

    ax2.plot(plot_data['hist_s'], plot_data['hist_v'], color='gray', alpha=0.4, linewidth=1.0, label='历史实际轨迹')
    ax2.plot(plot_data['opt_s'], plot_data['opt_v'], color='blue', linewidth=1.2, label='节能规划方案')
    ax2.set_title(f"全线 v-s 对比图 | 总里程: {s_opt_acc:.1f}m")
    ax2.set_xlabel("里程 (m)"); ax2.set_ylabel("速度 (km/h)"); ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "Optimized_Full_Line_Report.png", dpi=300)
    print(f"\n✅ 规划完成！总能耗: {total_p_e/1000:.3f}kWh, 实现时间: {total_p_t:.2f}s")

if __name__ == "__main__":
    run_optimization()