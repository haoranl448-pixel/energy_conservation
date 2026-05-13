# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings
from pathlib import Path

# ================= 1. 配置与路径 =================
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = Path(r"D:\energy_conservation\data\data_processed")
OPT_DIR = Path(r"output/optimization/trip1_v2_6feat")
REPORT_DIR = Path(r"output/trip1_final_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SLIP_RATIO = 1.015953550423103
TRIP_INDEX = 0 # 第1趟
STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂","三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰",
    "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ================= 2. 核心算法 (自包含) =================
# 此处假定你已经有 dp_optimize 和轨迹函数在 run_global_schedule_residual.py 中
try:
    from run_global_schedule_residual import dp_optimize, make_profile_trapezoid, load_params, refine_meiyan_for_time, refine_sigang_for_time
except ImportError:
    print("❌ 错误：请确保 run_global_schedule_residual.py 存在")
    sys.exit()

def export_detailed_report(hist_list, opt_alloc, output_path):
    rows = []
    total_h_e, total_p_e, total_h_t, total_p_t = 0, 0, 0, 0
    
    for i, sp in enumerate(STATIONS):
        h = hist_list[i]
        t_opt, v_opt, e_opt = opt_alloc[i] # v_opt 是关键
        
        energy_saved = h['energy_wh'] - e_opt
        rate = (energy_saved / h['energy_wh']) * 100 if h['energy_wh'] > 0 else 0
        
        rows.append({
            '站间区间': sp,
            '历史运行时间(s)': round(h['run_dur'], 2),
            '规划运行时间(s)': round(t_opt, 2),
            '规划速度(m/s)': round(v_opt, 4), # 🌟 回放脚本依赖这一列
            '历史能耗(Wh)': round(h['energy_wh'], 2),
            '规划能耗(Wh)': round(e_opt, 2),
            '节能贡献(Wh)': round(energy_saved, 2),
            '区间节能率': f"{rate:.2f}%"
        })
        total_h_e += h['energy_wh']; total_p_e += e_opt
        total_h_t += h['run_dur']; total_p_t += t_opt

    df = pd.DataFrame(rows)
    # 添加总计
    summary = {'站间区间': '--- 全线总计 ---', '历史运行时间(s)': total_h_t, '规划运行时间(s)': total_p_t,
               '历史能耗(Wh)': total_h_e, '规划能耗(Wh)': total_p_e, '节能贡献(Wh)': total_h_e - total_p_e,
               '区间节能率': f"{((total_h_e - total_p_e)/total_h_e*100):.2f}%", '规划速度(m/s)': '-'}
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df

# ================= 3. 执行流程 =================
def main():
    hist_list = []
    total_time_target = 0
    print("🔍 正在提取 Trip 6 数据...")
    for sp in STATIONS:
        df = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        for c in ['时刻', 'energy']: df[c] = pd.to_numeric(df[c], errors='coerce')
        segs = sorted(df['segment'].unique())
        data = df[df['segment'] == segs[TRIP_INDEX]].copy().dropna(subset=['时刻'])
        run_dur = data['时刻'].iloc[-1] - data['时刻'].iloc[0]
        hist_list.append({'sp': sp, 'run_dur': run_dur, 'energy_wh': data['energy'].sum()/3600.0})
        total_time_target += run_dur

    print("🚀 正在运行 DP 全局寻优...")
    curves = []
    for sp in STATIONS:
        df_c = pd.read_csv(OPT_DIR / sp / f"time_energy_curve_{sp}.csv")
        curves.append((df_c['t_arrival'].values, df_c['v_peak_opt'].values, df_c['E_wh'].values))
    
    alloc, E_total, T_used = dp_optimize(curves, total_time_target, slack=2.0)
    
    # 保存精细化 CSV
    report_path = REPORT_DIR / "Trip6_Detailed_Energy_Comparison.csv"
    export_detailed_report(hist_list, alloc, report_path)
    print(f"✅ 报表已生成(含速度列): {report_path}")

if __name__ == "__main__":
    main()