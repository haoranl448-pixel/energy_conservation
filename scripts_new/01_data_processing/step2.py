# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
# 引入之前的轨迹重建函数
from run_global_schedule_residual import make_profile_trapezoid, load_params, refine_meiyan_for_time, refine_sigang_for_time

DATA_DIR = Path(r"D:\energy_conservation\data\data_processed")
# SCHEDULE_CSV = Path(r"output/schedule/trip6_class_results/schedule_table.csv") # 你的DP输出
SCHEDULE_CSV = Path(r"output/schedule/trip6_class_results_residualV2/schedule_table.csv") # 你的DP输出
OUT_DIR = Path(r"output/trip6_analysis_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLIP_RATIO = 1.015953550423103
TRIP_INDEX = 5

STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂","三官堂-兴庄路", "兴庄路-兴海南路", "兴海南路-梅堰", 
    "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def run_comparison():
    plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 加载优化时刻表
    df_sched = pd.read_csv(SCHEDULE_CSV)
    params = load_params()
    
    t_h_all, v_h_all, s_h_all = [0], [0], [0]
    t_o_all, v_o_all, s_o_all = [0], [0], [0]
    th_off, to_off, s_off = 0, 0, 0
    report = []

    for i, sp in enumerate(STATIONS):
        # A. 读取历史 Trip 6
        df_h = pd.read_excel(DATA_DIR / f"results_{sp}.xlsx")
        data_h = df_h[df_h['segment'] == sorted(df_h['segment'].unique())[TRIP_INDEX]]
        v_hist = data_h['速度(m/s)'].values * 3.6
        t_hist = data_h['时刻'].values - data_h['时刻'].iloc[0]
        #s_hist = data_h['累计位移(m)'].values / SLIP_RATIO
        s_hist = data_h['累计位移(m)'].values 
        e_hist = data_h['energy'].sum() / 3600.0

        # B. 获取规划数据
        row_o = df_sched[df_sched['station_pair'] == sp].iloc[0]
        t_opt = row_o['target_time(s)']
        e_opt = row_o['energy(Wh)']
        p = params[sp]

       

        # C. 拼接
       

        # D. 更新偏移 (假设固定30s停站用于绘图连续性)
        th_off += (t_hist[-1] + 30); to_off += (t_opt + 30); s_off += s_hist[-1]
        
        report.append({
            '区间': sp, '历史时间': round(t_hist[-1],1), '规划时间': round(t_opt,1),
            '历史能耗(Wh)': round(e_hist,2), '规划能耗(Wh)': round(e_opt,2),
            '节能(Wh)': round(e_hist - e_opt, 2),
            '节能率': f"{((e_hist - e_opt)/e_hist*100):.2f}%" if e_hist>0 else "0%"
        })

    # 绘图
    
    pd.DataFrame(report).to_csv(OUT_DIR / "Trip6_Saving_Report_mlp.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 对标分析完成！图片与报表已保存至: {OUT_DIR}")

if __name__ == "__main__":
    run_comparison()