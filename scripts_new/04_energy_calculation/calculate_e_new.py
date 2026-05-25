# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ================= 1. 路径配置 =================
project_root = Path(__file__).parent.parent
# 优化结果存放的根目录 (Step 2 生成的 Pareto 曲线)
PARETO_DIR = project_root / "output" / "optimization" / "trip6_v2_6feat"
# 详细对标报告 (包含历史时间)
# 输出路径

REPORT_FILE = project_root / "output" / "trip6_analysis_report" / "Trip6_Saving_Report.csv"
PARAM_FILE = os.path.join(project_root, "data", "static", "section_params_trip6.csv")
DATA_DIR = os.path.join(project_root, "data", "data_processed")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual_v2")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "trip6_historical_playback")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def lookup_energy_from_table(station_pair, target_time):
    """
    核心查表逻辑：从 CSV 中寻找最接近目标时间的能耗和速度
    """
    csv_path = PARETO_DIR / station_pair / f"time_energy_curve_{station_pair}.csv"
    
    if not csv_path.exists():
        print(f"⚠️ 找不到区间 {station_pair} 的能效表")
        return None, None

    # 读取该站的能效菜单
    df_menu = pd.read_csv(csv_path)
    
    # 寻找与 target_time 最接近的行
    # 使用 abs 差值最小化来索引
    idx = (df_menu['t_arrival'] - target_time).abs().idxmin()
    closest_row = df_menu.loc[idx]
    
    # 检查时间偏差，如果偏差太大（比如超过1秒），说明该时间点在优化时没搜到
    time_gap = abs(closest_row['t_arrival'] - target_time)
    
    return closest_row['E_wh'], closest_row['v_peak_opt'], time_gap

def run_lookup_analysis():
    print("🔍 开始执行查表法能耗回放...")
    
    # 1. 加载对标报告（获取我们要查询的目标时间）
    if not REPORT_FILE.exists():
        print(f"❌ 找不到报告文件: {REPORT_FILE}")
        return
    
    df_report = pd.read_csv(REPORT_FILE)
    # 过滤掉总计行
    df_sections = df_report[df_report['区间'] != '--- 全线总计 ---'].copy()
    
    results = []
    
    for _, row in df_sections.iterrows():
        sp = row['区间']
        # 注意：这里我们查询的是‘规划运行时间(s)’对应的能耗
        target_t = row['历史时间']
        
        # 执行查表
        e_lookup, v_lookup, gap = lookup_energy_from_table(sp, target_t)
        
        results.append({
            
            '区间': sp, '历史T': target_t, '模型E':round(e_lookup, 2) if e_lookup else 0, '实测E': row['规划能耗(Wh)'], 'V_peak': round(v_lookup, 2) if v_lookup else 0
        })
        
        if e_lookup:
            print(f"✅ {sp:15} | 匹配时间:{target_t:6.1f}s | 能耗:{e_lookup:8.2f} Wh")

    # 2. 导出核验表
    df_final = pd.DataFrame(results)
    
    
    df_final.to_csv(os.path.join(OUTPUT_DIR, "playback_trip6_full.csv"), index=False)
    
    print("\n" + "="*50)
    print(f"🏁 查表核验完成！")
    print(f"   - 结果保存至: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_lookup_analysis()