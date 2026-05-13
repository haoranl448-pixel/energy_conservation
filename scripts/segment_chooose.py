# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================= 1. 路径与参数配置 =================
# 原始数据目录
DATA_DIR = Path(r"D:\energy_conservation\data\data_processed")
# 结果输出总目录
ANALYSIS_BASE = Path(r"D:\energy_conservation\output\analysis\vst_full_batch_report_chosen_new")
# 清洗后数据存放目录
CLEANED_DATA_DIR = Path(r"D:\energy_conservation\data\data_processed_new")

# --- 你需要修改的参数 ---
STATION_PAIR = ["南高教园区-下应路"]
SEGMENTS_TO_REMOVE = [19,        26,34     ,6,14,18,26,33,      34,31,33,32,30,29,28,27,26] # 在这里填入你想删除的 ID
#SEGMENTS_TO_REMOVE = [18,26,33,6,14,18,7,12,14,15,16,18,20,24,26]
# ================= 2. 文件夹初始化 =================
# 创建结构: analysis/vst_full_batch_report/站名/individual_segments
# station_folder = ANALYSIS_BASE / STATION_PAIR
# individual_folder = station_folder / "individual_segments"
# station_folder.mkdir(parents=True, exist_ok=True)
# individual_folder.mkdir(parents=True, exist_ok=True)
# CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def process_and_plot_vst():
    
    for sp in STATION_PAIR:
        file_path = DATA_DIR / f"results_{sp}.xlsx"
        if not file_path.exists():
            print(f"❌ 找不到文件: {file_path}")
            return
        station_folder = ANALYSIS_BASE / sp
        individual_folder = station_folder / "individual_segments"
        station_folder.mkdir(parents=True, exist_ok=True)
        individual_folder.mkdir(parents=True, exist_ok=True)
        CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)



        # A. 读取与清洗
        print(f"🚀 正在处理: {sp}")
        df = pd.read_excel(file_path)


        cols_to_fix = ['速度(m/s)', '累计位移(m)', '时刻', 'energy']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 剔除掉关键列含有 NaN 的行
        df = df.dropna(subset=['速度(m/s)', '累计位移(m)', '时刻'])


        df_cleaned = df[~df['segment'].isin(SEGMENTS_TO_REMOVE)].copy()
        




        # 保存清洗后的 Excel
        cleaned_excel_path = CLEANED_DATA_DIR / f"cleaned_{sp}.xlsx"
        df_cleaned.to_excel(cleaned_excel_path, index=False)

        # B. 绘图设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        remaining_segs = sorted(df_cleaned['segment'].unique())

        # --- 逻辑 1: 生成合并图 (Merged Plot) ---
        fig_merged, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        for seg_id in remaining_segs:
            seg_data = df_cleaned[df_cleaned['segment'] == seg_id].copy()
            t_rel = seg_data['时刻'] - seg_data['时刻'].iloc[0]
            v_kmh = seg_data['速度(m/s)'] * 3.6
            s_m = seg_data['累计位移(m)']

            # 添加到合并图
            ax1.plot(t_rel, v_kmh, label=f"Seg {seg_id}", alpha=0.7)
            ax2.plot(s_m, v_kmh, alpha=0.7)

            # --- 逻辑 2: 生成单趟详情图 (Individual Plots) ---
            fig_ind, (iax1, iax2) = plt.subplots(2, 1, figsize=(10, 8))
            iax1.plot(t_rel, v_kmh, color='dodgerblue', linewidth=2)
            iax1.set_title(f"Segment {seg_id} - 速度曲线", fontsize=12)
            iax1.set_ylabel("速度 (km/h)")
            iax1.grid(True, alpha=0.3)

            iax2.plot( s_m, v_kmh, color='seagreen', linewidth=2)
            iax2.set_title(f"Segment {seg_id} - 位移曲线", fontsize=12)
            iax2.set_ylabel("位移 (m)")
            iax2.set_xlabel("时间 (s)")
            iax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_ind.savefig(individual_folder / f"seg_{seg_id}_vst.png", dpi=150)
            plt.close(fig_ind) # 及时关闭节省内存

            # 完成合并图的装饰
            ax1.set_title(f"{sp} 全景汇总 - 速度 (v-t)", fontsize=15)
            ax1.set_ylabel("速度 (km/h)")
            ax1.grid(True, alpha=0.3)
            if len(remaining_segs) < 20: ax1.legend(loc='upper right', ncol=3, fontsize=8)

            ax2.set_title(f"全景汇总 - 位移 (v-s)", fontsize=15)
            ax2.set_ylabel("位移 (m)")
            ax2.set_xlabel("时间 (s)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            merged_plot_path = station_folder / f"merged_vst_{sp}.png"
            fig_merged.savefig(merged_plot_path, dpi=300)
            plt.close(fig_merged)

            print(f"✅ 处理完成！")
            print(f"📂 合并图与单趟图位置: {station_folder}")
            print(f"📄 清洗后Excel位置: {cleaned_excel_path}")

if __name__ == "__main__":
    process_and_plot_vst()