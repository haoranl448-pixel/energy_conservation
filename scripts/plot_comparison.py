# -*- coding: utf-8 -*-
"""
plot_comparison.py
功能：读取仿真结果与真实行车数据，绘制对比图 (v, a, s)
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接保存图片
import matplotlib.pyplot as plt


# 1. 获取当前脚本所在目录 (e:/energy_conservation/scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (e:/energy_conservation)
project_root = os.path.dirname(current_dir)

# 3. 重新定义路径 (对齐你的文件夹结构)
STATION_PAIR = "布政-张家潭"
TARGET_L = 1450.0 

# 指向根目录下的 data_processed
DATA_DIR = os.path.join(project_root, "data","data_processed")

# 指向 output/models/ato_results
OUTPUT_DIR = os.path.join(project_root, "output", "models", "ato_results")




# ================= 配置 =================
STATION_PAIR = "布政-张家潭"
SIM_RESULT_FILE = os.path.join(project_root, "output", "models","ato_results", "simulation_result.csv")


def plot_comparison():
    # ---------------------------------------
    # 1. 读取仿真数据
    # ---------------------------------------
    if not os.path.exists(SIM_RESULT_FILE):
        print(f"❌ 未找到仿真结果文件：{SIM_RESULT_FILE}，请先运行仿真脚本。")
        return
    
    df_sim = pd.read_csv(SIM_RESULT_FILE)
    print(f"✅ 已加载仿真数据，共 {len(df_sim)} 条记录")

    # ---------------------------------------
    # 2. 读取真实数据
    # ---------------------------------------
    pattern = os.path.join(DATA_DIR, f"results_{STATION_PAIR}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ 未找到真实数据文件：{pattern}")
        return
    
    real_data_path = files[0]
    print(f"✅ 正在加载真实数据用于对比：{real_data_path}")
    df_real = pd.read_excel(real_data_path)

    # 简单清洗，确保绘图不报错
    df_real = df_real.sort_values('时刻')
    
    # 对齐时间轴（可选）：如果真实数据不是从0开始，将其平移
    # 假设真实数据的'时刻'就是相对发车时间的秒数，如果不是，需要取消下面这行的注释
    # df_real['时刻'] = df_real['时刻'] - df_real['时刻'].min()

    # ---------------------------------------
    # 3. 绘图 (三联图)
    # ---------------------------------------
    # 设置中文字体，防止乱码 (尝试常见中文字体)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # --- 子图 1: 速度对比 ---
    # 真实数据
    axes[0].plot(df_real['时刻'], df_real['速度(m/s)'], 
                 color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='真实值 (Actual)')
    # 仿真数据
    axes[0].plot(df_sim['time'], df_sim['velocity'], 
                 color='#1f77b4', linewidth=2.5, label='ATO仿真 (Simulated)')
    
    axes[0].set_ylabel('速度 Velocity (m/s)', fontsize=12)
    axes[0].set_title(f'ATO仿真 vs 真实数据对比 ({STATION_PAIR})', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # --- 子图 2: 加速度对比 ---
    # 真实数据 (加速度通常波动很大，透明度调低一点)
    axes[1].plot(df_real['时刻'], df_real['加速度(m/s²)'], 
                 color='gray', linestyle='--', linewidth=1, alpha=0.5, label='真实值 (Actual)')
    # 仿真数据
    axes[1].plot(df_sim['time'], df_sim['acceleration'], 
                 color='#ff7f0e', linewidth=2, label='ATO仿真 (Simulated)')
    
    # 标出关键限制线
    axes[1].axhline(0.55, color='green', linestyle=':', alpha=0.5, label='牵引限制 0.55')
    axes[1].axhline(0.8, color='green', linestyle=':', alpha=0.5, label='牵引限制 0.8')
    axes[1].axhline(-0.5, color='red', linestyle=':', alpha=0.5, label='制动限制 -0.5')

    axes[1].set_ylabel('加速度 Accel (m/s²)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # --- 子图 3: 位移对比 ---
    # 真实数据
    axes[2].plot(df_real['时刻'], df_real['累计位移(m)'], 
                 color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='真实值 (Actual)')
    # 仿真数据
    axes[2].plot(df_sim['time'], df_sim['dist'], 
                 color='#2ca02c', linewidth=2.5, label='ATO仿真 (Simulated)')
    
    # 标出目标终点
    target_dist = df_sim['dist'].iloc[-1] # 取仿真最终距离作为参考，或者1450
    axes[2].axhline(1450, color='red', linestyle=':', alpha=0.5, label='目标终点 1450m')

    axes[2].set_ylabel('位移 Distance (m)', fontsize=12)
    axes[2].set_xlabel('时间 Time (s)', fontsize=12)
    axes[2].legend(loc='lower right')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    # ---------------------------------------
    # 4. 保存
    # ---------------------------------------
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "final_comparison_vs_real.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 对比图已生成并保存至: {save_path}")

if __name__ == "__main__":
    plot_comparison()