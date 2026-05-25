# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ================= 1. 路径与配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 数据来源目录
DATA_DIR = os.path.join(project_root, "data", "data_processed")
# 输出目录
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "raw_service_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标服务号
TARGET_SERVICES = ["53004", "53006"]

# 26 站严格顺序
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ================= 2. 数据拼接逻辑 =================

def get_raw_trip_data(service_id):
    print(f"正在提取服务号 {service_id} 的原始全线数据...")
    
    all_segments = []
    current_time_offset = 0.0  # 全局时间偏移
    current_energy_offset = 0.0  # 全局能耗偏移
    
    for sp in LINE5_SECTIONS:
        file_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        if not os.path.exists(file_path):
            continue
            
        # 读取已处理的原始数据
        df = pd.read_excel(file_path)
        
        # 稳健的服务号匹配（处理 53004 或 53004.0）
        df['ser_str'] = df['服务号'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
        df_seg = df[df['ser_str'] == str(service_id)].sort_values('时刻')
        
        if df_seg.empty:
            continue

        # 提取时间、速度和原始步进能耗
        t_local = df_seg['时刻'].values
        v_raw = df_seg['速度(m/s)'].values
        
        # 原始能耗：在你的 data_process.py 中，'energy' 列是网压*电流*dt 得到的实时焦耳/瓦秒
        # 如果你想看累积，我们要对这一列求和
        e_raw_steps = df_seg['energy'].values / 3.6e6  # 将 J(Ws) 转换为 kWh
        
        # 构造这一段的连续轨迹
        df_trip_seg = pd.DataFrame({
            'time': t_local + current_time_offset,
            'velocity': v_raw,
            'velocity_kmh': v_raw * 3.6,
            'section': sp
        })
        
        # 计算全局连续的累积能耗
        seg_cum_energy = np.cumsum(e_raw_steps) + current_energy_offset
        df_trip_seg['cum_energy_kwh'] = seg_cum_energy
        
        all_segments.append(df_trip_seg)
        
        # 更新偏移量：段末时间 + 30秒停站；能量直接继承上一段末尾
        current_time_offset = df_trip_seg['time'].iloc[-1] + 30.0 
        current_energy_offset = df_trip_seg['cum_energy_kwh'].iloc[-1]

    if not all_segments:
        return None
        
    return pd.concat(all_segments, ignore_index=True)

# ================= 3. 绘图与导出 =================

def main():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus']=False

    for sid in TARGET_SERVICES:
        df_trip = get_raw_trip_data(sid)
        
        if df_trip is None:
            print(f"❌ 未能找到服务号 {sid} 的全线数据")
            continue

        # 1. 导出原始拼接表
        save_csv = os.path.join(OUTPUT_DIR, f"raw_full_trip_{sid}.csv")
        df_trip.to_csv(save_csv, index=False)
        
        # 2. 绘制可视化图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # --- 子图 1: 原始 v-t ---
        ax1.plot(df_trip['time'], df_trip['velocity_kmh'], color='tab:blue', lw=1.2, label='原始速度监测')
        ax1.set_ylabel("速度 (km/h)", fontsize=12)
        ax1.set_title(f"服务号 {sid} 全线原始行车速度曲线 (v-t)", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, ls=':')
        
        # --- 子图 2: 原始 e-t ---
        ax2.plot(df_trip['time'], df_trip['cum_energy_kwh'], color='tab:red', lw=2, label='原始电网侧能耗')
        ax2.fill_between(df_trip['time'], df_trip['cum_energy_kwh'], color='tab:red', alpha=0.1)
        ax2.set_ylabel("累积消耗电量 (kWh)", fontsize=12)
        ax2.set_xlabel("时间 (s)", fontsize=12)
        ax2.set_title(f"服务号 {sid} 全线原始累积能耗曲线 (e-t) - 总计: {df_trip['cum_energy_kwh'].iloc[-1]:.2f} kWh", fontsize=14)
        ax2.grid(True, alpha=0.3, ls=':')

        # 标出站间分界线
        section_starts = df_trip.groupby('section')['time'].min()
        for sec_name, t_start in section_starts.items():
            ax1.axvline(t_start, color='gray', lw=0.6, ls='--', alpha=0.4)
            ax2.axvline(t_start, color='gray', lw=0.6, ls='--', alpha=0.4)

        plt.tight_layout()
        save_plot = os.path.join(OUTPUT_DIR, f"raw_trip_plot_{sid}.png")
        plt.savefig(save_plot, dpi=300)
        plt.close()
        
        print(f"✅ 完成！表格已存至: {save_csv}")
        print(f"✅ 完成！图表已存至: {save_plot}")

if __name__ == "__main__":
    main()