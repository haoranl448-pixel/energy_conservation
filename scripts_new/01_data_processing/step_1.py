# -*- coding: utf-8 -*-
import pandas as pd
import os
from pathlib import Path

# ================= 配置路径 =================
DATA_DIR = Path(r"D:\energy_conservation\data\data_processed")
TEMPLATE_CSV = Path(r"data/static/section_params.csv")
OUTPUT_CSV = Path(r"data/static/section_params_trip1.csv")
TRIP_INDEX = 0 # 第1趟

STATIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

def generate_input():
    # 1. 提取历史 Trip 6 信息
    real_masses = []
    total_run_time = 0
    
    print(f"🔍 正在从数据中提取第 {TRIP_INDEX+1} 趟车的真实参数...")
    
    for sp in STATIONS:
        file_path = DATA_DIR / f"results_{sp}.xlsx"
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 强制转换类型防止报错
        for col in ['速度(m/s)', '时刻', '重量']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        segs = sorted(df['segment'].unique())
        target_seg = segs[TRIP_INDEX]
        data = df[df['segment'] == target_seg].copy().dropna(subset=['速度(m/s)'])
        
        # 记录载重和时间
        mass = data['重量'].mean()
        run_dur = data['时刻'].iloc[-1] - data['时刻'].iloc[0]
        
        real_masses.append(mass)
        total_run_time += run_dur

    # 2. 读取并修改 CSV 模板
    df_params = pd.read_csv(TEMPLATE_CSV)
    # 确保站间距顺序一致，然后替换最后一列 MASS
    # 假设 CSV 里的 station_pair 顺序和我们的 STATIONS 列表一致
    df_params['MASS'] = real_masses
    
    # 3. 保存新文件
    df_params.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 50)
    print(f"✅ 新配置文件已生成: {OUTPUT_CSV}")
    print(f"🚀 核心数据如下：")
    print(f"   >>> 第 1 趟列车全线总运行时间 (不含停站): {total_run_time:.2f} 秒")
    print(f"   >>> 请将此时间输入到你的动态规划代码中跑优化。")
    print("-" * 50)

if __name__ == "__main__":
    generate_input()