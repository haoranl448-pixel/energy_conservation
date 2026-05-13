# -*- coding: utf-8 -*-
"""
generate_scalers.py
功能：不训练模型，直接读取数据并生成缺失的 scaler_x.pkl 和 scaler_y.pkl
速度：极快 (仅做数据IO和统计计算)
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= 配置 =================
LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

DATA_DIR = "data_processed"
OUTPUT_ROOT = "nn_results_transformer"

# ================= 辅助函数 (保持与训练逻辑一致) =================

def read_multi_day_results(data_dir, station_pair):
    pattern = os.path.join(data_dir, f"results_{station_pair}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files: return None
    dfs = []
    for fp in files:
        try: dfs.append(pd.read_excel(fp))
        except: pass
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True)

def ensure_cols(df, cols, fill=0.0):
    for c in cols:
        if c not in df.columns: df[c] = fill
    return df

def generate_for_section(station_pair):
    target_dir = os.path.join(OUTPUT_ROOT, station_pair)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    print(f"处理区间: {station_pair} ... ", end="")

    # 1. 读取数据
    df = read_multi_day_results(DATA_DIR, station_pair)
    if df is None or df.empty:
        print("❌ 无数据")
        return

    # 2. 特征工程 (必须与 train_transformer.py 完全一致)
    need_cols = ['时刻', '速度(m/s)', '加速度(m/s²)', 'curvature', 'gradient', '重量', 'energy', 'segment']
    df = ensure_cols(df, need_cols)
    
    # 5个特征
    raw_X = pd.DataFrame({
        'velocity': df['速度(m/s)'],
        'acceleration': df['加速度(m/s²)'],
        'curvature': df['curvature'],
        'gradient': df['gradient'],
        'mass': df['重量']
    }).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    raw_y = (pd.to_numeric(df['energy'], errors='coerce').fillna(0).values / 3.6e6) * 1000 
    segments = df['segment'].values

    # 3. 数据切分 (只用训练集拟合 Scaler)
    unique_segs = np.unique(segments)
    split_point = int(len(unique_segs) * 0.8)
    if split_point == 0: split_point = 1
    
    train_segs = unique_segs[:split_point]
    train_mask = np.isin(segments, train_segs)
    
    X_train = raw_X[train_mask]
    y_train = raw_y[train_mask]

    # 4. 拟合 Scaler
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    # 5. 保存
    with open(os.path.join(target_dir, 'scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
        
    with open(os.path.join(target_dir, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
        
    print("✅ 已生成")

if __name__ == "__main__":
    print(">>> 开始补全 Scaler 文件...")
    count = 0
    for sp in LINE5_SECTIONS:
        try:
            generate_for_section(sp)
            count += 1
        except Exception as e:
            print(f"\n❌ {sp} 失败: {e}")
            
    print(f"\n全部完成！共处理 {count} 个区间。")
    print("现在你可以运行 optimize_line5_transformer.py 了。")