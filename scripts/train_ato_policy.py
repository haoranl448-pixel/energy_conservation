# -*- coding: utf-8 -*-
"""
train_ato_policy.py
第一阶段：训练 ATO 策略模型
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
import os

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
# # ================= 配置参数 =================
# STATION_PAIR = "布政-张家潭"
# TARGET_L = 1450.0  # 目标里程 (m) - 将保存到config供仿真使用
# DATA_DIR = "data_processed"
# OUTPUT_DIR = "ato_results"

# 特征列：状态量
FEATURE_COLS = ['速度(m/s)', 'curvature', 'gradient', '重量', 'dist_rem', 'time_rem']
# 标签列：动作量
TARGET_COL = '加速度(m/s²)'

# -------------------------------------------
# 1. 定义网络结构 (MLP)
# -------------------------------------------
class ATOPolicyNet(nn.Module):
    def __init__(self, input_dim):
        super(ATOPolicyNet, self).__init__()
        # 增加网络容量以拟合真实数据的波动特征
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.01)
        return self.output(x)

def train_ato():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------
    # 2. 读取与预处理数据
    # -------------------------------------------
    pattern = os.path.join(DATA_DIR, f"results_{STATION_PAIR}*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ 未找到数据文件：{pattern}")
        return

    print(f"正在读取文件：{files[0]} ...")
    df = pd.read_excel(files[0])

    # 自动获取数据的最大时间和位移，用于对齐
    data_max_time = df['时刻'].max()
    data_max_dist = df['累计位移(m)'].max()
    print(f"📊 数据源概况: 总耗时={data_max_time:.2f}s, 总里程={data_max_dist:.2f}m")
    
    # 这里的Target Time暂时取数据的最大值，保证模型学到的"剩余时间"是有意义的
    # 如果想强制模型学跑得更快，可以手动指定一个小一点的值，但会导致标签不匹配




    # target_time_train = data_max_time

    target_time_train = 111


    # 计算剩余量特征
    df['dist_rem'] = TARGET_L - df['累计位移(m)']
    df['time_rem'] = target_time_train - df['时刻']
    
    # 获取平均重量（仿真用）
    mass_mean = df['重量'].mean()

    # 清洗数据 (去空值、去无穷大)
    df_clean = df[FEATURE_COLS + [TARGET_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    
    X_raw = df_clean[FEATURE_COLS].values
    y_raw = df_clean[TARGET_COL].values

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------
    # 3. 训练流程
    # -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    model = ATOPolicyNet(input_dim=X_train_scaled.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # 均方误差，不仅学趋势，也学波动

    print(" 开始训练 ATO 策略网络...")
    epochs = 400
    train_hist = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        train_hist.append(loss.item())
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # -------------------------------------------
    # 4. 保存模型与配置
    # -------------------------------------------
    # 保存训练配置，确保仿真时知道用什么基准
    config = {
        'target_dist': TARGET_L,
        'target_time': target_time_train,
        'mass_mean': mass_mean,
        'feature_cols': FEATURE_COLS
    }
    with open(os.path.join(OUTPUT_DIR, "train_config.pkl"), "wb") as f:
        pickle.dump(config, f)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "ato_model.pth"))
    with open(os.path.join(OUTPUT_DIR, "ato_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # 保存线路地图 (用于仿真查坡度/曲率)
    track_map = df[['累计位移(m)', 'curvature', 'gradient']].sort_values('累计位移(m)').drop_duplicates('累计位移(m)')
    track_map.to_pickle(os.path.join(OUTPUT_DIR, "track_map.pkl"))

    print(f"✅ 训练完成！配置文件已保存，仿真将基于均重 {mass_mean:.2f} 吨 进行。")
    
    # 简单画个Loss图
    plt.figure()
    plt.plot(train_hist)
    plt.title('Training Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"))
    plt.close()

if __name__ == "__main__":
    train_ato()



