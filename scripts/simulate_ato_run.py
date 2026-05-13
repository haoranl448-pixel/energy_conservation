# -*- coding: utf-8 -*-
"""
scripts/simulate_ato_run.py
ATO 仿真运行逻辑封装版
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================= 路径配置 =================

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 修正这里的路径，指向 output/models/ato_results
MODEL_DIR = os.path.join(project_root, "output", "models", "ato_results")
# 物理约束
MAX_SPEED_MS = 80.0 / 3.6 
FIXED_DECEL = -0.5
DT = 0.1 

def get_traction_limit(v_current):
    return 0.55 if v_current < 12.5 else 0.8

# ================= 网络定义 =================
class ATOPolicyNet(nn.Module):
    def __init__(self, input_dim):
        super(ATOPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.01)
        return self.output(x)

# ================= 核心仿真函数 =================
def simulate_logic(station_pair, target_t=None):
    """
    station_pair: 站对名称
    target_t: 外部指定的目标时间
    """
    config_path = os.path.join(MODEL_DIR, "train_config.pkl")
    if not os.path.exists(config_path):
        print(f"❌ 缺少配置文件: {config_path}")
        return None

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    TARGET_L = config['target_dist']
    # 如果指定了优化时间，则覆盖模型默认值
    TARGET_T = target_t if target_t is not None else config['target_time']
    MASS_VAL = config['mass_mean']

    # 加载模型
    scaler_path = os.path.join(MODEL_DIR, "ato_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "ato_model.pth")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    input_dim = len(scaler.mean_)
    model = ATOPolicyNet(input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 加载地图
    map_path = os.path.join(MODEL_DIR, "track_map.pkl")
    df_map = pd.read_pickle(map_path)
    s_vals = df_map['累计位移(m)'].values
    s_padded = np.concatenate([[-1000.0], s_vals, [TARGET_L + 2000.0]])
    curv_padded = np.concatenate([[0], df_map['curvature'].values, [0]])
    grad_padded = np.concatenate([[0], df_map['gradient'].values, [0]])
    f_curv = interp1d(s_padded, curv_padded, kind='nearest')
    f_grad = interp1d(s_padded, grad_padded, kind='nearest')

    # 仿真循环
    t, s, v = 0.0, 0.0, 0.0
    history = {'time': [], 'dist': [], 'velocity': [], 'acceleration': [], 'mode': []}
    max_steps = int(TARGET_T * 1.5 / DT)
    
    for _ in range(max_steps):
        cur_curv, cur_grad = float(f_curv(s)), float(f_grad(s))
        dist_rem, time_rem = TARGET_L - s, TARGET_T - t

        # AI 决策
        input_vec = np.array([[v, cur_curv, cur_grad, MASS_VAL, dist_rem, time_rem]])
        input_scaled = scaler.transform(input_vec)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            raw_acc = model(input_tensor).item()

        # 规则介入与安全保护
        final_acc = raw_acc
        mode = "AI_Cruise"
        required_brake_dist = (v**2) / (2 * abs(FIXED_DECEL))
        
        if dist_rem <= required_brake_dist + 2.0:
            final_acc = FIXED_DECEL
            mode = "ATP_Brake"
            if v < 0.2 and dist_rem < 0.2: final_acc = -v / DT
        else:
            if v >= MAX_SPEED_MS:
                final_acc = min(final_acc, 0.0)
                mode = "Speed_Limit"
            max_traction = get_traction_limit(v)
            if final_acc > max_traction:
                final_acc = max_traction
                mode = "Trac_Limit"
            if final_acc < FIXED_DECEL: final_acc = FIXED_DECEL

        if v <= 0.01 and final_acc < 0: final_acc, mode = 0.0, "Stop_Hold"

        # 状态更新
        v_next = max(v + final_acc * DT, 0.0)
        s_next = s + v * DT
        history['time'].append(t); history['dist'].append(s)
        history['velocity'].append(v); history['acceleration'].append(final_acc)
        history['mode'].append(mode)
        s, v, t = s_next, v_next, t + DT

        if (dist_rem < 0.5 and v < 0.1) or (s >= TARGET_L and v <= 0.0):
            break

    return pd.DataFrame(history)

if __name__ == "__main__":
    # 默认独立运行测试
    df = simulate_logic("布政-张家潭")
    if df is not None:
        print(f"测试运行结束，实际耗时: {df['time'].iloc[-1]:.2f}s")