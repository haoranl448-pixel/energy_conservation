# -*- coding: utf-8 -*-
"""
simulate_ato_with_comparison.py
第三阶段：ATO 仿真运行 (纯数据驱动风格版) + 真实数据对比
逻辑：
1. 移除人为经验设置的分段加速度限制 (0.55/0.8)，完全信赖神经网络学习到的驾驶风格。
2. 移除固定制动率 (-0.5)，改为由 ATP 计算“极限刹车点”作为安全兜底。
3. 仿真结束后，自动读取真实数据，绘制红蓝对比图。
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================= 仿真参数 =================
MODEL_DIR = "ato_results"
DATA_DIR = "data_processed"     # 真实数据存放目录
STATION_PAIR = "布政-张家潭"    # 用于查找真实数据文件
DT = 0.1                        # 仿真步长 0.1s

# 物理约束条件
MAX_SPEED_KMH = 80.0
MAX_SPEED_MS = MAX_SPEED_KMH / 3.6  # 22.22 m/s

# 车辆物理极限 (非经验限制，而是车辆能力上限，防止飞出宇宙)
MAX_TRACTION_PHYSICAL = 1.2  # 假设车辆最大推力能产生 1.2 m/s²
MAX_BRAKE_PHYSICAL = -1.2    # 假设车辆最大制动能产生 -1.2 m/s²

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

def simulate_and_compare():
    # ---------------------------------------
    # 1. 加载配置与模型
    # ---------------------------------------
    config_path = os.path.join(MODEL_DIR, "train_config.pkl")
    if not os.path.exists(config_path):
        print(" 缺少配置文件，请先运行第一阶段训练代码。")
        return

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    TARGET_L = config['target_dist']
    TARGET_T = config['target_time']
    MASS_VAL = config['mass_mean']
    FEATURE_COLS = config['feature_cols']

    print(f" 仿真目标: {TARGET_L}m, {TARGET_T:.1f}s, 限速 {MAX_SPEED_MS:.2f}m/s")
    print(f" 模式: 纯神经网络驾驶 (去除0.55/0.8限制)，ATP仅作为极限安全底线")

    # 加载Scaler和模型
    scaler_path = os.path.join(MODEL_DIR, "ato_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "ato_model.pth")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    input_dim = len(scaler.mean_)
    model = ATOPolicyNet(input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 加载地图插值
    map_path = os.path.join(MODEL_DIR, "track_map.pkl")
    df_map = pd.read_pickle(map_path)
    s_vals = df_map['累计位移(m)'].values
    
    # 扩展边界
    s_padded = np.concatenate([[-1000.0], s_vals, [TARGET_L + 2000.0]])
    curv_padded = np.concatenate([[0], df_map['curvature'].values, [0]])
    grad_padded = np.concatenate([[0], df_map['gradient'].values, [0]])
    
    f_curv = interp1d(s_padded, curv_padded, kind='nearest')
    f_grad = interp1d(s_padded, grad_padded, kind='nearest')

    # ---------------------------------------
    # 2. 仿真主循环
    # ---------------------------------------
    t = 0.0
    s = 0.0
    v = 0.0
    
    history = {'time': [], 'dist': [], 'velocity': [], 'acceleration': [], 'mode': []}
    
    max_steps = int(TARGET_T * 1.5 / DT)
    
    for step in range(max_steps):
        # 2.1 获取环境数据
        cur_curv = float(f_curv(s))
        cur_grad = float(f_grad(s))
        dist_rem = TARGET_L - s
        time_rem = TARGET_T - t

        # 2.2 神经网络计算 (AI Suggestion)
        input_vec = np.array([[v, cur_curv, cur_grad, MASS_VAL, dist_rem, time_rem]])
        try:
            input_scaled = scaler.transform(input_vec)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                raw_acc = model(input_tensor).item()
        except:
            raw_acc = 0.0

        # 2.3 决策融合 (Decision Fusion)
        final_acc = raw_acc
        control_mode = "AI_Driver" 

        # --- A. 极限安全底线 (ATP Emergency) ---
        # 计算：如果现在不以最大制动力(-1.0)刹车，是否会冲出终点？
        # 给一点余量 (buffer distance) 比如 5米
        safe_brake_limit = -1.0 
        required_brake_dist = (v**2) / (2 * abs(safe_brake_limit))
        
        if dist_rem <= required_brake_dist + 5.0:
            # 只有当AI玩脱了，快撞墙了，ATP才介入
            # 这里我们不直接锁死，而是给一个渐进式的刹车指令，确保停住
            # 计算正好停在终点需要的减速度
            needed_decel = -(v**2) / (2 * dist_rem + 0.1)
            # 限制在物理范围内
            needed_decel = max(needed_decel, MAX_BRAKE_PHYSICAL)
            
            # 只有当 AI 给的刹车力度不够时，ATP 才覆盖
            if final_acc > needed_decel:
                final_acc = needed_decel
                control_mode = "ATP_Rescue"
        
        # --- B. 物理与规则限制 (Soft Limits) ---
        else:
            # B1. 限速保护 (Speed Protection)
            if v >= MAX_SPEED_MS:
                final_acc = min(final_acc, 0.0) # 禁止加速
                control_mode = "Speed_Limit"
            
            # B2. 车辆能力限制 (Physical Clip)
            # 以前是 0.55/0.8，现在只限制绝对物理上限
            final_acc = np.clip(final_acc, MAX_BRAKE_PHYSICAL, MAX_TRACTION_PHYSICAL)
            
            # B3. 停车保持
            if v <= 0.05 and final_acc < 0 and dist_rem < 5:
                final_acc = 0.0
                control_mode = "Stop_Hold"

        # 2.4 状态更新
        v_next = v + final_acc * DT
        if v_next < 0: v_next = 0.0
        s_next = s + v * DT
        
        # 记录
        history['time'].append(t)
        history['dist'].append(s)
        history['velocity'].append(v)
        history['acceleration'].append(final_acc)
        history['mode'].append(control_mode)

        s = s_next
        v = v_next
        t += DT

        # 结束条件
        if s >= TARGET_L and v <= 0.05:
            print(f" 到达终点！耗时: {t:.2f}s")
            break
        if s > TARGET_L + 20:
            print("⚠️ 冲出终点过远！")
            break
        if t > TARGET_T + 30:
            print("⚠️ 超时！")
            break

    # ---------------------------------------
    # 3. 自动对比绘图
    # ---------------------------------------
    df_sim = pd.DataFrame(history)
    csv_save_path = os.path.join(MODEL_DIR, "simulation_result_no_limit.csv")
    df_sim.to_csv(csv_save_path, index=False)
    print(f"✅ 仿真数据已保存: {csv_save_path}")

    # 读取真实数据
    pattern = os.path.join(DATA_DIR, f"results_{STATION_PAIR}*.xlsx")
    files = sorted(glob.glob(pattern))
    if files:
        real_path = files[0]
        print(f"✅ 正在读取真实数据用于对比: {real_path}")
        df_real = pd.read_excel(real_path)
        df_real = df_real.sort_values('时刻')
    else:
        print("⚠️ 未找到真实数据文件，将只绘制仿真曲线。")
        df_real = None

    # 设置绘图字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False 

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # --- 速度对比 ---
    if df_real is not None:
        axes[0].plot(df_real['时刻'], df_real['速度(m/s)'], 
                     color='gray', linestyle='--', alpha=0.6, label='真实驾驶 (Human)')
    axes[0].plot(df_sim['time'], df_sim['velocity'], 
                 color='#1f77b4', linewidth=2.5, label='ATO (无经验限制)')
    axes[0].axhline(MAX_SPEED_MS, color='red', linestyle=':', label='限速 80km/h')
    
    axes[0].set_ylabel('速度 (m/s)')
    axes[0].set_title(f'ATO仿真 vs 真实数据 (去除人为加速度限制) - {STATION_PAIR}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- 加速度对比 ---
    if df_real is not None:
        axes[1].plot(df_real['时刻'], df_real['加速度(m/s²)'], 
                     color='gray', linestyle='--', alpha=0.4, label='真实值')
    axes[1].plot(df_sim['time'], df_sim['acceleration'], 
                 color='#ff7f0e', linewidth=1.5, label='ATO (AI Output)')
    
    # 不再画 0.55/0.8 的线，因为限制已经移除了
    axes[1].set_ylabel('加速度 (m/s²)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- 位移对比 ---
    if df_real is not None:
        axes[2].plot(df_real['时刻'], df_real['累计位移(m)'], 
                     color='gray', linestyle='--', alpha=0.6, label='真实值')
    axes[2].plot(df_sim['time'], df_sim['dist'], 
                 color='#2ca02c', linewidth=2.5, label='ATO')
    
    axes[2].axhline(TARGET_L, color='red', linestyle=':', label='目标终点')
    axes[2].set_ylabel('位移 (m)')
    axes[2].set_xlabel('时间 (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_save_path = os.path.join(MODEL_DIR, "final_comparison_no_limit.png")
    plt.savefig(plot_save_path, dpi=300)
    print(f"✅ 对比图已保存: {plot_save_path}")

if __name__ == "__main__":
    simulate_and_compare()