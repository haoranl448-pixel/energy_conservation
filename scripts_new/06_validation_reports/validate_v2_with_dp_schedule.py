# -*- coding: utf-8 -*-
import os, pickle, re, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# ================= 1. 配置 =================
STATION_PAIR = "布政-张家潭"
DT = 0.1 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(project_root, "output", "models", "ato_results_v3")
REF_CURVE_FILE = os.path.join(project_root, "output", "schedule", "class_based_results", "segments", STATION_PAIR, "vt_selected.csv")
PROC_FILE = os.path.join(project_root, "data", "data_processed", f"results_{STATION_PAIR}.xlsx")

class ATOPolicyNetV3(nn.Module):
    def __init__(self, input_dim):
        super(ATOPolicyNetV3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.fc(x)

def run_validation():
    # A. 加载资源
    with open(os.path.join(MODEL_DIR, "ato_scaler_v3.pkl"), "rb") as f: scaler = pickle.load(f)
    model = ATOPolicyNetV3(input_dim=7)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ato_model_v3.pth"), map_location='cpu'))
    model.eval()

    # B. 读取 115s 规划数据
    df_ref = pd.read_csv(REF_CURVE_FILE).sort_values('s_local(m)')
    target_l = df_ref['s_local(m)'].max()
    target_t = df_ref['t_local'].max()
    f_v_ref = interp1d(df_ref['s_local(m)'], df_ref['v(m/s)'], kind='linear', bounds_error=False, fill_value=0)

    # C. 读取路况
    df_phys = pd.read_excel(PROC_FILE)
    mass_val = df_phys['重量'].mean()
    f_curv = interp1d(df_phys['累计位移(m)'], df_phys['curvature'], kind='nearest', fill_value="extrapolate")
    f_grad = interp1d(df_phys['累计位移(m)'], df_phys['gradient'], kind='nearest', fill_value="extrapolate")

    # D. 仿真变量
    t, s, v = 0.0, 0.0, 0.0
    history = {'t': [], 'v': [], 'a': [], 'v_ref': []}
    
    # 平滑辅助变量
    last_acc = 0.5
    
    # ⭐ 核心控制参数
    Kp = 2.5      # 保持强力反馈，压住速度
    Alpha = 0.15  # 滤波系数：0.1-0.2 之间，专门用来磨平“密密麻麻”的锯齿

    print(f"🚀 验证开始 | 目标: {target_t}s")

    for _ in range(2000):
        s_lookup = min(s, target_l)
        v_ref_curr = float(f_v_ref(s_lookup))
        dist_rem = max(target_l - s, 0)
        
        # 1. 偏差反馈 (上一版证明最有效的逻辑)
        v_diff = v - v_ref_curr
        v_input = v + Kp * v_diff if v_diff > 0 else v

        # 2. AI 决策
        input_raw = np.array([[v_input, float(f_curv(s_lookup)), float(f_grad(s_lookup)), mass_val, dist_rem, 3.0, v_ref_curr]])
        input_scaled = scaler.transform(input_raw)
        with torch.no_grad():
            acc_raw = model(torch.tensor(input_scaled, dtype=torch.float32)).item()

        # 3. 强限制 (上一版的精髓，但改温和一点点防止过抖)
        if v > v_ref_curr + 0.5:
            acc_raw = min(acc_raw, -0.3)

        # 4. ⭐【核心改进】一阶滤波：磨平锯齿
        # 这一行会让加速度曲线变圆润，不再是密集的直线
        acc_smooth = Alpha * acc_raw + (1 - Alpha) * last_acc
        last_acc = acc_smooth
        acc_final = np.clip(acc_smooth, -1.2, 1.0)

        # 5. 【核心改进】软停车逻辑：解决“陡然下降”
        # 当距离终点很近时，蓝线必须跟随红线的减速趋势
        if dist_rem < 5.0:
            # 参考红线末端的制动趋势
            acc_final = min(acc_final, -0.6)

        # 6. 精准进站判定
        if dist_rem < 0.15 and v < 0.3:
            s, v, acc_final = target_l, 0.0, 0.0
            history['t'].append(t); history['v'].append(v); history['a'].append(acc_final); history['v_ref'].append(v_ref_curr)
            break

        # 更新
        v = max(v + acc_final * DT, 0.0)
        s += v * DT
        t += DT
        history['t'].append(t); history['v'].append(v); history['a'].append(acc_final); history['v_ref'].append(v_ref_curr)

        if t > target_t + 20: break

    # E. 绘图对比
    res = pd.DataFrame(history)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(res['t'], res['v']*3.6, label='AI 实际驾驶速度', lw=2.5)
    plt.plot(df_ref['t_local'], df_ref['v(m/s)']*3.6, 'r--', label='调度规划目标 (115s)', alpha=0.8)
    plt.title(f"V3 最终验证 | 实际用时: {t:.2f}s | 规划目标: {target_t:.2f}s")
    plt.ylabel("速度 (km/h)"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(212)
    plt.plot(res['t'], res['a'], color='orange', label='AI 输出加速度 (已平滑处理)')
    plt.axhline(0, color='black', lw=1, alpha=0.3)
    plt.ylabel("加速度 (m/s²)"); plt.xlabel("时间 (s)"); plt.grid(True, alpha=0.3); plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation()