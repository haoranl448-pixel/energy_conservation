# -*- coding: utf-8 -*-
"""
scripts/validate_opt_schedule.py (AI 风格还原 + 综合能耗结算版)
逻辑：
1. 还原 AI 驾驶风格，保留加速度波动，并与真实历史数据对比。
2. 集成物理引擎 (Physics) 与残差模型 (Transformer) 计算总能耗。
3. 自动匹配时刻表目标时间。
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径与环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入物理模型
from src.physics.train_simu import TrainTheoreticalEnergyModel

# 路径对齐
DATA_DIR = os.path.join(project_root, "data_processed")
ATO_MODEL_DIR = os.path.join(project_root, "output", "models", "ato_results")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
SCHEDULE_FILE = os.path.join(project_root, "output", "schedule", "schedule_results_residual", "schedule_table.csv")

DT = 0.1 
MAX_SPEED_MS = 80.0 / 3.6 
SEQ_LEN_RES = 30 # 残差模型要求的窗口长度

# ================= 2. 网络定义 (需与训练时严格一致) =================

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        out = self.transformer(src)
        return self.decoder(out[:, -1, :])

# ================= 3. 核心计算模块 =================

def run_ai_drive(target_t):
    """ATO 仿真控车逻辑"""
    with open(os.path.join(ATO_MODEL_DIR, "train_config.pkl"), "rb") as f:
        config = pickle.load(f)
    TARGET_L = config['target_dist']
    MASS_VAL = config['mass_mean']

    with open(os.path.join(ATO_MODEL_DIR, "ato_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = ATOPolicyNet(len(scaler.mean_))
    model.load_state_dict(torch.load(os.path.join(ATO_MODEL_DIR, "ato_model.pth"), map_location='cpu'))
    model.eval()

    df_map = pd.read_pickle(os.path.join(ATO_MODEL_DIR, "track_map.pkl"))
    f_curv = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")
    f_grad = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")

    t, s, v = 0.0, 0.0, 0.0
    history = {'time': [], 'dist': [], 'velocity': [], 'acceleration': []}
    
    for _ in range(int(target_t * 2.0 / DT)):
        dist_rem, time_rem = TARGET_L - s, target_t - t
        input_scaled = scaler.transform([[v, float(f_curv(s)), float(f_grad(s)), MASS_VAL, dist_rem, time_rem]])
        with torch.no_grad():
            raw_acc = model(torch.tensor(input_scaled, dtype=torch.float32)).item()

        # 物理截断与安全兜底
        final_acc = np.clip(raw_acc, -1.2, 1.0) 
        req_dist = (v**2) / (2 * 0.8) 
        if dist_rem <= req_dist + 2.0:
            needed_decel = -(v**2) / (2 * dist_rem + 0.1)
            final_acc = max(needed_decel, -1.2)
        if v >= MAX_SPEED_MS: final_acc = min(final_acc, 0.0)
        if v <= 0.01 and final_acc < 0: final_acc = 0.0

        v = max(v + final_acc * DT, 0.0); s += v * DT; t += DT
        history['time'].append(t); history['dist'].append(s)
        history['velocity'].append(v); history['acceleration'].append(final_acc)
        if s >= TARGET_L and v <= 0.05: break

    return pd.DataFrame(history), TARGET_L, MASS_VAL

def compute_total_energy_fusion(station_pair, df_sim, mass_val):
    """集成物理+残差的能耗结算"""
    sim_engine = TrainTheoreticalEnergyModel()
    t_seq, v_seq = df_sim['time'].values, df_sim['velocity'].values
    # A. 物理基准能耗
    e_phy_steps = sim_engine.run_batch_simulation(t_seq, v_seq, mass_val) # Wh
    
    # B. 残差预测能耗
    res_path = os.path.join(RES_MODEL_BASE, station_pair)
    if not os.path.exists(os.path.join(res_path, "best_res_model.pth")):
        return np.sum(e_phy_steps), np.sum(e_phy_steps)

    with open(os.path.join(res_path, "scaler_x.pkl"), "rb") as f: scaler_x = pickle.load(f)
    with open(os.path.join(res_path, "scaler_y.pkl"), "rb") as f: scaler_y = pickle.load(f)
    res_model = ResidualTransformer(input_dim=5)
    res_model.load_state_dict(torch.load(os.path.join(res_path, "best_res_model.pth"), map_location='cpu'))
    res_model.eval()

    a_seq = np.zeros_like(v_seq)
    a_seq[1:] = np.diff(v_seq) / DT
    dist_seq = df_sim['dist'].values
    grad_seq = np.array([sim_engine._interp_gradient(d) for d in dist_seq])
    
    raw_X = np.stack([v_seq, a_seq, e_phy_steps, grad_seq, np.full_like(v_seq, mass_val)], axis=1)
    scaled_X = scaler_x.transform(raw_X)

    res_wh = []
    for i in range(len(scaled_X)):
        if i < SEQ_LEN_RES - 1:
            res_wh.append(0.0); continue
        window = torch.tensor(scaled_X[i-SEQ_LEN_RES+1:i+1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = res_model(window).item()
        res_wh.append(scaler_y.inverse_transform([[p]])[0][0])
    
    return np.sum(e_phy_steps) + np.sum(res_wh), np.sum(e_phy_steps)

# ================= 4. 主程序 =================

def main():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
    
    # 1. 加载时刻表
    if not os.path.exists(SCHEDULE_FILE):
        print(f"❌ 找不到时刻表: {SCHEDULE_FILE}"); return
    df_table = pd.read_csv(SCHEDULE_FILE)
    target_sp = "布政-张家潭"
    row = df_table[df_table['station_pair'] == target_sp].iloc[0]
    t_opt, e_theory_dp = row['t_arrival(s)'], row['energy(Wh)']

    print(f"\n🚀 正在验证最优调度方案: {target_sp}")
    
    # 2. AI 司机驾驶仿真
    df_sim, s_target, mass = run_ai_drive(t_opt)
    
    # 3. 总能耗结算 (物理+残差)
    e_actual_total, e_phy = compute_total_energy_fusion(target_sp, df_sim, mass)
    
    # 4. 读取真实历史数据背景
    pattern = os.path.join(DATA_DIR, f"results_{target_sp}*.xlsx")
    real_files = glob.glob(pattern)
    df_real = pd.read_excel(real_files[0]) if real_files else None

    # 5. 打印结算报告
    t_actual = df_sim['time'].iloc[-1]
    print(f"\n{'*' * 20} 综合结算报告 {'*' * 20}")
    print(f"🏁 到达用时: {t_actual:.2f} s (调度目标: {t_opt:.2f} s)")
    print(f"🔌 调度预测能耗: {e_theory_dp:.2f} Wh")
    print(f"🔋 仿真实际总能耗: {e_actual_total:.2f} Wh")
    print(f"   ↳ 物理能耗: {e_phy:.2f} Wh")
    print(f"   ↳ 残差能耗: {e_actual_total - e_phy:.2f} Wh")
    print(f"📈 方案偏差: {(e_actual_total - e_theory_dp)/e_theory_dp:.2%}")
    print("*" * 54)

    # 6. 绘图对比 (还原右图风格)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # --- 速度对比 ---
    if df_real is not None:
        axes[0].plot(df_real['时刻'], df_real['速度(m/s)'], color='gray', ls='--', alpha=0.5, label='历史真实值 (Actual)')
    axes[0].plot(df_sim['time'], df_sim['velocity'], color='#1f77b4', lw=2.5, label='ATO执行优化方案 (Simulated)')
    axes[0].set_ylabel("速度 Velocity (m/s)"); axes[0].legend(); axes[0].grid(True, alpha=0.2)
    axes[0].set_title(f"ATO 验证: {target_sp} | 总能耗: {e_actual_total:.1f}Wh (调度预测: {e_theory_dp:.1f}Wh)")

    # --- 加速度波动 (AI 风格) ---
    if df_real is not None:
        axes[1].plot(df_real['时刻'], df_real['加速度(m/s²)'], color='gray', ls='--', alpha=0.3, label='历史真实值')
    axes[1].plot(df_sim['time'], df_sim['acceleration'], color='#ff7f0e', lw=1.5, label='ATO 动作波动')
    axes[1].axhline(0.55, color='g', ls=':', alpha=0.4, label='牵引限制')
    axes[1].axhline(-0.5, color='r', ls=':', alpha=0.4, label='制动限制')
    axes[1].set_ylabel("加速度 Accel (m/s²)"); axes[1].legend(); axes[1].grid(True, alpha=0.2)

    # --- 位移对比 ---
    if df_real is not None:
        axes[2].plot(df_real['时刻'], df_real['累计位移(m)'], color='gray', ls='--', alpha=0.5, label='历史真实值')
    axes[2].plot(df_sim['time'], df_sim['dist'], color='#2ca02c', lw=2.5, label='ATO 行驶轨迹')
    axes[2].axhline(s_target, color='r', ls=':', label='目标终点')
    axes[2].set_ylabel("位移 Distance (m)"); axes[2].set_xlabel("时间 Time (s)"); axes[2].legend(); axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    save_p = os.path.join(ATO_MODEL_DIR, "validation_full_energy_report.png")
    plt.savefig(save_p, dpi=300)
    print(f"✅ 综合报告图表已保存: {save_p}")

if __name__ == "__main__":
    main()



# # -*- coding: utf-8 -*-
# """
# scripts/validate_opt_schedule.py (闭环反馈 + 动态牵引曲线版)
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# import sys
# import glob
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 1. 环境与路径配置 =================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

# from src.physics.train_simu import TrainTheoreticalEnergyModel

# ATO_MODEL_DIR = os.path.join(project_root, "output", "models", "ato_results")
# RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
# SCHEDULE_FILE = os.path.join(project_root, "output", "schedule", "schedule_results_residual", "schedule_table.csv")
# DATA_DIR = os.path.join(project_root, "data", "data_processed")

# DT = 0.1 
# MAX_SPEED_MS = 80.0 / 3.6 

# # ================= 2. 核心控制逻辑 =================
# def get_dynamic_traction_limit(v_curr, v_req):
#     """动态物理牵引曲线：低速强、高速弱、赶点则放宽"""
#     # A. 基础物理边界
#     if v_curr < 10.0: base_limit = 1.1 # 启动阶段放宽到 1.1
#     elif v_curr < 20.0: base_limit = 1.1 - (v_curr - 10.0) * (0.5 / 10.0)
#     else: base_limit = 0.6
#     # B. 赶点激励
#     return base_limit + 0.1 if (v_req - v_curr) > 1.0 else base_limit

# class ATOPolicyNet(nn.Module):
#     def __init__(self, input_dim):
#         super(ATOPolicyNet, self).__init__()
#         self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(0.01), nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
#     def forward(self, x): return self.fc(x)

# # (此处省略位置编码和Transformer定义，与之前一致)
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         import math
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))
#     def forward(self, x): return x + self.pe[:, :x.size(1), :]

# class ResidualTransformer(nn.Module):
#     def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.input_linear = nn.Linear(input_dim, d_model); self.pos_encoder = PositionalEncoding(d_model)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layers, num_layers); self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
#     def forward(self, src): src = self.input_linear(src); src = self.pos_encoder(src); out = self.transformer(src); return self.decoder(out[:, -1, :])

# # ================= 3. 闭环仿真函数 =================
# def run_ai_drive_optimized(target_t):
#     with open(os.path.join(ATO_MODEL_DIR, "train_config.pkl"), "rb") as f: config = pickle.load(f)
#     TARGET_L = config['target_dist']; MASS_VAL = config['mass_mean']
#     with open(os.path.join(ATO_MODEL_DIR, "ato_scaler.pkl"), "rb") as f: scaler = pickle.load(f)
#     model = ATOPolicyNet(input_dim=7); model.load_state_dict(torch.load(os.path.join(ATO_MODEL_DIR, "ato_model.pth"), map_location='cpu')); model.eval()
    
#     df_map = pd.read_pickle(os.path.join(ATO_MODEL_DIR, "track_map.pkl"))
#     f_curv = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")
#     f_grad = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")

#     t, s, v = 0.0, 0.0, 0.0
#     history = {'time': [], 'dist': [], 'velocity': [], 'acceleration': []}
    
#     # 🏎️ 终极增益：如果落后，补油力度拉满
#     K_p = 2.5 
    
#     # 允许的巡航微超速 (m/s)
#     SPEED_ALLOWANCE = 0.8 # 进一步放宽限速，确保平均速度够

#     for _ in range(int(target_t * 3.0 / DT)):
#         dist_rem = max(TARGET_L - s, 0)
#         time_rem = max(target_t - t, 0.01)
#         v_req = dist_rem / time_rem # 关键：当前必须跑到的平均速度
        
#         # 1. 神经网络初步预测
#         input_data = [[v, float(f_curv(s)), float(f_grad(s)), MASS_VAL, dist_rem, time_rem, v_req]]
#         input_scaled = scaler.transform(input_data)
#         with torch.no_grad():
#             raw_acc = model(torch.tensor(input_scaled, dtype=torch.float32)).item()

#         # 2. 动态牵引能力 (更激进的启动曲线)
#         if v < 15.0: limit_a = 1.25 # 扩大恒转矩区
#         elif v < 21.0: limit_a = 1.0
#         else: limit_a = 0.65
        
#         # 3. 强力介入：如果当前速度 v 明显落后于 v_req，直接无视 AI 建议，全速追赶
#         v_error = v_req - v
#         if v_error > 0.3: # 如果慢了 0.3m/s 以上
#             final_acc = limit_a # 直接给最大力
#         else:
#             final_acc = raw_acc + (K_p * v_error) # 正常范围内的闭环修正

#         # 物理截断
#         final_acc = np.clip(final_acc, -1.3, limit_a)

#         # 4. 极限迟制动逻辑 (ATP 接管)
#         # 采用更激进的制动率 0.95，余量压缩到 0.05m
#         safe_decel = 0.95 
#         if dist_rem <= (v**2)/(2 * safe_decel) + 0.05:
#             final_acc = -(v**2)/(2 * dist_rem + 1e-9)
#             final_acc = max(final_acc, -1.35) # 紧急制动力拉满

#         # 5. 限速控制逻辑 (防止因赶点导致无限加速)
#         if v >= (MAX_SPEED_MS + SPEED_ALLOWANCE):
#             final_acc = min(final_acc, 0.0)

#         # 6. 精准停车：当距离极近且速度极低时，强制对齐终点
#         if dist_rem < 0.2 and v < 0.3:
#             s, v, final_acc = TARGET_L, 0.0, 0.0
#             history['time'].append(t); history['dist'].append(s)
#             history['velocity'].append(v); history['acceleration'].append(final_acc)
#             break

#         # 欧拉积分
#         v = max(v + final_acc * DT, 0.0); s += v * DT; t += DT
#         history['time'].append(t); history['dist'].append(s)
#         history['velocity'].append(v); history['acceleration'].append(final_acc)
        
#         if s >= TARGET_L and v <= 0.05: break

#     return pd.DataFrame(history), TARGET_L, MASS_VAL
# def compute_total_energy(station_pair, df_sim, mass_val):
#     engine = TrainTheoreticalEnergyModel()
#     e_phy_steps = engine.run_batch_simulation(df_sim['time'].values, df_sim['velocity'].values, mass_val)
#     res_path = os.path.join(RES_MODEL_BASE, station_pair)
#     if not os.path.exists(os.path.join(res_path, "best_res_model.pth")): return np.sum(e_phy_steps), np.sum(e_phy_steps)
#     with open(os.path.join(res_path, "scaler_x.pkl"), "rb") as f: sx = pickle.load(f); sy = pickle.load(open(os.path.join(res_path, "scaler_y.pkl"), "rb"))
#     res_model = ResidualTransformer(5); res_model.load_state_dict(torch.load(os.path.join(res_path, "best_res_model.pth"), map_location='cpu')); res_model.eval()
#     a_seq = np.zeros(len(df_sim)); a_seq[1:] = np.diff(df_sim['velocity'].values) / DT
#     scaled_X = sx.transform(np.stack([df_sim['velocity'].values, a_seq, e_phy_steps, np.array([engine._interp_gradient(d) for d in df_sim['dist'].values]), np.full(len(df_sim), mass_val)], axis=1))
#     res_wh = []
#     for i in range(len(scaled_X)):
#         if i < 29: res_wh.append(0.0); continue
#         with torch.no_grad(): p = res_model(torch.tensor(scaled_X[i-29:i+1], dtype=torch.float32).unsqueeze(0)).item()
#         res_wh.append(sy.inverse_transform([[p]])[0][0])
#     return np.sum(e_phy_steps) + np.sum(res_wh), np.sum(e_phy_steps)

# # ================= 4. 主程序 =================
# def main():
#     plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
#     df_table = pd.read_csv(SCHEDULE_FILE)
#     target_sp = "布政-张家潭"
#     row = df_table[df_table['station_pair'] == target_sp].iloc[0]
#     t_opt, e_theory = row['t_arrival(s)'], row['energy(Wh)']

#     print(f"🚀 [闭环+动态曲线验证] {target_sp} | 目标: {t_opt:.2f}s")
#     df_sim, s_target, mass = run_ai_drive_optimized(t_opt)
#     e_total, e_phy = compute_total_energy(target_sp, df_sim, mass)
    
#     print(f"🏁 到达耗时: {df_sim['time'].iloc[-1]:.2f}s (偏差: {df_sim['time'].iloc[-1]-t_opt:+.2f}s)")
#     print(f"🔋 实际总能耗: {e_total:.2f} Wh | 调度预测: {e_theory:.2f} Wh")

#     fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
#     axes[0].plot(df_sim['time'], df_sim['velocity'], lw=2, label='AI司机(动态控制)'); axes[0].legend(); axes[0].set_title(f"108.5s 挑战方案验证 - 总能耗: {e_total:.1f}Wh")
#     axes[1].plot(df_sim['time'], df_sim['acceleration'], color='orange', label='动态加速度输出'); axes[1].legend()
#     axes[2].plot(df_sim['time'], df_sim['dist'], color='green'); axes[2].axhline(s_target, color='r', ls=':')
#     plt.tight_layout(); plt.savefig(os.path.join(ATO_MODEL_DIR, "optimized_performance.png"), dpi=300)

# if __name__ == "__main__":
#     main()