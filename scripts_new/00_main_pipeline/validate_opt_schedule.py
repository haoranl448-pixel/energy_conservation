# -*- coding: utf-8 -*-
"""
scripts/validate_opt_schedule.py (AI 风格还原 + 综合能耗结算版)
逻辑：
1. 还原 AI 驾驶风格，保留加速度波动，并与真实历史数据对比。
2. 集成物理引擎 (Physics) 与残差模型 (Transformer) 计算总能耗。
3. 自动匹配时刻表目标时间。
"""
# os/glob：定位模型、时刻表和历史数据文件。
import os
import glob
# pickle：读取训练时保存的 scaler、配置和线路 map。
import pickle
# numpy/pandas：数值计算和表格处理。
import numpy as np
import pandas as pd
# torch：加载 ATO policy 和残差 Transformer 模型。
import torch
import torch.nn as nn
# matplotlib：输出验证对比图；Agg 后端适合无界面环境保存图片。
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
# interp1d：把轨道坡度/曲率按位移插值到仿真点。
from scipy.interpolate import interp1d
# sys：把项目根目录加入 import 路径。
import sys
# warnings：屏蔽非关键警告。
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径与环境配置 =================
# 当前脚本目录。
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录；在 scripts_new 二级目录下直接运行时需要注意路径层级。
project_root = os.path.dirname(current_dir)
# 允许导入 src.physics 等项目模块。
sys.path.append(project_root)

# 导入物理模型
from src.physics.train_simu import TrainTheoreticalEnergyModel

# 处理后的历史数据目录，用于读取真实运行曲线。
DATA_DIR = os.path.join(project_root, "data_processed")
# ATO policy 模型目录。
ATO_MODEL_DIR = os.path.join(project_root, "output", "models", "ato_results")
# 残差模型目录，通常按站间区间分子文件夹。
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
# DP 排图输出的 schedule_table。
SCHEDULE_FILE = os.path.join(project_root, "output", "schedule", "schedule_results_residual", "schedule_table.csv")

# 仿真步长，单位秒。
DT = 0.1 
# 最大限速 80 km/h，换算为 m/s。
MAX_SPEED_MS = 80.0 / 3.6 
# 残差模型要求的滑动窗口长度。
SEQ_LEN_RES = 30 # 残差模型要求的窗口长度

# ================= 2. 网络定义 (需与训练时严格一致) =================

class ATOPolicyNet(nn.Module):
    """ATO 控车策略网络：输入状态特征，输出当前加速度。"""

    def __init__(self, input_dim):
        super(ATOPolicyNet, self).__init__()
        # 三层全连接网络，输出一个加速度值。
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
    def forward(self, x):
        # LeakyReLU 避免负半轴完全失活。
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.01)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.01)
        return self.output(x)

class PositionalEncoding(nn.Module):
    """Transformer 位置编码，与残差模型训练结构保持一致。"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import math
        # 构造 [max_len, d_model] 的正弦/余弦位置编码。
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，随模型保存/加载，但不训练。
        self.register_buffer('pe', pe.unsqueeze(0))
    # 输入形状通常为 [batch, seq_len, d_model]。
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ResidualTransformer(nn.Module):
    """残差能耗模型：预测物理模型与实际能耗之间的偏差。"""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 输入特征映射到 Transformer 隐空间。
        self.input_linear = nn.Linear(input_dim, d_model)
        # 加入时序位置信息。
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer Encoder 与训练权重结构一致。
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        # 解码成单个残差值。
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, src):
        # 特征升维。
        src = self.input_linear(src)
        # 加位置编码。
        src = self.pos_encoder(src)
        # Transformer 编码完整时间窗口。
        out = self.transformer(src)
        # 取最后一个时间步预测当前残差。
        return self.decoder(out[:, -1, :])

# ================= 3. 核心计算模块 =================

def run_ai_drive(target_t):
    """根据目标运行时间，用 ATO policy 模型仿真一条速度曲线。"""

    # 读取训练时保存的目标区间长度、平均载重等配置。
    with open(os.path.join(ATO_MODEL_DIR, "train_config.pkl"), "rb") as f:
        config = pickle.load(f)
    TARGET_L = config['target_dist']
    MASS_VAL = config['mass_mean']

    # 读取 ATO policy 输入归一化器。
    with open(os.path.join(ATO_MODEL_DIR, "ato_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    # 模型输入维度来自 scaler。
    model = ATOPolicyNet(len(scaler.mean_))
    # 加载训练好的 ATO policy 权重。
    model.load_state_dict(torch.load(os.path.join(ATO_MODEL_DIR, "ato_model.pth"), map_location='cpu'))
    model.eval()

    # 读取线路坡度/曲率 map，用于构造当前状态特征。
    df_map = pd.read_pickle(os.path.join(ATO_MODEL_DIR, "track_map.pkl"))
    f_curv = interp1d(df_map['累计位移(m)'], df_map['curvature'], kind='nearest', fill_value="extrapolate")
    f_grad = interp1d(df_map['累计位移(m)'], df_map['gradient'], kind='nearest', fill_value="extrapolate")

    # 初始化仿真状态：时间、位移、速度。
    t, s, v = 0.0, 0.0, 0.0
    # 保存仿真过程，用于后续能耗计算和画图。
    history = {'time': [], 'dist': [], 'velocity': [], 'acceleration': []}
    
    # 最大循环步数给 target_t 留了 2 倍裕度，避免模型未及时停车。
    for _ in range(int(target_t * 2.0 / DT)):
        # 剩余距离和剩余时间是控车模型的重要状态。
        dist_rem, time_rem = TARGET_L - s, target_t - t
        # 构造模型输入：速度、曲率、坡度、载重、剩余距离、剩余时间。
        input_scaled = scaler.transform([[v, float(f_curv(s)), float(f_grad(s)), MASS_VAL, dist_rem, time_rem]])
        # 推理阶段关闭梯度。
        with torch.no_grad():
            raw_acc = model(torch.tensor(input_scaled, dtype=torch.float32)).item()

        # 物理截断与安全兜底：限制牵引/制动加速度。
        final_acc = np.clip(raw_acc, -1.2, 1.0) 
        # 按当前速度估算停车所需距离。
        req_dist = (v**2) / (2 * 0.8) 
        # 接近终点时强制进入制动逻辑，避免冲过终点。
        if dist_rem <= req_dist + 2.0:
            needed_decel = -(v**2) / (2 * dist_rem + 0.1)
            final_acc = max(needed_decel, -1.2)
        # 超过限速时禁止继续加速。
        if v >= MAX_SPEED_MS: final_acc = min(final_acc, 0.0)
        # 静止附近不允许继续给负加速度。
        if v <= 0.01 and final_acc < 0: final_acc = 0.0

        # 欧拉积分更新速度、位移、时间。
        v = max(v + final_acc * DT, 0.0); s += v * DT; t += DT
        # 记录当前仿真点。
        history['time'].append(t); history['dist'].append(s)
        history['velocity'].append(v); history['acceleration'].append(final_acc)
        # 到达终点且速度接近 0，认为本次仿真结束。
        if s >= TARGET_L and v <= 0.05: break

    # 返回仿真轨迹、目标距离和载重。
    return pd.DataFrame(history), TARGET_L, MASS_VAL

def compute_total_energy_fusion(station_pair, df_sim, mass_val):
    """对仿真轨迹进行物理能耗 + 残差能耗结算。"""

    # 创建物理能耗模型。
    sim_engine = TrainTheoreticalEnergyModel()
    # 提取时间和速度序列。
    t_seq, v_seq = df_sim['time'].values, df_sim['velocity'].values
    # A. 物理基准能耗。
    e_phy_steps = sim_engine.run_batch_simulation(t_seq, v_seq, mass_val) # Wh
    
    # B. 残差预测能耗。
    res_path = os.path.join(RES_MODEL_BASE, station_pair)
    # 如果该区间没有残差模型，则退化为纯物理能耗。
    if not os.path.exists(os.path.join(res_path, "best_res_model.pth")):
        return np.sum(e_phy_steps), np.sum(e_phy_steps)

    # 读取残差模型输入/输出 scaler。
    with open(os.path.join(res_path, "scaler_x.pkl"), "rb") as f: scaler_x = pickle.load(f)
    with open(os.path.join(res_path, "scaler_y.pkl"), "rb") as f: scaler_y = pickle.load(f)
    # 构建并加载残差 Transformer。
    res_model = ResidualTransformer(input_dim=5)
    res_model.load_state_dict(torch.load(os.path.join(res_path, "best_res_model.pth"), map_location='cpu'))
    res_model.eval()

    # 加速度由速度差分得到。
    a_seq = np.zeros_like(v_seq)
    a_seq[1:] = np.diff(v_seq) / DT
    # 位移序列用于从物理模型内部插值坡度。
    dist_seq = df_sim['dist'].values
    grad_seq = np.array([sim_engine._interp_gradient(d) for d in dist_seq])
    
    # 残差模型输入特征：速度、加速度、物理步能耗、坡度、载重。
    raw_X = np.stack([v_seq, a_seq, e_phy_steps, grad_seq, np.full_like(v_seq, mass_val)], axis=1)
    # 使用训练时 scaler 归一化。
    scaled_X = scaler_x.transform(raw_X)

    # 逐时间点构造滑动窗口预测残差。
    res_wh = []
    for i in range(len(scaled_X)):
        # 前 29 个点没有完整窗口，残差置 0。
        if i < SEQ_LEN_RES - 1:
            res_wh.append(0.0); continue
        # 取最近 30 步作为 Transformer 输入。
        window = torch.tensor(scaled_X[i-SEQ_LEN_RES+1:i+1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = res_model(window).item()
        # 输出反归一化为 Wh。
        res_wh.append(scaler_y.inverse_transform([[p]])[0][0])
    
    # 返回融合总能耗和物理基准能耗。
    return np.sum(e_phy_steps) + np.sum(res_wh), np.sum(e_phy_steps)

# ================= 4. 主程序 =================

def main():
    """读取排图方案，仿真一个目标区间，并输出验证图和能耗报告。"""

    # 配置中文字体和负号显示。
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
    
    # 1. 加载时刻表
    if not os.path.exists(SCHEDULE_FILE):
        print(f"❌ 找不到时刻表: {SCHEDULE_FILE}"); return
    df_table = pd.read_csv(SCHEDULE_FILE)
    # 当前脚本只验证一个目标区间，可按需要改成循环。
    target_sp = "布政-张家潭"
    # 从排图表中取该区间的目标运行时间和 DP 预测能耗。
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
    # 如果有真实数据，就在图里叠加历史曲线；没有则只画仿真曲线。
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

    # 6. 绘图对比：速度、加速度、位移三行图。
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
    # 保存验证图到 ATO 模型目录。
    save_p = os.path.join(ATO_MODEL_DIR, "validation_full_energy_report.png")
    plt.savefig(save_p, dpi=300)
    print(f"✅ 综合报告图表已保存: {save_p}")

if __name__ == "__main__":
    # 直接运行脚本时执行验证流程。
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
