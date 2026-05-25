# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 环境与路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.physics.train_simu import TrainTheoreticalEnergyModel

# 字体与清晰度
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.antialiased'] = True

DATA_DIR = os.path.join(project_root, "data", "data_processed")
MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")
OUTPUT_DIR = os.path.join(project_root, "output", "analysis", "full_line_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

SEQ_LEN = 30

# ================= 2. 模型定义 (Transformer 5特征版) =================
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
    def __init__(self, input_dim=5):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, 64)
        self.pos_encoder = PositionalEncoding(64) # 🌟 匹配 pos_encoder 子模块名
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2)
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, src):
        x = self.pos_encoder(self.input_linear(src))
        return self.decoder(self.transformer(x)[:, -1, :])

# ================= 3. 核心绘图函数 =================

def process_station_plot(sp):
    data_path = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
    res_dir = os.path.join(MODEL_BASE, sp)
    
    if not os.path.exists(data_path) or not os.path.exists(os.path.join(res_dir, "best_res_model.pth")):
        print(f"⚠️ 跳过 {sp}: 缺少数据或模型")
        return

    # A. 加载与清洗数据
    df = pd.read_excel(data_path)
    # 选取该站的第一趟完整记录
    seg_id = df['segment'].unique()[0]
    group = df[df['segment'] == seg_id].sort_values('时刻').copy()
    
    # 强制数值化
    for c in ['速度(m/s)', '累计位移(m)', '时刻', 'energy', 'gradient', '加速度(m/s²)']:
        group[c] = pd.to_numeric(group[c], errors='coerce')
    group = group.dropna(subset=['时刻', '速度(m/s)', '累计位移(m)'])

    # 🌟 零点对齐
    group['时刻'] = group['时刻'] - group['时刻'].iloc[0]
    group['累计位移(m)'] = group['累计位移(m)'] - group['累计位移(m)'].iloc[0]
    # 强制速度初始点为 0
    v_kmh = group['速度(m/s)'].values * 3.6
    v_kmh[0] = 0.0 
    
    t_seq, v_ms, s_m, mass_val = group['时刻'].values, group['速度(m/s)'].values, group['累计位移(m)'].values, group['重量'].iloc[0]

    # B. 物理与残差预测
    sim_model = TrainTheoreticalEnergyModel()
    e_phy_steps = sim_model.run_batch_simulation(t_seq, v_ms, mass_val)
    
    with open(f"{res_dir}/scaler_x.pkl", "rb") as f: sx = pickle.load(f)
    with open(f"{res_dir}/scaler_y.pkl", "rb") as f: sy = pickle.load(f)
    model = ResidualTransformer(5)
    model.load_state_dict(torch.load(f"{res_dir}/best_res_model.pth", map_location='cpu'))
    model.eval()

    raw_X = np.stack([v_ms, group['加速度(m/s²)'].values, e_phy_steps, group['gradient'].values, np.full_like(v_ms, mass_val)], axis=1)
    scaled_X = sx.transform(raw_X)
    
    res_preds = []
    for i in range(len(scaled_X)):
        if i < SEQ_LEN - 1: res_preds.append(0.0); continue
        win = torch.FloatTensor(scaled_X[i-SEQ_LEN+1:i+1]).unsqueeze(0)
        with torch.no_grad(): p = model(win).item()
        res_preds.append(sy.inverse_transform([[p]])[0][0])
    
    # 能耗结算 (Wh)
    e_real_cum = (group['cumulative_energy'].values / 3600.0)
    e_real_cum = e_real_cum - e_real_cum[0] # 相对起点的累积能耗
    e_fusion_cum = np.cumsum(e_phy_steps + np.array(res_preds))
    
    # 计算 R2
    r2 = r2_score(e_real_cum[SEQ_LEN:], e_fusion_cum[SEQ_LEN:])

    # C. 绘图
    fig, ax_v = plt.subplots(figsize=(13, 8))
    ax_e, ax_s = ax_v.twinx(), ax_v.twinx()
    ax_s.spines['right'].set_position(('outward', 65))

    # 绘制曲线
    l1, = ax_v.plot(t_seq, v_kmh, color='#3498db', lw=2.5, label='运行速度 (km/h)')
    l2, = ax_e.plot(t_seq, e_real_cum, color='black', lw=2, label='实测累积能耗 (Wh)')
    l3, = ax_e.plot(t_seq, e_fusion_cum, color='#e74c3c', lw=2, ls='--', label='残差融合预测能耗 (Wh)')
    l4, = ax_s.plot(t_seq, s_m, color='#2ecc71', lw=2.5, label='累计位移 (m)')

    # 修饰
    ax_v.set_xlabel("时间 Time (s)", fontsize=11)
    ax_v.set_ylabel("速度 Velocity (km/h)", color='black', fontsize=11)
    ax_e.set_ylabel("能耗 Energy (Wh)", color='black', fontsize=11)
    ax_s.set_ylabel("位移 Distance (m)", color='black', fontsize=11)
    
    ax_v.set_title(f"区间能耗大模型对标图: {sp}\n(R² = {r2:.5f})", fontsize=14, pad=15)
    ax_v.legend([l1, l2, l3, l4], [line.get_label() for line in [l1, l2, l3, l4]], loc='upper left')
    ax_v.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Validation_{sp}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return r2

# ================= 4. 主程序 =================

if __name__ == "__main__":
    print(f"🚀 开始全线 26 站能耗模型对标...")
    r2_list = []
    for sp in LINE5_SECTIONS:
        print(f"▶️ 正在对标: {sp}...", end="", flush=True)
        try:
            r2 = process_station_plot(sp)
            if r2:
                print(f" ✅ R²={r2:.5f}")
                r2_list.append(r2)
            else:
                print(" ❌ 失败")
        except Exception as e:
            print(f" ❌ 报错: {e}")

    if r2_list:
        print(f"\n🏁 全线对标完成！平均 R² = {np.mean(r2_list):.5f}")
        print(f"📂 所有对标图已存入: {OUTPUT_DIR}")