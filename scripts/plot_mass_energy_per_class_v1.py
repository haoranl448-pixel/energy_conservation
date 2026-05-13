# -*- coding: utf-8 -*-
"""
功能：横坐标为质量(215.24-290t)，纵坐标为总能耗(kWh)
分度值：1.0t (提速平衡版)
不同 Class 对应不同曲线，每个站出一张汇总图。
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 环境与路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.physics.train_simu import TrainTheoreticalEnergyModel

# 输入路径
DATA_DIR = os.path.join(project_root, "data", "data_processed")
TABLE_DIR = os.path.join(project_root, "output", "analysis", "class_tables_strict")
STD_TIME_FILE = os.path.join(TABLE_DIR, "standard_class_times.csv")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")

# 输出目录
OUTPUT_ROOT = os.path.join(project_root, "output", "analysis", "energy_mass_curves")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 核心参数修正
MASS_RANGE = np.arange(215.24, 240.0 + 1.0, 1.0) # 修改为 1.0t 分度
DT = 0.05
SEQ_LEN_RES = 30

LINE5_SECTIONS = [
    "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
    "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
    "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
    "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
    "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
    "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
]

# ================= 2. Transformer 模型结构定义 =================
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
        self.input_linear = nn.Linear(input_dim, d_model); self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.LeakyReLU(0.01), nn.Linear(32, 1))
    def forward(self, src):
        src = self.input_linear(src); src = self.pos_encoder(src)
        out = self.transformer(src); return self.decoder(out[:, -1, :])

# ================= 3. 计算引擎 =================

def generate_vt_profile(target_t, target_l):
    """理想梯形轨迹生成器"""
    a_acc, a_dec = 0.75, -0.75
    k = 0.5 * (1/a_acc + 1/abs(a_dec))
    roots = np.roots([k, -target_t, target_l])
    v_peak = min(r for r in roots if r > 0)
    t_acc, t_dec = v_peak/a_acc, abs(v_peak/a_dec)
    t_cruise = max(0, target_t - t_acc - t_dec)
    t_arr = np.arange(0, target_t + DT, DT)
    v_list, a_list = [], []
    for tt in t_arr:
        if tt < t_acc: v_list.append(a_acc * tt); a_list.append(a_acc)
        elif tt < t_acc + t_cruise: v_list.append(v_peak); a_list.append(0.0)
        else: v_list.append(max(v_peak + a_dec * (tt - t_acc - t_cruise), 0.0)); a_list.append(a_dec)
    v_arr = np.array(v_list)
    s_arr = np.cumsum(v_arr * DT)
    return t_arr, v_arr, np.array(a_list), s_arr * (target_l / s_arr[-1])

def get_final_energy_kwh(mass, t_ref, v_ref, a_ref, s_ref, f_grad, engine, res_model, sx, sy):
    """物理功 + Transformer 残差预测，返回终点 kWh"""
    # 1. 物理部分
    e_phy = engine.run_batch_simulation(t_ref, v_ref, mass)
    
    # 2. 残差部分 (Batch 处理优化)
    grad_seq = f_grad(s_ref)
    raw_X = np.stack([v_ref, a_ref, e_phy, grad_seq, np.full_like(v_ref, mass)], axis=1)
    scaled_X = sx.transform(raw_X)
    
    windows = []
    for i in range(SEQ_LEN_RES-1, len(scaled_X)):
        windows.append(scaled_X[i-SEQ_LEN_RES+1 : i+1])
    
    res_wh_sum = 0.0
    if windows:
        # 批量预测提速
        input_tensors = torch.tensor(np.array(windows), dtype=torch.float32)
        with torch.no_grad():
            preds_scaled = res_model(input_tensors).cpu().numpy()
        res_wh_sum = np.sum(sy.inverse_transform(preds_scaled))
        
    return (np.sum(e_phy) + res_wh_sum) / 1000.0

# ================= 4. 主循环 =================

def main():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
    df_lookup = pd.read_csv(STD_TIME_FILE).set_index('区段')
    engine = TrainTheoreticalEnergyModel()

    for sp in LINE5_SECTIONS:
        print(f"▶️ 处理站间区间: {sp}")
        res_dir = os.path.join(RES_MODEL_BASE, sp)
        if not os.path.exists(os.path.join(res_dir, "best_res_model.pth")): continue
        
        # 加载资源
        with open(os.path.join(res_dir, "scaler_x.pkl"), "rb") as f: sx = pickle.load(f)
        with open(os.path.join(res_dir, "scaler_y.pkl"), "rb") as f: sy = pickle.load(f)
        res_model = ResidualTransformer(5)
        res_model.load_state_dict(torch.load(os.path.join(res_dir, "best_res_model.pth"), map_location='cpu'))
        res_model.eval()

        data_file = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
        df_data = pd.read_excel(data_file)
        f_grad = interp1d(df_data['累计位移(m)'], df_data['gradient'], kind='nearest', fill_value="extrapolate")
        actual_l = df_data['累计位移(m)'].max()

        plt.figure(figsize=(11, 7))
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

        for i in range(1, 6):
            class_key = f"Class{i}"
            if class_key not in df_lookup.columns: continue
            
            t_target = float(df_lookup.loc[sp, class_key])
            t_ref, v_ref, a_ref, s_ref = generate_vt_profile(t_target, actual_l)
            
            print(f"   [Class {i}] 计算中...", end='', flush=True)
            energy_points = [get_final_energy_kwh(m, t_ref, v_ref, a_ref, s_ref, f_grad, engine, res_model, sx, sy) for m in MASS_RANGE]
            
            plt.plot(MASS_RANGE, energy_points, label=f"等级 {i} ({t_target}s)", color=colors[i-1], lw=2.2, marker='o', markersize=3)
            print(" 完成")

        plt.title(f"{sp} 总能耗-质量敏感度全景分析 (1.0t 分度)", fontsize=14)
        plt.xlabel("列车总质量 (t)", fontsize=12); plt.ylabel("单趟行程总能耗 (kWh)", fontsize=12)
        plt.legend(); plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_ROOT, f"{sp}_energy_vs_mass.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 图片已导出至: {save_path}\n")

if __name__ == "__main__":
    main()