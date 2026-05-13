# -*- coding: utf-8 -*-
"""
全线版：分站、分等级输出质量-能耗敏感度图 (变量名及数据鲁棒性修正版)
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

DATA_DIR = os.path.join(project_root, "data", "data_processed")
TABLE_DIR = os.path.join(project_root, "output", "analysis", "class_tables_strict")
STD_TIME_FILE = os.path.join(TABLE_DIR, "standard_class_times.csv")
RES_MODEL_BASE = os.path.join(project_root, "output", "models", "nn_results_residual")

OUTPUT_ROOT = os.path.join(project_root, "output", "analysis", "mass_sensitivity_all_stations")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 质量梯度设定
MASS_STEPS = [215.24, 235.24, 255.24, 275.24] 
DT = 0.05
SEQ_LEN_RES = 30

# LINE5_SECTIONS = [
#     "布政-张家潭", "张家潭-同德路", "同德路-石碶", "石碶-雅渡", "雅渡-庙堰",
#     "庙堰-钟公庙", "钟公庙-鄞州区政府", "鄞州区政府-钱湖南路", "钱湖南路-南高教园区",
#     "南高教园区-下应路", "下应路-大洋江", "大洋江-泗港", "泗港-曹隘", "曹隘-柳隘",
#     "柳隘-海晏北路", "海晏北路-民安东路", "民安东路-会展中心", "会展中心-院士路",
#     "院士路-盎孟港", "盎孟港-三官堂", "三官堂-兴庄路", "兴庄路-兴海南路",
#     "兴海南路-梅堰", "梅堰-永茂路", "永茂路-镇海大道", "镇海大道-骆驼桥"
# ]
LINE5_SECTIONS = [
    "梅堰-永茂路"
]

# ================= 2. 网络定义 =================
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

# ================= 3. 核心计算函数 =================

def generate_vt_profile(target_t, target_l):
    """生成理想轨迹"""
    acc_val, dec_val = 0.75, -0.75
    k = 0.5 * (1/acc_val + 1/abs(dec_val))
    roots = np.roots([k, -target_t, target_l])
    v_peak = min(r for r in roots if r > 0)
    t_acc, t_dec = v_peak/acc_val, abs(v_peak/dec_val)
    t_cruise = max(0, target_t - t_acc - t_dec)
    
    t_arr = np.arange(0, target_t + DT, DT)
    v_list, a_list = [], []
    for tt in t_arr:
        if tt < t_acc: 
            v_list.append(acc_val * tt); a_list.append(acc_val)
        elif tt < t_acc + t_cruise: 
            v_list.append(v_peak); a_list.append(0.0)
        else:
            dt_dec = tt - (t_acc + t_cruise)
            v_list.append(max(v_peak + dec_val * dt_dec, 0.0)); a_list.append(dec_val)
    
    v_arr = np.array(v_list)
    s_arr = np.cumsum(v_arr * DT)
    # 🌟 修复点：确保归一化时变量名一致 (s_arr vs s_array)
    return t_arr, v_arr, np.array(a_list), s_arr * (target_l / s_arr[-1])

def predict_energy(t_ref, v_ref, a_ref, s_ref, mass_val, f_grad, res_model, sx, sy):
    """计算物理+残差能耗"""
    engine = TrainTheoreticalEnergyModel()
    e_phy = engine.run_batch_simulation(t_ref, v_ref, mass_val)
    grad_seq = f_grad(s_ref)
    raw_X = np.stack([v_ref, a_ref, e_phy, grad_seq, np.full_like(v_ref, mass_val)], axis=1)
    scaled_X = sx.transform(raw_X)
    res_preds = []
    for i in range(len(scaled_X)):
        if i < SEQ_LEN_RES - 1: res_preds.append(0.0); continue
        win = torch.tensor(scaled_X[i-SEQ_LEN_RES+1 : i+1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): p_scaled = res_model(win).item()
        res_preds.append(sy.inverse_transform([[p_scaled]])[0][0])
    return np.cumsum(e_phy + np.array(res_preds))

# ================= 4. 主程序 =================

def main():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
    if not os.path.exists(STD_TIME_FILE):
        print(f"❌ 找不到标准时间表 {STD_TIME_FILE}"); return
    df_lookup = pd.read_csv(STD_TIME_FILE).set_index('区段')

    for sp in LINE5_SECTIONS:
        print(f"\n▶️ 处理区间: {sp}")
        station_out_dir = os.path.join(OUTPUT_ROOT, sp)
        res_dir = os.path.join(RES_MODEL_BASE, sp)
        if not os.path.exists(os.path.join(res_dir, "best_res_model.pth")): continue
            
        try:
            with open(os.path.join(res_dir, "scaler_x.pkl"), "rb") as f: sx = pickle.load(f)
            with open(os.path.join(res_dir, "scaler_y.pkl"), "rb") as f: sy = pickle.load(f)
            res_model = ResidualTransformer(5)
            res_model.load_state_dict(torch.load(os.path.join(res_dir, "best_res_model.pth"), map_location='cpu'))
            res_model.eval()

            # 数据清洗：处理 float/str 混合报错
            data_file = os.path.join(DATA_DIR, f"results_{sp}.xlsx")
            df_data = pd.read_excel(data_file)
            df_data['累计位移(m)'] = pd.to_numeric(df_data['累计位移(m)'], errors='coerce')
            df_data['gradient'] = pd.to_numeric(df_data['gradient'], errors='coerce')
            df_data = df_data.dropna(subset=['累计位移(m)', 'gradient']).sort_values('累计位移(m)')

            f_grad = interp1d(df_data['累计位移(m)'], df_data['gradient'], kind='nearest', fill_value="extrapolate")
            actual_l = df_data['累计位移(m)'].max()
            
            if pd.isna(actual_l) or actual_l <= 0: continue

        except Exception as e:
            print(f"   ❌ 资源加载或清洗失败: {e}"); continue

        os.makedirs(station_out_dir, exist_ok=True)

        for i in range(1, 6):
            class_key = f"Class{i}"
            if class_key not in df_lookup.columns or pd.isna(df_lookup.loc[sp, class_key]): continue
            
            t_target = float(df_lookup.loc[sp, class_key])
            t_ref, v_ref, a_ref, s_ref = generate_vt_profile(t_target, actual_l)
            
            plt.figure(figsize=(10, 6))
            for m in MASS_STEPS:
                cum_energy = predict_energy(t_ref, v_ref, a_ref, s_ref, m, f_grad, res_model, sx, sy)
                plt.plot(s_ref, cum_energy, label=f"质量: {m}t", lw=2)
            
            plt.title(f"{sp} - 等级 {i} (目标:{t_target}s) 质量敏感度分析", fontsize=14)
            plt.xlabel("行驶里程 (m)"); plt.ylabel("累积总能耗 (Wh)")
            plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(station_out_dir, f"class_{i}.png")
            plt.savefig(save_path, dpi=200); plt.close()
        
        print(f"   ✅ {sp} 处理完成")

    print(f"\n🎉 任务完成！")

if __name__ == "__main__":
    main()